import os
import shutil
import argparse
import json
from tqdm import tqdm

import numpy as np
import cv2
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from pycocotools import mask as Mask

# Used for mask-pose consistency computation and to find the closest keypoint to the pose for negative keypoints
STRICT_KPT_THRESHOLD = 0.9

def compute_pairwise_ious(masks):
    ious = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            iou = Mask.iou([masks[i]], [masks[j]], [0]).item()
            ious.append(iou)
    ious = np.array(ious)

    return ious

def compute_one_mask_pose_consistency(args, mask, pos_keypoints=None, neg_keypoints=None):
    if mask is None:
        return 0
    
    kpts_in_mask_mean = 0
    if pos_keypoints is not None:
        kpts = pos_keypoints
        kpts = kpts[kpts[:, 2] > STRICT_KPT_THRESHOLD, :2]
        kpts_int = np.floor(kpts).astype(int)
        kpts_int[:, 0] = np.clip(kpts_int[:, 0], 0, mask.shape[1]-1)
        kpts_int[:, 1] = np.clip(kpts_int[:, 1], 0, mask.shape[0]-1)
        kpts_in_mask = mask[kpts_int[:, 1], kpts_int[:, 0]]
        kpts_in_mask_mean = kpts_in_mask.mean() if kpts_in_mask.size > 0 else 0
    
    other_kpts_in_mask_mean = 0
    if neg_keypoints is not None:
        other_kpts = neg_keypoints
        other_kpts = other_kpts[other_kpts[:, 2] > STRICT_KPT_THRESHOLD, :2]
        other_kpts_int = np.floor(other_kpts).astype(int)
        other_kpts_int[:, 0] = np.clip(other_kpts_int[:, 0], 0, mask.shape[1]-1)
        other_kpts_int[:, 1] = np.clip(other_kpts_int[:, 1], 0, mask.shape[0]-1)
        other_kpts_in_mask = mask[other_kpts_int[:, 1], other_kpts_int[:, 0]]
        other_kpts_in_mask = ~ other_kpts_in_mask.astype(bool)
        other_kpts_in_mask_mean = other_kpts_in_mask.mean() if other_kpts_in_mask.size > 0 else 0
    
    mask_pose_consistency = kpts_in_mask_mean*0.5 + other_kpts_in_mask_mean*0.5
    return mask_pose_consistency

def select_keypoints(args, kpts, num_visible, bbox=None, ignore_limbs=False, method=None, pos_kpts=None):
    "Implements different methods for selecting keypoints for pose2seg"

    methods = ["confidence", "distance", "distance+confidence", "closest"]
    if method is None:
        method = args.selection_method
    assert method in methods, "Unknown method for selecting keypoints"

    limbs_id = [
            0, 0, 0,        # Face
            1, 1,           # Ears
            2, 2,           # Shoulders - body
            3, 4, 3, 4,     # Arms
            7, 7,           # Hips - body
            5, 6, 5, 6,     # Legs
        ]
    limbs_id = np.array(limbs_id)

    # Select 1 keypoint from the face
    if not ignore_limbs:
        facial_kpts = kpts[:3, :]
        facial_conf = kpts[:3, 2]
        facial_point = facial_kpts[np.argmax(facial_conf)]
        if facial_point[-1] >= args.conf_thr:
            kpts = np.concatenate([facial_point[None, :], kpts[3:]], axis=0)
            limbs_id = limbs_id[2:]

    # Ignore invisible keypoints
    kpts_conf = kpts[:, 2]
    this_kpts = kpts[kpts_conf >= args.conf_thr, :2]
    if not ignore_limbs:
        limbs_id = limbs_id[kpts_conf >= args.conf_thr]
    kpts_conf = kpts_conf[kpts_conf >= args.conf_thr]

    if method == "confidence":

        # Sort by confidence
        sort_idx = np.argsort(kpts_conf[kpts_conf >= args.conf_thr])[::-1]
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]

    elif method == "distance":
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(this_kpts[:, :2] - bbox_center, axis=1)
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_inter_dist = np.min(dist_matrix, axis=1)
        sort_idx = np.argsort(dists + 3*min_inter_dist)[::-1]
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]
    
    elif method == "distance+confidence":

        # Sort by confidence
        sort_idx = np.argsort(kpts_conf[kpts_conf >= args.conf_thr])[::-1]
        confidences = kpts[sort_idx, 2]
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]
        
        # Compute distance matrix between all pairs
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)

        # First keypoint is the one with the highest confidence        
        selected_idx = [0]
        confidences[0] = -1
        for _ in range(this_kpts.shape[0] - 1):
            # Compute the distance to the closest selected keypoint
            min_dist = np.min(dist_matrix[:, selected_idx], axis=1)
            
            # Consider only keypoints with confidence in top 50%
            min_dist[confidences < np.percentile(confidences, 80)] = -1
            
            next_idx = np.argmax(min_dist)
            selected_idx.append(next_idx)
            confidences[next_idx] = -1

        this_kpts = this_kpts[selected_idx]
        kpts_conf = kpts_conf[selected_idx]

    elif method == "closest":
        
        this_kpts = this_kpts[kpts_conf > STRICT_KPT_THRESHOLD, :]
        kpts_conf = kpts_conf[kpts_conf > STRICT_KPT_THRESHOLD]
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(this_kpts[:, :2] - bbox_center, axis=1)
        sort_idx = np.argsort(dists)
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]

    return this_kpts, kpts_conf

def find_dataset_path(dataset, subset):
    path_to_home = "/datagrid/personal/purkrmir/"
    subset_name = subset.lower()

    if dataset == "COCO":
        data_root = os.path.join(path_to_home, "data/COCO/original")
        subset_name = "{:s}2017".format(subset.lower())
        image_folder = os.path.join(data_root, subset_name)
    elif dataset == "OCHuman":
        data_root = os.path.join(path_to_home, "data/OCHuman/COCO-like")
        subset_name = "{:s}2017".format(subset.lower())
        image_folder = os.path.join(data_root, subset_name)
    elif dataset == "OCHuman-tiny":
        data_root = os.path.join(path_to_home, "data/OCHuman/tiny")
        subset_name = "{:s}2017".format(subset.lower())
        image_folder = os.path.join(data_root, subset_name)
    elif dataset == "CrowdPose":
        data_root = os.path.join(path_to_home, "data/CrowdPose")
    elif dataset == "MPII":
        data_root = os.path.join(path_to_home, "data/MPII")
        image_folder = os.path.join(data_root, "images")
    elif dataset == "AIC":
        data_root = os.path.join(path_to_home, "data/AIC")
        image_folder = os.path.join(data_root, "images")
    else:
        raise ValueError("Unknown dataset")

    return data_root, image_folder, subset_name
    
def load_data(dataset_type, dataset_path, subset, gt_file=None):
    # Load the dataset

    if gt_file is not None:
        print("Loading GT file: {:s}".format(gt_file))
        with open(gt_file, "r") as f:
            data = json.load(f)
    else:
        if dataset_type.upper() in ["COCO", "OCHUMAN", "OCUMAN-TINY"]:
            with open(os.path.join(dataset_path, "annotations", "person_keypoints_{:s}.json".format(subset)), "r") as f:
                data = json.load(f)
        elif dataset_type.upper() in ["MPII", "AIC"]:
            with open(os.path.join(dataset_path, "annotations", "{:s}_{:s}.json".format(dataset_type.lower(), subset)), "r") as f:
                data = json.load(f)

            if dataset_type.upper() == "MPII":
                mpii_data = {'images': [], 'annotations': []}
                mpii_imgName2Id = {}
                for ann in data:
                    image_name = ann["image"]
                    image_id = abs(hash(image_name)) % (10 ** 8)
                    if image_name not in mpii_imgName2Id:
                        mpii_imgName2Id[image_name] = image_id
                        image = cv2.imread(os.path.join(dataset_path, "images", image_name))
                        mpii_data["images"].append({
                            "file_name": image_name,
                            "height": image.shape[0],
                            "width": image.shape[1],
                            "id": image_id,
                        })
                    
                    ann["image_id"] = mpii_imgName2Id[image_name]
                    kpts = np.array(ann["joints"]).reshape(-1, 2)
                    kptsv = np.array(ann["joints_vis"]).reshape(-1, 1) * 2

                    keypoints = np.stack([kpts[:, 0], kpts[:, 1], kptsv[:, 0]], axis=1).flatten().tolist()
                    ann["keypoints"] = keypoints
                    bbox_center = ann["center"]
                    bbox_scale = ann["scale"] * 200
                    bbox_xywh = [bbox_center[0] - bbox_scale/2, bbox_center[1] - bbox_scale/2, bbox_scale, bbox_scale]
                    ann["bbox"] = bbox_xywh
                    mpii_data["annotations"].append(ann)

                data = mpii_data

        else:
            raise ValueError("Unknown dataset type. Only COCO, OCHuman, OCHuman-tiny, MPII, AIC are supported")

    return data

def parse_images(args, data):
    # Parse images with annotations for image-wise processing
    parsed_data = {}
    print("Parsing IMAGES...")
    for image in tqdm(data["images"], ascii=True):
        image_id = image["id"]
        parsed_data[image_id] = {
            "file_name": image["file_name"],
            "height": image["height"],
            "width": image["width"],
            "annotations": []
        }

    print("Parsing ANNOTATIONS  ...")
    some_annotations = False
    for annotation in tqdm(data["annotations"], ascii=True):
        iscrowd = annotation.get("iscrowd", False)
        
        # Ignore crowd annotations
        if iscrowd:
            continue

        # Ignore annotations with no keypoints
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        vis_kpts = keypoints[:, 2] > args.conf_thr

        if vis_kpts.sum() <= 0:
            continue

        image_id = annotation["image_id"]
        parsed_data[image_id]["annotations"].append(annotation)
        some_annotations = True

    if not some_annotations:
        print("No annotations found in the dataset!")

    return parsed_data

def unparse_images(parsed_data):
    # Unparse the images for saving
    data = {
        "images": [],
        "annotations": []
    }

    print("Unparsing IMAGES and ANNOTATIONS...")
    for image_id, image_data in tqdm(parsed_data.items(), ascii=True):
        data["images"].append({
            "id": image_id,
            "file_name": image_data["file_name"],
            "height": image_data["height"],
            "width": image_data["width"]
        })

        for annotation in image_data["annotations"]:
            annotation["image_id"] = image_id
            annotation["iscrowd"] = 0
            if 'segmentation' in annotation:
                rle_mask_before = annotation["segmentation"]
                if isinstance(rle_mask_before, list):
                    continue
                binary_mask = Mask.decode(rle_mask_before)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                segmentations = [contour.astype(float).flatten().tolist() for contour in contours if contour.size >= 6]
                annotation["segmentation"] = segmentations
                if len(segmentations) <= 0:
                    del annotation["segmentation"]

            data["annotations"].append(annotation)

    return data

def prepare_model(args):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_checkpoint = os.path.join(this_dir, "..", "checkpoints", "sam2_hiera_base_plus.pt")
    model_cfg = "sam2_hiera_b+.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
    model = SAM2ImagePredictor(
        sam2,
        max_hole_area=10.0,
        max_sprinkle_area=50.0,
    )
    return model

def compute_mask_pose_consistency(args, masks, keypoints):
    mask_pose_consistency = []

    for idx in range(len(masks)):
        mask = masks[idx]
        kpts = keypoints[idx]
        other_kpts = np.concatenate([keypoints[:idx], keypoints[idx+1:]], axis=0).reshape(-1, 3)
        if mask is None:
            mask_pose_consistency.append(0)
            continue
        kpts = kpts[kpts[:, 2] > STRICT_KPT_THRESHOLD, :2]
        other_kpts = other_kpts[other_kpts[:, 2] > STRICT_KPT_THRESHOLD, :2]
        kpts_int = np.floor(kpts).astype(int)
        other_kpts_int = np.floor(other_kpts).astype(int)
        
        kpts_int[:, 0] = np.clip(kpts_int[:, 0], 0, mask.shape[1]-1)
        kpts_int[:, 1] = np.clip(kpts_int[:, 1], 0, mask.shape[0]-1)
        other_kpts_int[:, 0] = np.clip(other_kpts_int[:, 0], 0, mask.shape[1]-1)
        other_kpts_int[:, 1] = np.clip(other_kpts_int[:, 1], 0, mask.shape[0]-1)
        
        kpts_in_mask = mask[kpts_int[:, 1], kpts_int[:, 0]]
        other_kpts_in_mask = mask[other_kpts_int[:, 1], other_kpts_int[:, 0]]
        other_kpts_in_mask = ~ other_kpts_in_mask.astype(bool)
        kpts_in_mask_mean = kpts_in_mask.mean() if kpts_in_mask.size > 0 else 0
        other_kpts_in_mask_mean = other_kpts_in_mask.mean() if other_kpts_in_mask.size > 0 else 0
        mask_pose_consistency.append(kpts_in_mask_mean*0.5 + other_kpts_in_mask_mean*0.5)

    mask_pose_consistency = np.array(mask_pose_consistency)

    return mask_pose_consistency

