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

import matplotlib.pyplot as plt

from sam2.distinctipy import get_colors

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your script")
    # Dataset type. One of ("COCO", "OCHuman", "CrowdPose", "MPII", "AIC")
    parser.add_argument("--dataset", type=str, default="COCO")

    # Dataset subset. One of ("train", "val", "test")
    parser.add_argument("--subset", type=str, default="val")

    parser.add_argument("--gt-file", type=str, default=None)

    # Number of images to process
    parser.add_argument("--num-images", type=int, default=10)

    # Number of images to process
    parser.add_argument("--conf-thr", type=float, default=0.3)

    # Number of keypoints to use
    parser.add_argument("--num-pos-keypoints", type=int, default=17)
    parser.add_argument("--num-neg-keypoints", type=int, default=17)

    parser.add_argument("--debug-folder", type=str, default="debug", help="Folder to save debug images")
    parser.add_argument("--out-filename", type=str, default="sam_masks_single")
    
    parser.add_argument("--selection-method", type=str, default="distance+confidence")
    
    # Boolean flags, default to False
    parser.add_argument("--test-keypoints", action="store_true")
    parser.add_argument("--monte-carlo-search", action="store_true")
    parser.add_argument('--mask-out', action="store_true")
    parser.add_argument('--output-as-list', action="store_true")

    # Special flags for better pose2seg
    parser.add_argument("--expand-bbox", action="store_true", help="Expand bbox if any of the selected pose kpts is outside the bbox")
    parser.add_argument("--oracle", action="store_true", help="Evaluate dt mask compared to gt mask and take the best one")
    parser.add_argument("--crop", action="store_true", help="Crop the image to the 1.5x bbox size to increase the resolution")
    

    # Boolean flags, default to True
    parser.add_argument('--use-bbox', action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug-vis", action=argparse.BooleanOptionalAction)
    parser.add_argument("--update-bboxes", action=argparse.BooleanOptionalAction)
    parser.add_argument('--vis-by-name', action=argparse.BooleanOptionalAction)


    args = parser.parse_args()

    # If boolean argument is not selected, set it to True
    for arg in args.__dict__:
        if args.__dict__[arg] is None:
            args.__dict__[arg] = True 

    # Check the dataset and subset
    assert args.dataset in ["COCO", "OCHuman", "OCHuman-tiny", "CrowdPose", "MPII", "AIC"]
    assert args.subset in ["train", "val", "test"]

    args.dataset_path, args.images_root, args.subset = find_dataset_path(args.dataset, args.subset)

    return args

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

def parse_images(data):
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
        
        this_kpts = this_kpts[kpts_conf > 0.9, :]
        kpts_conf = kpts_conf[kpts_conf > 0.9]
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(this_kpts[:, :2] - bbox_center, axis=1)
        sort_idx = np.argsort(dists)
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]

    return this_kpts, kpts_conf

def process_image(args, image_data, model):
    image_path = os.path.join(args.images_root, image_data["file_name"])
    image = cv2.imread(image_path)

    bbox_ious = []
    mask_ious = []

    if image is None:
        raise ValueError("Image not found: {:s}".format(image_path))

    if not (args.crop and args.use_bbox):
        model.set_image(image)

    image_kpts = []
    for annotation in image_data["annotations"]:
        this_kpts = np.array(annotation["keypoints"]).reshape(-1, 3)
        this_kpts[this_kpts[:, 2] < args.conf_thr, :2] = 0
        image_kpts.append(this_kpts)
    image_kpts = np.array(image_kpts)

    image_masks = []
    dt_bboxes = []
    gt_bboxes = []
    gt_masks = []
    # for annotation in tqdm(image_data["annotations"], ascii=True):
    pos_kpts_for_vis = []
    for ann_idx, annotation in enumerate(image_data["annotations"]):
        bbox_xywh = annotation["bbox"]
        
        bbox_area = bbox_xywh[2] * bbox_xywh[3]
        image_area = image.shape[0] * image.shape[1]
        if (bbox_area / image_area) < 0.1:
            continue
        
        gt_mask = annotation.get("segmentation", None)
        if gt_mask is not None and len(gt_mask) > 0:
            gtm_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
            gtm_rle = Mask.merge(gtm_rle)
            gt_masks.append(gtm_rle)
        else:
            gt_masks.append(None)
            
        bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
        gt_bboxes.append(bbox_xyxy)
        this_kpts = image_kpts[ann_idx]
        other_kpts = None
        if len(image_kpts) > 1:
            other_kpts = np.concatenate([image_kpts[:ann_idx], image_kpts[ann_idx + 1:]], axis=0)


        dt_mask, pos_kpts, neg_kpts = pose2seg(
            args,
            model,
            bbox_xyxy,
            pos_kpts=this_kpts,
            neg_kpts=other_kpts,
            image = image if (args.crop and args.use_bbox) else None,
            gt_mask=gt_masks[-1]
        )
        pos_kpts_for_vis.append(pos_kpts)

        # if args.use_bbox:
        #     bbox_mask = np.zeros_like(dt_mask)
        #     bbox_xyxy_int = np.round(bbox_xyxy).astype(int)
        #     bbox_mask[bbox_xyxy_int[1]:bbox_xyxy_int[3], bbox_xyxy_int[0]:bbox_xyxy_int[2]] = 1
        #     dt_mask = np.logical_and(dt_mask, bbox_mask)

        dt_mask_rle = Mask.encode(np.asfortranarray(dt_mask.astype(np.uint8)))
        image_masks.append(dt_mask_rle)

        image_data["annotations"][ann_idx]["segmentation"] = dt_mask_rle

        if args.update_bboxes:
            dt_bbox = Mask.toBbox(dt_mask_rle).tolist()
            image_data["annotations"][ann_idx]["bbox"] = dt_bbox

        if gt_mask is not None:
            gt_mask_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
            gt_mask_rle = Mask.merge(gt_mask_rle)

            mask_iou = Mask.iou([gt_mask_rle], [dt_mask_rle], [0]).item()
            mask_ious.append(mask_iou)
        else:
            gt_mask_rle = None
            mask_iou = 0
            mask_ious.append(0)
            
        dt_bbox = Mask.toBbox(dt_mask_rle).tolist()
        dt_bboxes.append(dt_bbox)
        bbox_iou = Mask.iou([bbox_xywh], [dt_bbox], [0]).item()
        bbox_ious.append(bbox_iou)

        # # Toss a coin. If heads, visualize masks, bboxes and IoUs
        # if args.debug_vis and np.random.rand() < 1.0:
        #     visualize_masks(image, dt_mask_rle, gt_mask_rle, dt_bbox, bbox_xywh, bbox_iou, mask_iou, pos_kpts=pos_kpts, neg_kpts=neg_kpts)

    if args.debug_vis and np.random.rand() < 1.0 and len(image_masks) > 0:
        gt_bboxes = np.array(gt_bboxes)
        dt_bboxes = np.array(dt_bboxes)
        batch_visualize_masks(
            image,
            image_masks, 
            pos_kpts_for_vis,
            gt_bboxes,
            dt_bboxes,
            gt_masks,
            bbox_ious,
            mask_ious,
            image_path = image_path if args.vis_by_name else None,
            mask_out = args.mask_out
        )

    pairwise_ious = compute_pairwise_ious(image_masks)

    return image_data, bbox_ious, mask_ious, pairwise_ious, [0]

def compute_pairwise_ious(masks):
    ious = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            iou = Mask.iou([masks[i]], [masks[j]], [0]).item()
            ious.append(iou)
    ious = np.array(ious)

    return ious

def pose2seg(args, model, bbox_xyxy=None, pos_kpts=None, neg_kpts=None, image=None, gt_mask=None):
    
    # Filter-out un-annotated and invisible keypoints
    if pos_kpts is not None:
        pos_kpts = pos_kpts.reshape(-1, 3)
        valid_kpts = pos_kpts[:, 2] > args.conf_thr

        pose_bbox = np.array([pos_kpts[:, 0].min(), pos_kpts[:, 1].min(), pos_kpts[:, 0].max(), pos_kpts[:, 1].max()])
        pos_kpts, conf = select_keypoints(args, pos_kpts, num_visible=valid_kpts.sum(), bbox=bbox_xyxy)

        pos_kpts_backup = np.concatenate([pos_kpts, conf[:, None]], axis=1)

        if pos_kpts.shape[0] > args.num_pos_keypoints:
            pos_kpts = pos_kpts[:args.num_pos_keypoints, :]

    else:
        pose_bbox = None
        pos_kpts = np.empty((0, 2), dtype=np.float32)
        pos_kpts_backup = np.empty((0, 2), dtype=np.float32)

    if neg_kpts is not None:
        neg_kpts = neg_kpts.reshape(-1, 3)
        valid_kpts = neg_kpts[:, 2] > args.conf_thr

        neg_kpts, conf = select_keypoints(args, neg_kpts, num_visible=valid_kpts.sum(), bbox=bbox_xyxy, ignore_limbs=True, method="closest", pos_kpts=pos_kpts)
        selected_neg_kpts = neg_kpts
        neg_kpts_backup = np.concatenate([neg_kpts, conf[:, None]], axis=1)

        if neg_kpts.shape[0] > args.num_neg_keypoints:
            selected_neg_kpts = neg_kpts[:args.num_neg_keypoints, :]

    else:
        selected_neg_kpts = np.empty((0, 2), dtype=np.float32)
        neg_kpts_backup = np.empty((0, 2), dtype=np.float32)

    # Concatenate positive and negative keypoints
    kpts = np.concatenate([pos_kpts, selected_neg_kpts], axis=0)
    kpts_labels = np.concatenate([np.ones(pos_kpts.shape[0]), np.zeros(selected_neg_kpts.shape[0])], axis=0)


    # print(kpts.shape, kpts_labels.shape)
    # print(kpts)
    # print(kpts_labels)

    # Take only the positive keypoints
    # kpts = pos_kpts
    # kpts_labels = np.ones(kpts.shape[0])

    bbox = bbox_xyxy if args.use_bbox else None
    # bbox = pose_bbox if args.use_bbox else None

    if (args.use_bbox and args.expand_bbox):
        # Expand the bbox such that it contains all positive keypoints
        pose_bbox = np.array([pos_kpts[:, 0].min()-2, pos_kpts[:, 1].min()-2, pos_kpts[:, 0].max()+2, pos_kpts[:, 1].max()+2])
        expanded_bbox = np.array(bbox)
        expanded_bbox[:2] = np.minimum(bbox[:2], pose_bbox[:2])
        expanded_bbox[2:] = np.maximum(bbox[2:], pose_bbox[2:]) 
        # if (expanded_bbox != bbox).any():
        #     breakpoint()
        bbox = expanded_bbox

    if args.crop and args.use_bbox and image is not None:
        # Crop the image to the 1.5 * bbox size
        crop_bbox = np.array(bbox)
        bbox_center = np.array([(crop_bbox[0] + crop_bbox[2]) / 2, (crop_bbox[1] + crop_bbox[3]) / 2])
        bbox_size = np.array([crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]])
        bbox_size = 1.5 * bbox_size
        crop_bbox = np.array([bbox_center[0] - bbox_size[0] / 2, bbox_center[1] - bbox_size[1] / 2, bbox_center[0] + bbox_size[0] / 2, bbox_center[1] + bbox_size[1] / 2])
        crop_bbox = np.round(crop_bbox).astype(int)
        crop_bbox = np.clip(crop_bbox, 0, [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        original_image_size = image.shape[:2]
        image = image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]

        # Update the keypoints
        kpts = kpts - crop_bbox[:2]
        bbox[:2] = bbox[:2] - crop_bbox[:2]
        bbox[2:] = bbox[2:] - crop_bbox[:2]

        model.set_image(image)

    masks, scores, logits = model.predict(
        point_coords=kpts,
        point_labels=kpts_labels,
        box=bbox,
        multimask_output=False,
    )
    mask = masks[0]

    if args.crop and args.use_bbox and image is not None:
        # Pad the mask to the original image size
        mask_padded = np.zeros(original_image_size, dtype=np.uint8)
        mask_padded[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]] = mask
        mask = mask_padded

        bbox[:2] = bbox[:2] + crop_bbox[:2]
        bbox[2:] = bbox[2:] + crop_bbox[:2]

    if args.oracle:
        dt_mask_pose_consistency = compute_mask_pose_consistency(args, mask, pos_kpts_backup, neg_kpts_backup)
        gt_mask_binary = Mask.decode(gt_mask).astype(bool) if gt_mask is not None else None
        gt_mask_pose_consistency = compute_mask_pose_consistency(args, gt_mask_binary, pos_kpts_backup, neg_kpts_backup)

        dt_is_worse = dt_mask_pose_consistency < gt_mask_pose_consistency
        if dt_is_worse:
            mask = gt_mask_binary

    return mask, pos_kpts, neg_kpts

def compute_mask_pose_consistency(args, mask, pos_keypoints=None, neg_keypoints=None):
    if mask is None:
        return 0
    
    kpts_in_mask_mean = 0
    if pos_keypoints is not None:
        kpts = pos_keypoints
        kpts = kpts[kpts[:, 2] > 0.9, :2]
        kpts_int = np.floor(kpts).astype(int)
        kpts_int[:, 0] = np.clip(kpts_int[:, 0], 0, mask.shape[1]-1)
        kpts_int[:, 1] = np.clip(kpts_int[:, 1], 0, mask.shape[0]-1)
        kpts_in_mask = mask[kpts_int[:, 1], kpts_int[:, 0]]
        kpts_in_mask_mean = kpts_in_mask.mean() if kpts_in_mask.size > 0 else 0
    
    other_kpts_in_mask_mean = 0
    if neg_keypoints is not None:
        other_kpts = neg_keypoints
        other_kpts = other_kpts[other_kpts[:, 2] > 0.9, :2]
        other_kpts_int = np.floor(other_kpts).astype(int)
        other_kpts_int[:, 0] = np.clip(other_kpts_int[:, 0], 0, mask.shape[1]-1)
        other_kpts_int[:, 1] = np.clip(other_kpts_int[:, 1], 0, mask.shape[0]-1)
        other_kpts_in_mask = mask[other_kpts_int[:, 1], other_kpts_int[:, 0]]
        other_kpts_in_mask = ~ other_kpts_in_mask.astype(bool)
        other_kpts_in_mask_mean = other_kpts_in_mask.mean() if other_kpts_in_mask.size > 0 else 0
    
    mask_pose_consistency = kpts_in_mask_mean*0.5 + other_kpts_in_mask_mean*0.5
    return mask_pose_consistency

def pose2seg_itterative(args, model, bbox_xyxy=None, pos_kpts=None, neg_kpts=None):
    # Filter-out un-annotated and invisible keypoints
    if pos_kpts is not None:
        pos_kpts = pos_kpts.reshape(-1, 3)
        valid_kpts = pos_kpts[:, 2] > args.conf_thr

        pos_kpts = select_keypoints(args, pos_kpts, num_visible=valid_kpts.sum(), bbox=bbox_xyxy)

        if pos_kpts.shape[0] > args.num_pos_keypoints:
            pos_kpts = pos_kpts[:args.num_pos_keypoints, :]

    else:
        pos_kpts = np.empty((0, 2), dtype=np.float32)

    if neg_kpts is not None:
        neg_kpts = neg_kpts.reshape(-1, 3)
        valid_kpts = neg_kpts[:, 2] > args.conf_thr

        neg_kpts = select_keypoints(args, neg_kpts, num_visible=valid_kpts.sum(), bbox=bbox_xyxy, ignore_limbs=True, method="closest", pos_kpts=pos_kpts)

        if neg_kpts.shape[0] > args.num_neg_keypoints:
            neg_kpts = neg_kpts[:args.num_neg_keypoints, :]

    else:
        neg_kpts = np.empty((0, 2), dtype=np.float32)

    bbox = bbox_xyxy if args.use_bbox else None

    neg_kpts_int = np.round(neg_kpts).astype(int)
    pos_kpts_int = np.round(pos_kpts).astype(int)
    pos_kpts_consistent = np.ones(pos_kpts.shape[0], dtype=bool)
    neg_kpts_consistent = np.ones(neg_kpts.shape[0], dtype=bool)
    pos_kpts_consistent[0] = False

    max_iters = 5
    iters = 0
    max_consistent_mask = None
    max_consistency = 0
    while not (pos_kpts_consistent.all() and neg_kpts_consistent.all()):
        inconsistent_pos_kpts = pos_kpts[~pos_kpts_consistent]
        inconsistent_neg_kpts = neg_kpts[~neg_kpts_consistent]

        kpts = np.concatenate([inconsistent_pos_kpts, inconsistent_neg_kpts], axis=0)
        kpts_labels = np.concatenate([np.ones(inconsistent_pos_kpts.shape[0]), np.zeros(inconsistent_neg_kpts.shape[0])], axis=0)

        masks, scores, logits = model.predict(
            point_coords=kpts,
            point_labels=kpts_labels,
            box=bbox,
            multimask_output=False,
        )
        mask = masks[0]

        neg_kpts_consistent = ~(mask[neg_kpts_int[:, 1], neg_kpts_int[:, 0]]).astype(bool)
        pos_kpts_consistent = (mask[pos_kpts_int[:, 1], pos_kpts_int[:, 0]]).astype(bool)

        if pos_kpts_consistent.sum() + neg_kpts_consistent.sum() > max_consistency:
            max_consistency = pos_kpts_consistent.sum() + neg_kpts_consistent.sum()
            max_consistent_mask = mask

        iters += 1
        if iters >= max_iters:
            break

    mask = max_consistent_mask
    neg_kpts_consistent = ~(mask[neg_kpts_int[:, 1], neg_kpts_int[:, 0]]).astype(bool)
    pos_kpts_consistent = (mask[pos_kpts_int[:, 1], pos_kpts_int[:, 0]]).astype(bool)

    pos_kpts = pos_kpts[pos_kpts_consistent]
    neg_kpts = neg_kpts[neg_kpts_consistent]

    
    return mask, pos_kpts, neg_kpts

def batch_visualize_masks(image, masks_rle, image_kpts, bboxes_xyxy, dt_bboxes, gt_masks_raw, bbox_ious, mask_ious, image_path=None, mask_out=False):
    # Decode dt_masks_rle
    dt_masks = []
    for mask_rle in masks_rle:
        mask = Mask.decode(mask_rle)
        dt_masks.append(mask) 
    dt_masks = np.array(dt_masks)

    # Decode gt_masks_raw
    gt_masks = []
    for gt_mask in gt_masks_raw:
        if gt_mask is None:
            gt_masks.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
        else:
            # gt_mask_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
            # gt_mask_rle = Mask.merge(gt_mask_rle)
            gt_mask_rle = gt_mask
            mask = Mask.decode(gt_mask_rle)
            gt_masks.append(mask)
    gt_masks = np.array(gt_masks)

    # Generate random color for each mask
    if mask_out:
        dt_mask_image = dt_masks.max(axis=0)
        dt_mask_image = (~ dt_mask_image.astype(bool)).astype(np.uint8)
        dt_mask_image = cv2.resize(dt_mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        dt_mask_image = image * dt_mask_image[:, :, None]
    else:
        colors = (np.array(get_colors(dt_masks.shape[0])) * 255).astype(int)
        
        # colors = np.random.randint(0, 255, (dt_masks.shape[0], 3))
        # # Make sure no colors are too dark
        # np.clip(colors, 50, 255, out=colors)

        # Repeat masks to 3 channels
        dt_masks = np.repeat(dt_masks[:, :, :, None], 3, axis=3)
        gt_masks = np.repeat(gt_masks[:, :, :, None], 3, axis=3)

        # Colorize masks
        dt_masks = dt_masks * colors[:, None, None, :]
        gt_masks = gt_masks * colors[:, None, None, :]
            
        # Collapse masks to 3 channels
        dt_mask_image = dt_masks.max(axis=0)
        gt_mask_image = gt_masks.max(axis=0)

        # Convert to uint8
        dt_mask_image = dt_mask_image.astype(np.uint8)
        gt_mask_image = gt_mask_image.astype(np.uint8)

        # Resize masks to image size
        dt_mask_image = cv2.resize(dt_mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask_image = cv2.resize(gt_mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Add masks to image
    if not mask_out:

        dt_mask_image = cv2.addWeighted(image, 0.6, dt_mask_image, 0.4, 0)
        # Draw contours around the masks
        for mask, color in zip(dt_masks, colors):
            color = color.astype(int).tolist()

            mask = mask.astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dt_mask_image, contours, -1, color, 1)

        gt_mask_image = cv2.addWeighted(image, 0.6, gt_mask_image, 0.4, 0)

    # Draw keypoints
    if image_kpts is not None and not mask_out:
        for instance_kpts, color in zip(image_kpts, colors):
            color = tuple(color.astype(int).tolist())
            for kpt in instance_kpts:
                cv2.circle(dt_mask_image, kpt.astype(int)[:2], 3, color, -1)
                cv2.circle(gt_mask_image, kpt.astype(int)[:2], 3, color, -1)

    # Draw bboxes
    if bboxes_xyxy is not None and not mask_out:
        bboxes_xyxy = np.array(bboxes_xyxy)
        dt_bboxes = np.array(dt_bboxes)
        dt_bboxes[:, 2:] += dt_bboxes[:, :2]
        for gt_bbox, dt_bbox, color, biou in zip(bboxes_xyxy, dt_bboxes, colors, bbox_ious):
            color = tuple(color.astype(int).tolist())
            gbox = gt_bbox.astype(int)
            dbox = dt_bbox.astype(int)
            cv2.rectangle(dt_mask_image, (dbox[0], dbox[1]), (dbox[2], dbox[3]), color, 2)
            cv2.rectangle(gt_mask_image, (gbox[0], gbox[1]), (gbox[2], gbox[3]), color, 2)

            # Write IOU on th etop-left corner of the bbox
            cv2.putText(dt_mask_image, "{:.2f}".format(biou), (dbox[0], dbox[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(gt_mask_image, "{:.2f}".format(biou), (gbox[0], gbox[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image
    bbox_ious = np.array(bbox_ious)
    mask_ious = np.array(mask_ious)
    if image_path is not None:
        save_name = os.path.basename(image_path)
    else:
        save_name = "batch_bbox_{:06.2f}_mask_{:06.2f}_{:02d}kpts_{:06d}.jpg".format(
            bbox_ious.mean(), mask_ious.mean(), args.num_pos_keypoints, np.random.randint(1000000),
        )

    if mask_out:
        cv2.imwrite(os.path.join(args.debug_folder, save_name), dt_mask_image)               
    else:
        cv2.imwrite(os.path.join(args.debug_folder, save_name), np.hstack([gt_mask_image, dt_mask_image]))            

def visualize_masks(image, dt_mask_rle, gt_mask_rle, dt_bbox, gt_bbox, bbox_iou, mask_iou, pos_kpts=None, neg_kpts=None):
    dt_mask = Mask.decode(dt_mask_rle)*255

    if not gt_mask_rle:
        gt_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    else:
        gt_mask = Mask.decode(gt_mask_rle)*255

    if not dt_mask.any():
        dt_mask = np.random.randint(0, 2, (image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        cv2.circle(dt_mask_image, kpt.astype(int)[:2], 3, (255, 0, 0), -1)
        cv2.circle(gt_mask_image, kpt.astype(int)[:2], 3, (255, 0, 0), -1)

    dt_mask = cv2.resize(dt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    dt_mask = cv2.cvtColor(dt_mask * 255, cv2.COLOR_GRAY2BGR)
    dt_mask *= np.array([255, 255, 0], dtype=np.uint8)
    gt_mask = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
    gt_mask *= np.array([0, 255, 0], dtype=np.uint8)

    dt_mask = cv2.addWeighted(image, 0.5, dt_mask, 0.5, 0)
    gt_mask = cv2.addWeighted(image, 0.5, gt_mask, 0.5, 0)

    if pos_kpts is not None:
        for kpt in pos_kpts:
            cv2.circle(dt_mask, kpt.astype(int)[:2], 3, (255, 0, 0), -1)
            cv2.circle(gt_mask, kpt.astype(int)[:2], 3, (255, 0, 0), -1)
    if neg_kpts is not None:
        for kpt in neg_kpts:
            cv2.circle(dt_mask, kpt.astype(int)[:2], 3, (0, 0, 255), -1)
            cv2.circle(gt_mask, kpt.astype(int)[:2], 3, (0, 0, 255), -1)

    dt_bbox = np.array(dt_bbox).astype(np.int32)
    dt_bbox[2:] += dt_bbox[:2]
    gt_bbox = np.array(gt_bbox).astype(np.int32)
    gt_bbox[2:] += gt_bbox[:2]

    cv2.rectangle(dt_mask, (dt_bbox[0], dt_bbox[1]), (dt_bbox[2], dt_bbox[3]), (255, 255, 0), 2)
    cv2.rectangle(gt_mask, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)

    save_name = "bbox_{:06.2f}_mask_{:06.2f}_{:06d}.jpg".format(bbox_iou, mask_iou, np.random.randint(1000000))
    cv2.imwrite(os.path.join(args.debug_folder, save_name), np.hstack([gt_mask, dt_mask]))
   
def print_ious_stats(ious, iou_type):
    print("{:s} IOUs:".format(iou_type))
    print("  Mean: {:.4f}".format(ious.mean()))
    print("  Median: {:.4f}".format(np.median(ious)))
    print("  Min: {:.4f}".format(ious.min()))
    print("  Max: {:.4f}".format(ious.max()))

def monte_carlo_search(args, data, model):
    x_start = 1
    y_start = 0
    x_end = 17
    y_end = 20
    x_step = 1
    y_step = 1
    grid_size = (x_end-x_start+1, y_end-y_start+1)
    grid_size = (grid_size[0]//x_step, grid_size[1]//y_step)

    x_coords = np.arange(x_start, x_end+1, x_step)
    y_coords = np.arange(y_start, y_end+1, y_step)

    bbox_ious = np.zeros(grid_size)  # Store max values obtained from processing
    mask_ious = np.zeros(grid_size)  # Store max values obtained from processing
    pair_ious = np.zeros(grid_size)  # Store max values obtained from processing
    image_count_grid = np.zeros(grid_size, dtype=int)  # Track number of images processed
    people_count_grid = np.zeros(grid_size, dtype=int)  # Track number of people processed

    num_images_per_cell = len(data.keys())
    image_list = list(data.keys())

    # Sort images by number of annotations. The first images will have more annotations
    # Since we want multi-body problem, take the hardest images first
    image_list = sorted(image_list, key=lambda x: len(data[x]["annotations"]), reverse=True)

    total_iterations = grid_size[0] * grid_size[1] * args.num_images
    
    print("Running Monte Carlo search...")
    for iter_idx in tqdm(range(total_iterations), ascii=True):
        
        # Compute the UCB (upper confidence bound) value for each cell
        avg_bbox_ious = bbox_ious / np.maximum(people_count_grid, 1)
        variance = np.sqrt(np.log(iter_idx + 1) / np.maximum(image_count_grid, 1))
        ucb = avg_bbox_ious + 1.0 * variance

        # Choose a cell to process
        x, y = np.unravel_index(np.argmax(ucb), ucb.shape)
        
        if image_count_grid[x, y] < num_images_per_cell:
            # Process a new image for this cell
            image_id = image_list[image_count_grid[x, y]]
            image_data = data[image_id]

            args.num_pos_keypoints = x*x_step+x_start
            args.num_neg_keypoints = y*x_step+y_start

            bious, mious, pious = process_image(args, image_data, model)
            
            # Update the value as a moving average
            people_count_grid[x, y] += len(bious)
            bbox_ious[x, y] += np.sum(bious)
            mask_ious[x, y] += np.sum(mious)
            pair_ious[x, y] += np.sum(pious)
            
            # Increment the count of images processed for this cell
            image_count_grid[x, y] += 1

    bbox_ious = bbox_ious / np.maximum(people_count_grid, 1)
    mask_ious = mask_ious / np.maximum(people_count_grid, 1)
    pair_ious = pair_ious / np.maximum(people_count_grid, 1)

    return bbox_ious, mask_ious, pair_ious, image_count_grid, x_coords, y_coords

def eval_search_grid(args, bbox_ious, mask_ious, pair_ious):
    
    # Save the heatmap as an image
    plt.close("all")
    plt.figure()
    plt.imshow(bbox_ious, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Bounding Box IOU")
    plt.xlabel("Num Neg Kpts")
    plt.ylabel("Num Pos Kpts")
    # Make star marks for the best values
    best_bbox = np.unravel_index(np.argmax(bbox_ious), bbox_ious.shape)
    plt.scatter(best_bbox[1], best_bbox[0], marker="*", color="red", s=100)
    plt.savefig("debug/bbox_heatmap.png")

    plt.close("all")
    plt.figure()
    # Show with logaritmic cmap
    plt.imshow(bbox_ious[2:, 1:], cmap="hot", interpolation="nearest")
    plt.colorbar()
    # plt.xticks(ticks=np.arange(0, max_num_kpts), labels=np.arange(1, max_num_kpts+1))
    # plt.yticks(ticks=np.arange(0, max_num_kpts-1), labels=np.arange(2, max_num_kpts))
    plt.title("Bounding Box IOU")
    plt.xlabel("Num Neg Kpts")
    plt.ylabel("Num Pos Kpts")
    plt.savefig("debug/bbox_heatmap_focus.png")

    plt.close("all")
    plt.figure()
    plt.imshow(mask_ious, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Mask IOU")
    plt.xlabel("Num Neg Kpts")
    plt.ylabel("Num Pos Kpts")
    best_bbox = np.unravel_index(np.argmax(bbox_ious), bbox_ious.shape)
    plt.scatter(best_bbox[1], best_bbox[0], marker="*", color="red", s=100)
    plt.savefig("debug/mask_heatmap.png")

    plt.close("all")
    plt.figure()
    plt.imshow(mask_ious[2:, 1:], cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Mask IOU")
    plt.xlabel("Num Neg Kpts")
    plt.ylabel("Num Pos Kpts")
    plt.savefig("debug/mask_heatmap_focus.png")

    plt.close("all")
    plt.figure()
    bbox_norm = bbox_ious.flatten() - bbox_ious.min()
    bbox_norm /= bbox_norm.max()
    mask_norm = mask_ious.flatten() - mask_ious.min()
    mask_norm /= mask_norm.max()
    pair_norm = pair_ious.flatten() - pair_ious.min()
    pair_norm /= pair_norm.max()
    plt.plot(bbox_norm, label="BBoxes", color="blue")
    plt.plot(mask_norm, label="Masks", color="red")
    plt.plot(pair_norm, label="Pairwise", color="green")
    plt.legend()
    plt.grid()
    plt.title("Bounding Box and Mask IOU correlation")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized IOU")
    plt.savefig("debug/iou_correlation.png")


def main(args):
    # Load the data from the json file
    data = load_data(args.dataset, args.dataset_path, args.subset, args.gt_file)

    # Parse images with annotations for image-wise processing
    parsed_data = parse_images(data)

    # Prepare the model
    model = prepare_model(args)

    # Remove all images from the debug folder
    if args.debug_vis:
        shutil.rmtree(args.debug_folder, ignore_errors=True)
    os.makedirs(args.debug_folder, exist_ok=True)

    # Process the images
    print("Generating masks to images...")
    tmp_i = 0

    if args.monte_carlo_search:
        bbox_ious, mask_ious, pair_ious, icount, xs, ys = monte_carlo_search(args, parsed_data, model)
        eval_search_grid(args, bbox_ious, mask_ious, pair_ious)
        # Save the data for further analysis
        np.savez("debug/monte_carlo_search_{:d}.npz".format(args.num_images), bbox_ious=bbox_ious, mask_ious=mask_ious, pair_ious=pair_ious, icount=icount)
        # breakpoint()

    elif args.test_keypoints:
        max_num_kpts = 17
        num_pos_kpts = np.arange(0, max_num_kpts+1)
        num_neg_kpts = np.arange(1, max_num_kpts+1)

        bbox_test_heatmap = np.zeros((max_num_kpts+1, max_num_kpts))
        mask_test_heatmap = np.zeros((max_num_kpts+1, max_num_kpts))

        with tqdm(total=len(num_pos_kpts)*len(num_neg_kpts), ascii=True) as pbar:
            for pos_kpts in num_pos_kpts:
                for neg_kpts in num_neg_kpts:
                    args.num_pos_keypoints = pos_kpts
                    args.num_neg_keypoints = neg_kpts
                    
                    bbox_ious = []
                    mask_ious = []
                    pair_ious = []

                    tmp_i = 0
                    for image_id, image_data in parsed_data.items():
                        _, bious, mious, pious, ppts = process_image(args, image_data, model)
                        
                        bbox_ious.extend(bious)
                        mask_ious.extend(mious)
                        pair_ious.extend(pious)

                        if tmp_i > args.num_images:
                            break
                        tmp_i += 1

                    bbox_ious = np.array(bbox_ious)
                    mask_ious = np.array(mask_ious)

                    print("Num Pos Kpts: {:d}, Num Neg Kpts: {:d}".format(pos_kpts, neg_kpts))
                    print_ious_stats(bbox_ious, "Bounidng Box")
                    print_ious_stats(mask_ious, "Mask")

                    bbox_test_heatmap[pos_kpts, neg_kpts - 1] = bbox_ious.mean()
                    mask_test_heatmap[pos_kpts, neg_kpts - 1] = mask_ious.mean()

                    pbar.update()

    else:
        bbox_ious = []
        mask_ious = []
        pair_ious = []

        # Sort the images by number of annotations
        image_ids = sorted(parsed_data.keys(), key=lambda x: len(parsed_data[x]["annotations"]), reverse=True)

        if args.num_images > 0 and args.num_images < len(image_ids):
            image_ids = image_ids[:args.num_images]

        for image_id in tqdm(image_ids, ascii=True):
            image_data = parsed_data[image_id]

            if len(image_data["annotations"]) == 0:
                continue
            new_image_data, bious, mious, pious, ppts = process_image(args, image_data, model)
            
            parsed_data[image_id] = new_image_data

            bbox_ious.extend(bious)
            mask_ious.extend(mious)
            pair_ious.extend(pious)

        save_data = unparse_images(parsed_data)

        # Add script information to the data
        save_data["info"] = {
            "Description": "Masks generated by SAM2",
            "script_name": os.path.basename(__file__),
            "args": vars(args)
        }
        # Make unique hash describing script name and arguments
        hash_str = abs(hash(str(vars(args))+os.path.basename(__file__)))
        save_data["info"]["hash"] = hash_str

        if args.output_as_list:
            save_data = save_data["annotations"]

        save_data_dir = os.path.join(args.dataset_path, "sam_masks")
        os.makedirs(save_data_dir, exist_ok=True)
        with open(os.path.join(save_data_dir, "{:s}.json".format(args.out_filename)), "w") as f:
            json.dump(save_data, f)

        bbox_ious = np.array(bbox_ious)
        mask_ious = np.array(mask_ious)
        pair_ious = np.array(pair_ious)

        print_ious_stats(bbox_ious, "Bounding Box")
        print_ious_stats(mask_ious, "Mask")
        print_ious_stats(pair_ious, "Pairwise Mask")

if __name__ == "__main__":
    
    args = parse_args()
    main(args)