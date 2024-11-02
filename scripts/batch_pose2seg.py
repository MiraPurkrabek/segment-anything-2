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

DEBUG_FOLDER = "debug_batch" 

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your script")
    # Dataset type. One of ("COCO", "OCHuman", "CrowdPose", "MPII", "AIC")
    parser.add_argument("--dataset", type=str, default="COCO")

    # Dataset subset. One of ("train", "val", "test")
    parser.add_argument("--subset", type=str, default="val")

    parser.add_argument("--gt-file", type=str, default=None)

    # Number of images to process
    parser.add_argument("--num-images", type=int, default=10)

    # Number of keypoints to use
    parser.add_argument("--num-pos-keypoints", type=int, default=17)

    # Boolean flags, default to False
    parser.add_argument("--debug-vis", action="store_true")
    parser.add_argument("--test-keypoints", action="store_true")
    parser.add_argument("--monte-carlo-search", action="store_true")

    # Boolean flags, default to True
    parser.add_argument('--use-bbox', action=argparse.BooleanOptionalAction)
    parser.add_argument('--mask-out', action=argparse.BooleanOptionalAction)
    parser.add_argument('--vis-by-name', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Check the dataset and subset
    assert args.dataset in ["COCO", "OCHuman", "CrowdPose", "MPII", "AIC"]
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
        if dataset_type.upper() in ["COCO", "OCHUMAN"]:
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
            raise ValueError("Unknown dataset type. Only COCO, OCHuman, MPII, AIC are supported")

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
        vis_kpts = keypoints[:, 2] == 2
        
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
                binary_mask = Mask.decode(annotation["segmentation"])
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

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    model = SAM2ImagePredictor(
        sam2,
        max_hole_area=10.0,
        max_sprinkle_area=50.0,
    )
    return model

def select_keypoints(args, kpts, num_visible, bbox=None, inter_dist_matrix=None):
    "Implements different methods for selecting keypoints for pose2seg"

    methods = ["random", "distance", "prob_body", "body_systematic", "inter_distance", "hybrid"]
    method = "distance"

    limbs_id = [
            0, 0, 0,        # Face
            1, 1,           # Ears
            2, 2,           # Shoulders - body
            3, 4, 3, 4,     # Arms
            7, 7,           # Hips - body
            5, 6, 5, 6,     # Legs
        ]
    limbs_id = np.array(limbs_id)

    # Filter-out invisible keypoints
    limbs_id = limbs_id[kpts[:, 2] == 2]
    this_kpts = kpts[kpts[:, 2] == 2, :2]
    inter_dist_matrix = inter_dist_matrix[kpts[:, 2] == 2, :, :]

    if method == "random":
        selected_idx = np.random.choice(np.arange(this_kpts.shape[0]), num_visible, replace=False)
        selected_kpts = this_kpts[selected_idx]

    elif method == "distance":
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(this_kpts[:, :2] - bbox_center, axis=1)
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_inter_dist = np.min(dist_matrix, axis=1)
        sort_idx = np.argsort(dists + 3*min_inter_dist)[::-1]
        this_kpts = this_kpts[sort_idx, :2]
    
    elif method == "inter_distance":
        # Compute distances to the center of the bbox
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        bbox_diag = np.linalg.norm(bbox[:2] - bbox[2:])
        dists = np.linalg.norm(this_kpts[:, :2] - bbox_center, axis=1)
        norm_dists = dists / (bbox_diag/2)
        
        # Compute the distance matrix between all positive keypoints
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_inter_dist = np.min(dist_matrix, axis=1)
        norm_inter_dist = min_inter_dist / (bbox_diag)

        # Compute distance to the closest negative keypoint
        neg_dists = np.nanmin(inter_dist_matrix, axis=(1, 2))
        norm_neg_dists = neg_dists / (bbox_diag)

        # breakpoint()
        value = 4*norm_inter_dist# - norm_neg_dists
        # value = norm_dists + norm_inter_dist - norm_neg_dists
        sort_idx = np.argsort(value)[::-1]
        this_kpts = this_kpts[sort_idx, :2]

    elif method == "hybrid":
        # Compute distances to the center of the bbox
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        bbox_diag = np.linalg.norm(bbox[:2] - bbox[2:])
        
        # Compute the distance matrix between all positive keypoints
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_inter_dist = np.min(dist_matrix, axis=1)
        norm_inter_dist = min_inter_dist / (bbox_diag)

        # Compute distance to the closest negative keypoint
        neg_dists = np.nanmin(inter_dist_matrix, axis=(1, 2))
        norm_neg_dists = neg_dists / (bbox_diag)

        # breakpoint()
        value = 4*norm_inter_dist - norm_neg_dists
        # value = norm_dists + norm_inter_dist - norm_neg_dists
        sort_idx = np.argsort(value)[::-1]
        this_kpts = this_kpts[sort_idx, :2]
        limbs_id = limbs_id[sort_idx]

        selected_this_kpts = []
        selected_mask = np.zeros(this_kpts.shape[0], dtype=bool)
        
        # Try to take one facial keypoint
        for i in range(this_kpts.shape[0]):
            limb_id = limbs_id[i]
            if limb_id == 0:
                selected_this_kpts.append(this_kpts[i])
                selected_mask[i] = True
                break

        hips_taken = False
        # Try to take one hips keypoint
        for i in range(this_kpts.shape[0]):
            limb_id = limbs_id[i]
            if limb_id == 7:
                selected_this_kpts.append(this_kpts[i])
                selected_mask[i] = True
                hips_taken = True
                break
        if not hips_taken:
            for i in range(this_kpts.shape[0]):
                limb_id = limbs_id[i]
                if limb_id == 2:
                    selected_this_kpts.append(this_kpts[i])
                    selected_mask[i] = True
                    break

        # Take the rest of the keypoints in pre-sorted order
        for i in range(this_kpts.shape[0]):
            if selected_mask[i]:
                continue
            selected_this_kpts.append(this_kpts[i])
        this_kpts = np.array(selected_this_kpts)


    elif method == "prob_body":
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dists = np.linalg.norm(this_kpts[:, :2] - bbox_center, axis=1)
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_inter_dist = np.min(dist_matrix, axis=1)
        sort_idx = np.argsort(dists + 3*min_inter_dist)[::-1]
        this_kpts = this_kpts[sort_idx, :2]
        limbs_id = limbs_id[sort_idx]
        
        selected_this_kpts = []
        selected_mask = np.zeros(this_kpts.shape[0], dtype=bool)
        selected_limb_ids = np.zeros(8, dtype=bool)
        
        # Take one keypoint from each limb
        for i in range(this_kpts.shape[0]):
            limb_id = limbs_id[i]
            if selected_limb_ids[limb_id]:
                continue
            selected_this_kpts.append(this_kpts[i])
            selected_mask[i] = True

        # Take the rest of the keypoints in pre-sorted order
        for i in range(this_kpts.shape[0]):
            if selected_mask[i]:
                continue
            selected_this_kpts.append(this_kpts[i])
        this_kpts = np.array(selected_this_kpts)


        # unq, cnt = np.unique(limbs_id, return_counts=True)
        
        # index_map = np.searchsorted(unq, limbs_id)
        # prob = 1.0 / cnt
        # prob = prob[index_map]
        # prob /= prob.sum()
        # sort_idx = np.argsort(prob)[::-1]
        # this_kpts = this_kpts[sort_idx, :2]
        # selected_idx = np.random.choice(np.arange(this_kpts.shape[0]), num_visible, replace=False, p=prob)
        # this_kpts = this_kpts[selected_idx]

    elif method == "body_systematic":
        raise NotImplementedError("Not implemented yet")

    return this_kpts


def process_image(args, image_data, model):
    image_path = os.path.join(args.images_root, image_data["file_name"])
    image = cv2.imread(image_path)

    bbox_ious = []
    mask_ious = []

    if image is None:
        raise ValueError("Image not found: {:s}".format(image_path))

    model.set_image(image)

    image_kpts = []
    image_bboxes = []
    num_valid_kpts = []
    gt_masks = []
    for annotation in image_data["annotations"]:
        this_kpts = np.array(annotation["keypoints"]).reshape(-1, 3)
        num_visible = (this_kpts[:, 2] == 2).sum()
        if num_visible <= 0:
            continue
        num_valid_kpts.append(num_visible)
        image_bboxes.append(np.array(annotation["bbox"]))
        this_kpts[this_kpts[:, 2] != 2, :2] = 0
        image_kpts.append(this_kpts)
        gt_masks.append(annotation.get("segmentation", None))
    image_kpts = np.array(image_kpts)
    image_bboxes = np.array(image_bboxes)
    num_valid_kpts = np.array(num_valid_kpts)

    # Prepare keypoints such that all instances have the same number of keypoints
    # First sort keypoints by their distance to the center of the bounding box
    # If some are missing, duplicate the last one
    prepared_kpts = []

    # Compute the distance matrix between all keypoints in the image
    tmp_kpts = image_kpts[:, :, :2].reshape((-1, 2)).copy().astype(np.float32)
    num_instances = image_kpts.shape[0]
    kpts_nan = image_kpts[:, :, 2].reshape((-1, 17)) != 2
    try:
        tmp_kpts[kpts_nan.flatten(), :] = np.nan
    except:
        breakpoint()
    distance_matrix = np.linalg.norm(tmp_kpts[None, :] - tmp_kpts[:, None], axis=2).reshape((num_instances, 17, num_instances, 17))

    for i, (bbox, kpts, num_visible) in enumerate(zip(image_bboxes, image_kpts, num_valid_kpts)):
        # if i == 0:
        #     dmatrix = distance_matrix[i, :, 1:, :]
        # elif i == num_instances - 1:
        #     dmatrix = distance_matrix[i, :, :i, :]
        # else:
            
        dmatrix = np.concatenate((distance_matrix[i, :, :i, :], distance_matrix[i, :, i+1:, :]), axis=1)
        # breakpoint()
        this_kpts = select_keypoints(args, kpts, num_visible, bbox, dmatrix)

        # Duplicate the last keypoint if some are missing
        if this_kpts.shape[0] < num_valid_kpts.max():
            this_kpts = np.concatenate([this_kpts, np.tile(this_kpts[-1], (num_valid_kpts.max() - this_kpts.shape[0], 1))], axis=0)

        prepared_kpts.append(this_kpts)
    image_kpts = np.array(prepared_kpts)
    kpts_labels = np.ones(image_kpts.shape[:2])
    
    image_bboxes_xyxy = None
    if args.use_bbox:
        image_bboxes_xyxy = np.array(image_bboxes)
        image_bboxes_xyxy[:, 2:] += image_bboxes_xyxy[:, :2]

    # Threshold the number of positive keypoints
    if args.num_pos_keypoints > 0 and args.num_pos_keypoints < image_kpts.shape[1]:
        image_kpts = image_kpts[:, :args.num_pos_keypoints, :]
        kpts_labels = kpts_labels[:, :args.num_pos_keypoints]

    elif args.num_pos_keypoints == 0:
        image_kpts = None
        kpts_labels = None

    masks, scores, logits = model.predict(
        point_coords=image_kpts,
        point_labels=kpts_labels,
        box=image_bboxes_xyxy,
        multimask_output=False,
    )

    # Reshape the masks to (N, C, H, W). If the model outputs (C, H, W), add a number of masks dimension
    if len(masks.shape) == 3:
        masks = masks[None, :, :, :]

    masks = masks[:, 0, :, :]

    if masks.shape[0] != len(image_data["annotations"]):
        print("Mismatch in number of masks and annotations: {:d} vs {:d}".format(masks.shape[0], len(image_data["annotations"])))
        breakpoint()

    bbox_ious = []
    mask_ious = []
    pos_pts_in = []
    
    image_masks = []
    dt_bboxes = []
    for instance_i in range(masks.shape[0]):
        mask = masks[instance_i]
        bbox_xywh = image_bboxes[instance_i]
        bbox_xyxy = bbox_xywh.copy()
        bbox_xyxy[2:] += bbox_xyxy[:2]
        gt_mask = gt_masks[instance_i]

        # Mask prediction with GT bbox
        if args.use_bbox:
            bbox_int = np.round(bbox_xyxy).astype(int)
            bbox_mask = np.zeros_like(mask)
            bbox_mask[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]] = 1
            mask = np.logical_and(mask, bbox_mask)

        dt_mask_rle = Mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        image_masks.append(dt_mask_rle)

        image_data["annotations"][instance_i]["segmentation"] = dt_mask_rle
        image_data["annotations"][instance_i]["score"] = float(scores[instance_i])

        dt_bbox = Mask.toBbox(dt_mask_rle).tolist()
        dt_bboxes.append(dt_bbox)
        bbox_iou = Mask.iou([bbox_xywh], [dt_bbox], [0]).item()
        bbox_ious.append(bbox_iou)

        if image_kpts is not None:
            instance_kpts = image_kpts[instance_i]
            instance_kpts_int = np.floor(instance_kpts).astype(int)
            instance_kpts_int[:, 0] = np.clip(instance_kpts_int[:, 0], 0, mask.shape[1]-1)
            instance_kpts_int[:, 1] = np.clip(instance_kpts_int[:, 1], 0, mask.shape[0]-1)
            instance_kpts_in_mask = mask[instance_kpts_int[:, 1], instance_kpts_int[:, 0]]
            pos_pts_in.append(instance_kpts_in_mask.mean())
        else:
            pos_pts_in.append(0)

        if gt_mask is not None:
            gt_mask_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
            gt_mask_rle = Mask.merge(gt_mask_rle)


            mask_iou = Mask.iou([gt_mask_rle], [dt_mask_rle], [0]).item()
            mask_ious.append(mask_iou)
        else:
            mask_iou = 0
            mask_ious.append(0)

    if args.debug_vis and np.random.rand() < 1.0:
        batch_visualize_masks(
            image,
            image_masks,
            image_kpts,
            image_bboxes_xyxy,
            dt_bboxes,
            gt_masks,
            bbox_ious,
            mask_ious,
            image_path = image_path if args.vis_by_name else None,
            mask_out = args.mask_out,
        )
    
    pairwise_ious = compute_pairwise_ious(image_masks)

    bbox_ious = np.array(bbox_ious)
    mask_ious = np.array(mask_ious)
    pairwise_ious = np.array(pairwise_ious)
    pos_pts_in = np.array(pos_pts_in)

    return image_data, bbox_ious, mask_ious, pairwise_ious, pos_pts_in

def compute_pairwise_ious(masks):
    ious = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            iou = Mask.iou([masks[i]], [masks[j]], [0]).item()
            ious.append(iou)
    ious = np.array(ious)

    return ious

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
            gt_mask_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
            gt_mask_rle = Mask.merge(gt_mask_rle)
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
        colors = np.random.randint(0, 255, (dt_masks.shape[0], 3))

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
        dt_mask_image = cv2.addWeighted(image, 0.5, dt_mask_image, 0.5, 0)
        gt_mask_image = cv2.addWeighted(image, 0.5, gt_mask_image, 0.5, 0)

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
        cv2.imwrite(os.path.join(DEBUG_FOLDER, save_name), dt_mask_image)               
    else:
        cv2.imwrite(os.path.join(DEBUG_FOLDER, save_name), np.hstack([gt_mask_image, dt_mask_image]))            

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
    cv2.imwrite(os.path.join(DEBUG_FOLDER, save_name), np.hstack([gt_mask, dt_mask]))
   
def print_ious_stats(ious, iou_type):
    print("{:s} IOUs:".format(iou_type))
    print("  Mean: {:.4f}".format(ious.mean()))
    print("  Median: {:.4f}".format(np.median(ious)))
    print("  Min: {:.4f}".format(ious.min()))
    print("  Max: {:.4f}".format(ious.max()))

def monte_carlo_search(args, data, model):
    x_start = 0 if args.use_bbox else 1
    x_end = 17
    x_step = 1
    
    x_coords = np.arange(x_start, x_end+1, x_step).astype(int)
    
    bbox_ious = np.zeros(x_coords.shape, dtype=np.float32)  # Store max values obtained from processing
    mask_ious = np.zeros(x_coords.shape, dtype=np.float32)  # Store max values obtained from processing
    pair_ious = np.zeros(x_coords.shape, dtype=np.float32)  # Store max values obtained from processing
    ppts_count = np.zeros(x_coords.shape, dtype=np.float32)  # Store max values obtained from processing
    image_count_grid = np.zeros_like(x_coords)  # Track number of images processed
    people_count_grid = np.zeros_like(x_coords)  # Track number of people processed

    num_images_per_cell = len(data.keys())
    image_list = list(data.keys())

    # Sort images by number of annotations. The first images will have more annotations
    # Since we want multi-body problem, take the hardest images first
    image_list = sorted(image_list, key=lambda x: len(data[x]["annotations"]), reverse=True)

    total_iterations = len(x_coords) * args.num_images
    
    print("Running Monte Carlo search...")
    for iter_idx in tqdm(range(total_iterations), ascii=True):
        
        # Compute the UCB (upper confidence bound) value for each cell
        avg_bbox_ious = bbox_ious / np.maximum(people_count_grid, 1)
        variance = np.sqrt(np.log(iter_idx + 1) / np.maximum(image_count_grid, 1))
        ucb = avg_bbox_ious + 1.0 * variance

        # Choose a cell to process
        x = np.argmax(ucb)
        
        if image_count_grid[x] < num_images_per_cell:
            # Process a new image for this cell
            image_id = image_list[image_count_grid[x]]
            image_data = data[image_id]

            args.num_pos_keypoints = x*x_step+x_start

            _, bious, mious, pious, ppts = process_image(args, image_data, model)
            
            # Update the value as a moving average
            people_count_grid[x] += len(bious)
            bbox_ious[x] += np.sum(bious)
            mask_ious[x] += np.sum(mious)
            pair_ious[x] += np.sum(pious)
            ppts_count[x] += np.sum(ppts)
            
            # Increment the count of images processed for this cell
            image_count_grid[x] += 1

        if args.debug_vis:
            print("\n\n")

            print("UCB      : [", end="")
            for u in ucb:
                if u == ucb.max():
                    print("\033[92m{:6.3f}\033[0m".format(u), end=", ")
                elif u == ucb.min():
                    print("\033[91m{:6.3f}\033[0m".format(u), end=", ")
                else:
                    print("{:6.3f}".format(u), end=", ")
            print("]")

            print("Img count: [", end="")
            for i in image_count_grid:
                if i == image_count_grid.max():
                    print("\033[92m{:6d}\033[0m".format(i), end=", ")
                elif i == image_count_grid.min():
                    print("\033[91m{:6d}\033[0m".format(i), end=", ")
                else:
                    print("{:6d}".format(i), end=", ")
            print("]")

            print("BBox IOU : [", end="")
            for b in bbox_ious / np.maximum(people_count_grid, 1):
                if b == (bbox_ious / np.maximum(people_count_grid, 1)).max():
                    print("\033[92m{:6.3f}\033[0m".format(b), end=", ")
                elif b == (bbox_ious / np.maximum(people_count_grid, 1)).min():
                    print("\033[91m{:6.3f}\033[0m".format(b), end=", ")
                else:
                    print("{:6.3f}".format(b), end=", ")
            print("]")

            print("Mask IOU : [", end="")
            for m in mask_ious / np.maximum(people_count_grid, 1):
                if m == (mask_ious / np.maximum(people_count_grid, 1)).max():
                    print("\033[92m{:6.3f}\033[0m".format(m), end=", ")
                elif m == (mask_ious / np.maximum(people_count_grid, 1)).min():
                    print("\033[91m{:6.3f}\033[0m".format(m), end=", ")
                else:
                    print("{:6.3f}".format(m), end=", ")
            print("]")

            print("PPts cnt : [", end="")
            for p in ppts_count / np.maximum(people_count_grid, 1):
                if p == (ppts_count / np.maximum(people_count_grid, 1)).max():
                    print("\033[92m{:6.3f}\033[0m".format(p), end=", ")
                elif p == (ppts_count / np.maximum(people_count_grid, 1)).min():
                    print("\033[91m{:6.3f}\033[0m".format(p), end=", ")
                else:
                    print("{:6.3f}".format(p), end=", ")
            print("]")


    bbox_ious = bbox_ious / np.maximum(people_count_grid, 1)
    mask_ious = mask_ious / np.maximum(people_count_grid, 1)
    pair_ious = pair_ious / np.maximum(people_count_grid, 1)
    ppts_count = ppts_count / np.maximum(people_count_grid, 1)

    return bbox_ious, mask_ious, pair_ious, ppts_count, image_count_grid, x_coords

def eval_search_vector(args, bbox_ious, mask_ious, pair_ious, ppts_count):
    
    # Save the heatmap as an image
    plt.close("all")
    plt.figure()
    plt.plot(bbox_ious, label="Bounding Box", color="blue", marker="x")
    plt.plot(mask_ious, label="Mask", color="red", marker="x")
    plt.plot(pair_ious, label="Pairwise", color="green", marker="x")
    plt.plot(ppts_count, label="Pos Kpts", color="black", marker="x")

    # Mark the best value
    best_bbox = np.argmax(bbox_ious)
    best_mask = np.argmax(mask_ious)
    best_pair = np.argmin(pair_ious)
    best_ppts = np.argmax(ppts_count)
    plt.scatter(best_bbox, bbox_ious[best_bbox], marker="*", color="blue", s=100)
    plt.scatter(best_mask, mask_ious[best_mask], marker="*", color="red", s=100)
    plt.scatter(best_pair, pair_ious[best_pair], marker="*", color="green", s=100)
    plt.scatter(best_ppts, ppts_count[best_ppts], marker="*", color="black", s=100)

    plt.title("Bounding Box and Mask IOU")
    plt.xlabel("Num Pos Kpts")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid()
    plt.savefig("{:s}/search_plot.png".format(DEBUG_FOLDER))

    print("#"*40)
    print("Best Bounding Box IOU: {:.4f} at {:d} keypoints".format(bbox_ious[best_bbox], best_bbox))
    print("\t(worst: {:.4f} at {:d} keypoints)".format(bbox_ious.min(), np.argmin(bbox_ious)))
    print("Best Mask IOU: {:.4f} at {:d} keypoints".format(mask_ious[best_mask], best_mask))
    print("\t(worst: {:.4f} at {:d} keypoints)".format(mask_ious.min(), np.argmin(mask_ious)))
    print("Best Pairwise IOU: {:.4f} at {:d} keypoints".format(pair_ious[best_pair], best_pair))
    print("\t(worst: {:.4f} at {:d} keypoints)".format(pair_ious.min(), np.argmin(pair_ious)))
    print("Best Pos Kpts: {:.4f} at {:d} keypoints".format(ppts_count[best_ppts], best_ppts))
    print("\t(worst: {:.4f} at {:d} keypoints)".format(ppts_count.min(), np.argmin(ppts_count)))

    plt.close("all")
    plt.figure()
    bbox_norm = bbox_ious.flatten() - bbox_ious.min()
    bbox_norm /= (bbox_norm.max() + 1e-8)
    mask_norm = mask_ious.flatten() - mask_ious.min()
    mask_norm /= (mask_norm.max() + 1e-8)
    pair_norm = pair_ious.flatten() - pair_ious.min()
    pair_norm /= (pair_norm.max() + 1e-8)
    plt.plot(bbox_norm, label="BBoxes", color="blue")
    plt.plot(mask_norm, label="Masks", color="red")
    plt.plot(pair_norm, label="Pairwise", color="green")
    plt.legend()
    plt.grid()
    plt.title("Bounding Box and Mask IOU correlation")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized IOU")
    plt.savefig("{:s}/iou_correlation.png".format(DEBUG_FOLDER))


def main(args):
    # Load the data from the json file
    data = load_data(args.dataset, args.dataset_path, args.subset, args.gt_file)

    # Parse images with annotations for image-wise processing
    parsed_data = parse_images(data)

    # Prepare the model
    model = prepare_model(args)

    # Remove all images from the debug folder
    if args.debug_vis:
        shutil.rmtree(DEBUG_FOLDER, ignore_errors=True)
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    # Process the images
    print("Generating masks to images...")
    tmp_i = 0

    if args.monte_carlo_search:
        bbox_ious, mask_ious, pair_ious, ppts_count, icount, xs = monte_carlo_search(args, parsed_data, model)
        eval_search_vector(args, bbox_ious, mask_ious, pair_ious, ppts_count)
        # Save the data for further analysis
        np.savez("{:s}/monte_carlo_search_{:d}.npz".format(DEBUG_FOLDER, args.num_images), bbox_ious=bbox_ious, mask_ious=mask_ious, pair_ious=pair_ious, icount=icount)
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
            
            # # Draw the masks on the image to see correct parsing
            # image = cv2.imread(os.path.join(args.images_root, new_image_data["file_name"]))
            # for ann_i, ann in enumerate(new_image_data["annotations"]):
            #     mask = Mask.decode(ann["segmentation"])
            #     mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            #     mask = mask[:, :, None] * np.array([0, 255, 0], dtype=np.uint8)
            #     save_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

            #     for kpt in np.array(ann["keypoints"]).reshape(-1, 3):
            #         if kpt[2] == 2:
            #             cv2.circle(save_image, kpt[:2].astype(int), 3, (0, 0, 255), -1)
            #     cv2.imwrite(os.path.join(DEBUG_FOLDER, "mask_{:d}_{:d}.jpg".format(abs(hash(new_image_data['file_name'])), ann_i)), save_image)

            parsed_data[image_id] = new_image_data

            bbox_ious.extend(bious)
            mask_ious.extend(mious)
            pair_ious.extend(pious)

        save_data = unparse_images(parsed_data)
        save_data_dir = os.path.join(args.dataset_path, "sam_masks")
        os.makedirs(save_data_dir, exist_ok=True)
        with open(os.path.join(save_data_dir, "masks_{:s}.json".format(args.subset)), "w") as f:
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