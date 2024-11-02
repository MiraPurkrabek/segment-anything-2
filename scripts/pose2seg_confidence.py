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

from sam2.visualization import batch_visualize_masks
from sam2.pose2seg_helper import (
    select_keypoints, compute_pairwise_ious,
    compute_one_mask_pose_consistency, find_dataset_path,
    load_data, parse_images, unparse_images,
    prepare_model
)

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
    parser.add_argument('--mask-out', action="store_true")
    parser.add_argument('--output-as-list', action="store_true")

    # Special flags for better pose2seg
    parser.add_argument("--expand-bbox", action="store_true", help="Expand bbox if any of the selected pose kpts is outside the bbox")
    parser.add_argument("--oracle", action="store_true", help="Evaluate dt mask compared to gt mask and take the best one")
    parser.add_argument("--crop", action="store_true", help="Crop the image to the 1.5x bbox size to increase the resolution")
    parser.add_argument('--ignore-SAM', action="store_true", help="Ignore the SAM model and use the provided masks")

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


        if not args.ignore_SAM:
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
        else:
            dt_mask = gt_masks[-1]
            if dt_mask is not None:
                dt_mask = Mask.decode(dt_mask).astype(bool)

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
            args,
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
        dt_mask_pose_consistency = compute_one_mask_pose_consistency(args, mask, pos_kpts_backup, neg_kpts_backup)
        gt_mask_binary = Mask.decode(gt_mask).astype(bool) if gt_mask is not None else None
        gt_mask_pose_consistency = compute_one_mask_pose_consistency(args, gt_mask_binary, pos_kpts_backup, neg_kpts_backup)

        dt_is_worse = dt_mask_pose_consistency < gt_mask_pose_consistency
        if dt_is_worse:
            mask = gt_mask_binary

    return mask, pos_kpts, neg_kpts

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

def main(args):
    args = parse_args()

    # Load the data from the json file
    data = load_data(args.dataset, args.dataset_path, args.subset, args.gt_file)

    # Parse images with annotations for image-wise processing
    parsed_data = parse_images(args, data)

    # Prepare the model
    model = prepare_model(args)

    # Remove all images from the debug folder
    if args.debug_vis:
        shutil.rmtree(args.debug_folder, ignore_errors=True)
    os.makedirs(args.debug_folder, exist_ok=True)

    # Process the images
    print("Generating masks to images...")
    tmp_i = 0

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


if __name__ == "__main__":
    
    args = parse_args()
    main(args)