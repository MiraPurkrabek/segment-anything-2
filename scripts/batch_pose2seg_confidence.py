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
    
    parser.add_argument("--debug-folder", type=str, default="debug_batch", help="Folder to save debug images")
    parser.add_argument("--out-filename", type=str, default="sam_masks_single")    
    parser.add_argument("--selection-method", type=str, default="distance+confidence")
    
    # Boolean flags, default to False
    parser.add_argument('--mask-out', action="store_true")
    parser.add_argument('--output-as-list', action="store_true")

    # Special flags for better pose2seg
    parser.add_argument("--expand-bbox", action="store_true", help="Expand bbox if any of the selected pose kpts is outside the bbox")
    parser.add_argument("--oracle", action="store_true", help="Evaluate dt mask compared to gt mask and take the best one")
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

    model.set_image(image)

    image_kpts = []
    image_bboxes = []
    num_valid_kpts = []
    gt_masks = []
    for annotation in image_data["annotations"]:
        this_kpts = np.array(annotation["keypoints"]).reshape(-1, 3)
        num_visible = (this_kpts[:, 2] > args.conf_thr).sum()
        if num_visible <= 0:
            continue
        num_valid_kpts.append(num_visible)
        image_bboxes.append(np.array(annotation["bbox"]))
        this_kpts[this_kpts[:, 2] < args.conf_thr, :2] = 0
        image_kpts.append(this_kpts)
        gtm = annotation.get("segmentation", None)
        if gtm is not None and len(gtm) > 0:
            gtm_rle = Mask.frPyObjects(gtm, image.shape[0], image.shape[1])
            gtm_rle = Mask.merge(gtm_rle)
            gt_masks.append(gtm_rle)
        else:
            gt_masks.append(None)
    image_kpts = np.array(image_kpts)
    image_bboxes = np.array(image_bboxes)
    num_valid_kpts = np.array(num_valid_kpts)
    gt_masks = np.array(gt_masks)

    image_kpts_backup = image_kpts.copy()
    
    # Prepare keypoints such that all instances have the same number of keypoints
    # First sort keypoints by their distance to the center of the bounding box
    # If some are missing, duplicate the last one
    prepared_kpts = []
    for bbox, kpts, num_visible in zip(image_bboxes, image_kpts, num_valid_kpts):
        
        this_kpts, _ = select_keypoints(args, kpts, num_visible, bbox)

        # Duplicate the last keypoint if some are missing
        if this_kpts.shape[0] < num_valid_kpts.max():
            this_kpts = np.concatenate([this_kpts, np.tile(this_kpts[-1], (num_valid_kpts.max() - this_kpts.shape[0], 1))], axis=0)

        prepared_kpts.append(this_kpts)
    image_kpts = np.array(prepared_kpts)
    kpts_labels = np.ones(image_kpts.shape[:2])
    
    # Threshold the number of positive keypoints
    if args.num_pos_keypoints > 0 and args.num_pos_keypoints < image_kpts.shape[1]:
        image_kpts = image_kpts[:, :args.num_pos_keypoints, :]
        kpts_labels = kpts_labels[:, :args.num_pos_keypoints]

    elif args.num_pos_keypoints == 0:
        image_kpts = None
        kpts_labels = None

    image_bboxes_xyxy = None
    if args.use_bbox:
        image_bboxes_xyxy = np.array(image_bboxes)
        image_bboxes_xyxy[:, 2:] += image_bboxes_xyxy[:, :2]

        # Expand the bbox to include the positive keypoints
        if args.expand_bbox:
            pose_bbox = np.array([np.min(image_kpts[:, :, 0])-2, np.min(image_kpts[:, :, 1])-2, np.max(image_kpts[:, :, 0])+2, np.max(image_kpts[:, :, 1])+2]).T
            expanded_bbox = np.array(image_bboxes)
            expanded_bbox[:, :2] = np.minimum(expanded_bbox[:, :2], pose_bbox[:, :2])
            expanded_bbox[:, 2:] = np.maximum(expanded_bbox[:, 2:], pose_bbox[:, 2:])
            image_bboxes_xyxy = expanded_bbox

    if args.ignore_SAM:
        bbox_ious = np.zeros(image_kpts.shape[0])
        mask_ious = np.zeros(image_kpts.shape[0])
        pairwise_ious = np.zeros(image_kpts.shape[0])
        pos_pts_in = np.zeros(image_kpts.shape[0])
        dt_is_worse_ratio = 1.0

        
        if args.debug_vis and np.random.rand() < 1.0:
            batch_visualize_masks(
                args,
                image,
                gt_masks,
                image_kpts,
                image_bboxes_xyxy,
                None,
                gt_masks,
                bbox_ious,
                mask_ious,
                image_path = image_path if args.vis_by_name else None,
                mask_out = args.mask_out,
            )

    else:
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

        dt_is_worse_ratio = 0.0
        if args.oracle:
            # Measure 'mask-pose_conistency' by computing number of keypoints inside the mask
            # Compute for both gt (if available) and predicted masks and then choose the one with higher consistency
            dt_mask_pose_consistency = compute_mask_pose_consistency(args, masks, image_kpts_backup)
            gt_masks_binary = [Mask.decode(gt_mask).astype(bool) for gt_mask in gt_masks if gt_mask is not None]
            gt_mask_pose_consistency = compute_mask_pose_consistency(args, gt_masks_binary, image_kpts_backup)

            dt_is_worse = (dt_mask_pose_consistency < gt_mask_pose_consistency)
            # print("Number of instances where dt is worse: {:3d} ({:5.1f} %)".format(dt_is_worse.sum(), 100*dt_is_worse.mean()))
            dt_is_worse_ratio = dt_is_worse.mean()

            new_masks = []
            for dt_mask, gt_mask, dt_consistency, gt_consistency in zip(masks, gt_masks_binary, dt_mask_pose_consistency, gt_mask_pose_consistency):
                if gt_mask is None:
                    new_masks.append(dt_mask)
                    continue
                if dt_consistency > gt_consistency:
                    new_masks.append(dt_mask)
                else:
                    new_masks.append(gt_mask)
            masks = np.array(new_masks)

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
            
            # # Mask prediction with GT bbox
            # if args.use_bbox:
            #     bbox_int = np.round(bbox_xyxy).astype(int)
            #     bbox_mask = np.zeros_like(mask)
            #     bbox_mask[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]] = 1
            #     mask = np.logical_and(mask, bbox_mask)

            dt_mask_rle = Mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            image_masks.append(dt_mask_rle)

            image_data["annotations"][instance_i]["segmentation"] = dt_mask_rle

            if args.update_bboxes:
                dt_bbox = Mask.toBbox(dt_mask_rle).tolist()
                image_data["annotations"][instance_i]["bbox"] = dt_bbox

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
                # gt_mask_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
                # gt_mask_rle = Mask.merge(gt_mask_rle)
                gt_mask_rle = gt_mask

                mask_iou = Mask.iou([gt_mask_rle], [dt_mask_rle], [0]).item()
                mask_ious.append(mask_iou)
            else:
                mask_iou = 0
                mask_ious.append(0)

        if args.debug_vis and np.random.rand() < 1.0:
            batch_visualize_masks(
                args,
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

    return image_data, bbox_ious, mask_ious, pairwise_ious, dt_is_worse_ratio

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