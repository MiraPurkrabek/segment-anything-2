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

from sam2.visualization import batch_visualize_masks

DEBUG_FOLDER = "debug" 

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your script")
    # Dataset type. One of ("COCO", "OCHuman", "CrowdPose", "MPII", "AIC")
    parser.add_argument("--dataset", type=str, default="COCO")

    # Dataset subset. One of ("train", "val", "test")
    parser.add_argument("--subset", type=str, default="val")

    # Number of images to process
    parser.add_argument("--num-images", type=int, default=10)

    # Number of keypoints to use
    parser.add_argument("--num-pos-keypoints", type=int, default=17)
    parser.add_argument("--num-neg-keypoints", type=int, default=17)

    # Boolean flags, default to False
    parser.add_argument("--debug-vis", action="store_true")
    parser.add_argument("--test-keypoints", action="store_true")
    parser.add_argument("--monte-carlo-search", action="store_true")

    # Boolean flags, default to True
    parser.add_argument('--use-bbox', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    # Check the dataset and subset
    assert args.dataset in ["COCO", "OCHuman", "CrowdPose", "MPII", "AIC"]
    assert args.subset in ["train", "val", "test"]

    args.dataset_path, args.subset = find_dataset_path(args.dataset, args.subset)

    args.vis_by_name = True

    return args

def find_dataset_path(dataset, subset):
    path_to_home = "/datagrid/personal/purkrmir/"
    subset_name = subset.lower()

    if dataset == "COCO":
        path_to_data = os.path.join(path_to_home, "data/COCO/original")
        subset_name = "{:s}2017".format(subset.lower())
    elif dataset == "OCHuman":
        path_to_data = os.path.join(path_to_home, "data/OCHuman/COCO-like")
        subset_name = "{:s}2017".format(subset.lower())
    elif dataset == "CrowdPose":
        path_to_data = os.path.join(path_to_home, "data/CrowdPose")
    elif dataset == "MPII":
        path_to_data = os.path.join(path_to_home, "data/MPII")
    elif dataset == "AIC":
        path_to_data = os.path.join(path_to_home, "data/AIC")
    else:
        raise ValueError("Unknown dataset")

    return path_to_data, subset_name
    
def load_data(dataset_path, subset):
    # Load the dataset
    with open(os.path.join(dataset_path, "annotations", "person_keypoints_{:s}.json".format(subset)), "r") as f:
        data = json.load(f)

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
    for annotation in tqdm(data["annotations"], ascii=True):
        iscrowd = annotation["iscrowd"]
        
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

    return parsed_data

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

def process_image(args, image_data, model):
    image_path = os.path.join(args.dataset_path, args.subset, image_data["file_name"])
    image = cv2.imread(image_path)

    bbox_ious = []
    mask_ious = []

    model.set_image(image)

    image_kpts = []
    for annotation in image_data["annotations"]:
        this_kpts = np.array(annotation["keypoints"]).reshape(-1, 3)
        this_kpts[this_kpts[:, 2] != 2, :2] = 0
        image_kpts.append(this_kpts)
    image_kpts = np.array(image_kpts)

    image_masks = []
    dt_bboxes = []
    pos_image_kpts = []
    gt_bboxes = []
    gt_masks = []
    # for annotation in tqdm(image_data["annotations"], ascii=True):
    for ann_idx, annotation in enumerate(image_data["annotations"]):
        bbox_xywh = annotation["bbox"]
        gt_mask = annotation.get("segmentation", None)
        gt_masks.append(gt_mask)
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
        )

        pos_image_kpts.append(pos_kpts)

        # if args.use_bbox:
        #     bbox_mask = np.zeros_like(dt_mask)
        #     bbox_xyxy_int = np.round(bbox_xyxy).astype(int)
        #     bbox_mask[bbox_xyxy_int[1]:bbox_xyxy_int[3], bbox_xyxy_int[0]:bbox_xyxy_int[2]] = 1
        #     dt_mask = np.logical_and(dt_mask, bbox_mask)

        dt_mask_rle = Mask.encode(np.asfortranarray(dt_mask.astype(np.uint8)))
        image_masks.append(dt_mask_rle)

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


    if args.debug_vis and np.random.rand() < 1.0:
        gt_bboxes = np.array(gt_bboxes)
        dt_bboxes = np.array(dt_bboxes)
        

        filtered_gt_bboxes = []
        filtered_dt_bboxes = []
        filtered_gt_masks = []
        filtered_bbox_ious = []
        filtered_mask_ious = []
        filtered_pos_image_kpts = []
        filtered_image_masks = []
        for i in range(len(image_masks)):
            num_pos_kpts = len(pos_image_kpts[i])
            if num_pos_kpts < args.num_pos_keypoints - 2:
                continue
            filtered_gt_bboxes.append(gt_bboxes[i])
            filtered_dt_bboxes.append(dt_bboxes[i])
            filtered_gt_masks.append(gt_masks[i])
            filtered_bbox_ious.append(bbox_ious[i])
            filtered_mask_ious.append(mask_ious[i])
            filtered_pos_image_kpts.append(pos_image_kpts[i])
            filtered_image_masks.append(image_masks[i])
        
        image_masks = filtered_image_masks
        gt_bboxes = filtered_gt_bboxes
        dt_bboxes = filtered_dt_bboxes
        gt_masks = filtered_gt_masks
        bbox_ious = filtered_bbox_ious
        mask_ious = filtered_mask_ious
        pos_image_kpts = filtered_pos_image_kpts

        try:
            batch_visualize_masks(
                args,
                image,
                image_masks, 
                pos_image_kpts,
                gt_bboxes,
                dt_bboxes,
                gt_masks,
                bbox_ious,
                mask_ious,
                image_path=image_path,
            )
        except Exception as e:
            print(e)
                

    pairwise_ious = compute_pairwise_ious(image_masks)

    return bbox_ious, mask_ious, pairwise_ious

def compute_pairwise_ious(masks):
    ious = []
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            iou = Mask.iou([masks[i]], [masks[j]], [0]).item()
            ious.append(iou)
    ious = np.array(ious)

    return ious

def pose2seg(args, model, bbox_xyxy=None, pos_kpts=None, neg_kpts=None):
    # Filter-out un-annotated and invisible keypoints
    if pos_kpts is not None:
        pos_kpts = pos_kpts.reshape(-1, 3)
        valid_kpts = pos_kpts[:, 2] == 2
        pos_kpts = pos_kpts[valid_kpts, :2]

        # Sort keypoinst by their distance to the center of the bounding box
        if bbox_xyxy is not None:
            bbox_center = np.array([(bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2])
        else:
            bbox_center = np.mean(pos_kpts, axis=0)
        
        dists = np.linalg.norm(pos_kpts - bbox_center, axis=1)
        distance_matrix = np.linalg.norm(pos_kpts[:, None, :] - pos_kpts[None, :, :], axis=2)
        np.fill_diagonal(distance_matrix, np.inf)
        min_inter_dist = np.min(distance_matrix, axis=1)
        sort_idx = np.argsort(dists+min_inter_dist)[::-1]
        pos_kpts = pos_kpts[sort_idx, :]

        if pos_kpts.shape[0] > args.num_pos_keypoints:
            pos_kpts = pos_kpts[:args.num_pos_keypoints, :]

    else:
        pos_kpts = np.empty((0, 2), dtype=np.float32)

    if neg_kpts is not None:
        neg_kpts = neg_kpts.reshape(-1, 3)
        valid_kpts = neg_kpts[:, 2] == 2
        neg_kpts = neg_kpts[valid_kpts, :2]

        # For each negative keypoint, compute its distance to the bounding box
        if bbox_xyxy is not None:
            x_dist = np.maximum(
                np.maximum(bbox_xyxy[0] - neg_kpts[:, 0], 0),
                np.maximum(neg_kpts[:, 0] - bbox_xyxy[2], 0),
            )
            y_dist = np.maximum(
                np.maximum(bbox_xyxy[1] - neg_kpts[:, 1], 0),
                np.maximum(neg_kpts[:, 1] - bbox_xyxy[3], 0),
            )
            dists = x_dist**2 + y_dist**2
            sort_idx = np.argsort(dists)
            neg_kpts = neg_kpts[sort_idx, :]
        else:
            # Shuffle the keypoints
            np.random.shuffle(neg_kpts, axis=0)
        
        if neg_kpts.shape[0] > args.num_neg_keypoints:
            neg_kpts = neg_kpts[:args.num_neg_keypoints, :]

    else:
        neg_kpts = np.empty((0, 2), dtype=np.float32)

    # Concatenate positive and negative keypoints
    kpts = np.concatenate([pos_kpts, neg_kpts], axis=0)
    kpts_labels = np.concatenate([np.ones(pos_kpts.shape[0]), np.zeros(neg_kpts.shape[0])], axis=0)

    # print(kpts.shape, kpts_labels.shape)
    # print(kpts)
    # print(kpts_labels)

    # Take only the positive keypoints
    # kpts = pos_kpts
    # kpts_labels = np.ones(kpts.shape[0])

    bbox = bbox_xyxy if args.use_bbox else None

    masks, scores, logits = model.predict(
        point_coords=kpts,
        point_labels=kpts_labels,
        box=bbox,
        multimask_output=False,
    )

    mask = masks[0]
    return mask, pos_kpts, neg_kpts

def _batch_visualize_masks(image, masks_rle, image_kpts, bboxes_xyxy, dt_bboxes, gt_masks_raw, bbox_ious, mask_ious):
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
    dt_mask_image = cv2.addWeighted(image, 0.5, dt_mask_image, 0.5, 0)
    gt_mask_image = cv2.addWeighted(image, 0.5, gt_mask_image, 0.5, 0)

    # Draw keypoints
    if image_kpts is not None:
        for instance_kpts, color in zip(image_kpts, colors):
            color = tuple(color.astype(int).tolist())
            for kpt in instance_kpts:
                cv2.circle(dt_mask_image, kpt.astype(int)[:2], 3, color, -1)
                cv2.circle(gt_mask_image, kpt.astype(int)[:2], 3, color, -1)

    # Draw bboxes
    if bboxes_xyxy is not None:
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
    save_name = "batch_bbox_{:06.2f}_mask_{:06.2f}_{:02d}kpts_{:06d}.jpg".format(
        bbox_ious.mean(), mask_ious.mean(), args.num_pos_keypoints, np.random.randint(1000000),
    )
    cv2.imwrite(os.path.join(DEBUG_FOLDER, save_name), np.hstack([gt_mask_image, dt_mask_image]))

def visualize_masks(image, dt_mask_rle, gt_mask_rle, dt_bbox, gt_bbox, bbox_iou, mask_iou, pos_kpts=None, neg_kpts=None):
    dt_mask = Mask.decode(dt_mask_rle)*255

    if not gt_mask_rle:
        gt_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    else:
        gt_mask = Mask.decode(gt_mask_rle)*255

    if not dt_mask.any():
        dt_mask = np.random.randint(0, 2, (image.shape[0], image.shape[1]), dtype=np.uint8) * 255

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
    data = load_data(args.dataset_path, args.subset)

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
                        bious, mious, pious = process_image(args, image_data, model)
                        
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
            bious, mious, pious = process_image(args, image_data, model)
            
            bbox_ious.extend(bious)
            mask_ious.extend(mious)
            pair_ious.extend(pious)

            if tmp_i > args.num_images:
                break
            tmp_i += 1

        bbox_ious = np.array(bbox_ious)
        mask_ious = np.array(mask_ious)
        pair_ious = np.array(pair_ious)

        print_ious_stats(bbox_ious, "Bounding Box")
        print_ious_stats(mask_ious, "Mask")
        print_ious_stats(pair_ious, "Pairwise Mask")

if __name__ == "__main__":
    
    args = parse_args()
    main(args)