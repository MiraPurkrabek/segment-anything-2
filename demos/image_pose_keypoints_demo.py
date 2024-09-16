import os
import json
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# PATH_TO_COCO = "../data/COCO/original/val2017"
# IMAGE_NAME = "000000000872"
PATH_TO_COCO = "../data/OCHuman/COCO-like/val2017"
IMAGE_NAME = "005079"
IMAGE_EXT = "jpg"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    save_dir = os.path.join(this_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "{:s}_mask.png".format(IMAGE_NAME)), bbox_inches='tight', pad_inches=0)

def draw_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    
    return mask_image
    

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size)#, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size)#, edgecolor='white', linewidth=1.25)   

def show_box(box, ax, color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, instance_ID=-1, gt_bbox=None, mask_bbox=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    H, W = image.shape[:2]
    merged_mask = np.zeros((H, W, 4), dtype=np.float32)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # breakpoint()
        ax = plt.gca()
        drawed_mask = draw_mask(mask, ax, borders=borders, random_color=True)

        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())

        if gt_bbox is not None:
            show_box(gt_bbox, plt.gca(), 'green')
        
        if mask_bbox is not None:
            show_box(mask_bbox, plt.gca(), 'red')

        # Merge masks such that the final mask is the union of all masks but each with its own color
        merged_mask = np.maximum(merged_mask, drawed_mask)
        
    # Draw the merged mask
    ax = plt.gca()
    ax.imshow(merged_mask, alpha=1.0)

    plt.axis('off')
    # plt.show()
    # Save the image
    save_dir = os.path.join(this_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    save_name = "{:s}_mask.png".format(IMAGE_NAME) if instance_ID == -1 else "{:s}_mask_{:d}.png".format(IMAGE_NAME, instance_ID)
    plt.savefig(os.path.join(save_dir, save_name), bbox_inches='tight', pad_inches=0)

def get_bounding_box(mask):
    # Find the rows and columns where the mask is non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get the bounding box coordinates
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Return the bounding box as (row_min, col_min, row_max, col_max)
    return cmin, rmin, cmax, rmax

this_dir = os.path.dirname(os.path.abspath(__file__))
image = Image.open(os.path.join(PATH_TO_COCO, "{:s}.{:s}".format(IMAGE_NAME, IMAGE_EXT)))
image = np.array(image.convert("RGB"))

# Resize image
image_height, image_width = image.shape[:2]
scale_factor = 1
max_width = 1080
if image_width > 640:
    image = np.array(Image.fromarray(image).resize((max_width, int(max_width * image_height / image_width))))
    scale_factor = image_width / max_width

# Print image size
print("Image shape", image.shape)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = os.path.join(this_dir, "..", "checkpoints", "sam2_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

predictor = SAM2ImagePredictor(sam2, max_hole_area=2.0, max_sprinkle_area=10.0)
predictor.set_image(image)

# Load the COCO annotations
coco_anns = json.load(open(os.path.join(PATH_TO_COCO, "..", "annotations", "person_keypoints_val2017.json")))
selected_imgID = None
for img in coco_anns["images"]:
    if img["file_name"] == "{:s}.{:s}".format(IMAGE_NAME, IMAGE_EXT):
        selected_imgID = img["id"]
        break
if selected_imgID is None:
    raise ValueError("Image not found in COCO annotations")

# Get the keypoints for the selected image
image_keypoints = []
image_bboxes = []
for ann in coco_anns["annotations"]:
    if ann["image_id"] == selected_imgID:
        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
        kpts_mask = keypoints[:, 2] != 2
        if not kpts_mask.any():
            continue
        keypoints[kpts_mask, :] = -1
        keypoints = keypoints[:, :2]
        image_keypoints.append(keypoints)
        bbox_xywh = ann["bbox"]
        bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
        image_bboxes.append(bbox_xyxy)

image_keypoints = np.array(image_keypoints) / scale_factor

for instance_i in range(image_keypoints.shape[0]):
    print("#"*20)
    print("Instance", instance_i)

    instance_mask = np.zeros(image_keypoints.shape[0], dtype=bool)
    instance_mask[instance_i] = True
    positive_keypoints = image_keypoints[instance_mask]
    if instance_mask.all():
        negative_keypoints = np.array([])
    else:
        negative_keypoints = image_keypoints[~instance_mask]

    positive_keypoints = positive_keypoints.reshape(-1, 2)
    negative_keypoints = negative_keypoints.reshape(-1, 2)

    pos_mask = positive_keypoints[:, 0] > 0
    neg_mask = negative_keypoints[:, 0] > 0

    positive_keypoints = positive_keypoints[pos_mask]
    negative_keypoints = negative_keypoints[neg_mask]

    MAX_KEYPOINTS = 17
    if len(positive_keypoints) > MAX_KEYPOINTS:
        # Randomly select MAX_KEYPOINTS keypoints
        selected_indices = np.random.choice(len(positive_keypoints), MAX_KEYPOINTS, replace=False)
        positive_keypoints = positive_keypoints[selected_indices]
    if len(negative_keypoints) > MAX_KEYPOINTS:
        # Randomly select MAX_KEYPOINTS keypoints
        selected_indices = np.random.choice(len(negative_keypoints), MAX_KEYPOINTS, replace=False)
        negative_keypoints = negative_keypoints[selected_indices]

    points = np.concatenate([positive_keypoints, negative_keypoints], axis=0)
    labels = np.concatenate([np.ones(len(positive_keypoints)), np.zeros(len(negative_keypoints))], axis=0)

    # print("Positive keypoints", positive_keypoints)
    # print("Negative keypoints", negative_keypoints)

    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    # Select the best mask

    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    # breakpoint()

    n_masks = scores.shape[-1]
    scores = scores.reshape(-1, n_masks)
    masks = masks.reshape(-1, n_masks, *masks.shape[1:])

    # For each instance, select its best mask
    ind = np.argmax(scores, axis=-1)
    masks = masks[np.arange(masks.shape[0]), ind]
    scores = scores[np.arange(scores.shape[0]), ind]
    # logits = logits[np.arange(logits.shape[0]), ind]

    print("Masks", masks.shape)  # (number_of_masks) x H x W

    mask_bbox = get_bounding_box(masks[0])
    print("Mask bbox", mask_bbox)
    gt_bbox = image_bboxes[instance_i]
    print("GT bbox", gt_bbox)

    show_masks(image, masks, scores, point_coords=points, input_labels=labels, borders=False, instance_ID=instance_i, gt_bbox=gt_bbox, mask_bbox=mask_bbox)

