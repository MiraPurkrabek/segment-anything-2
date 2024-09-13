import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_NAME = "SKV_test_frame_000001"
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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size)#, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size)#, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    H, W = image.shape[:2]
    merged_mask = np.zeros((H, W, 4), dtype=np.float32)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # breakpoint()
        ax = plt.gca()
        drawed_mask = draw_mask(mask, ax, borders=borders, random_color=True)

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
    plt.savefig(os.path.join(save_dir, "{:s}_mask.png".format(IMAGE_NAME)), bbox_inches='tight', pad_inches=0)

this_dir = os.path.dirname(os.path.abspath(__file__))
image = Image.open(os.path.join(this_dir, "..", "notebooks", "images", "{:s}.{:s}".format(IMAGE_NAME, IMAGE_EXT)))
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

predictor = SAM2ImagePredictor(sam2)
predictor.set_image(image)

input_points = np.array([
    [
        [1928, 285],
        [1877, 394],
        [1924, 365],
    ],  
    [
        [1412, 140],
        [1399, 194],
        [1425, 200],
    ],  
    [
        [1308, 265],
        [1319, 360],
        [1301, 388],
    ],  
    [
        [909, 431],
        [923, 580],
        [896, 567],
    ],  
    [
        [1132, 457],
        [1132, 592],
        [1129, 646],
    ],  
    [
        [1625, 39],
        [1625, 72],
        [1655, 128],
    ],  
    [
        [1848, 14],
        [1851, 54],
        [1879, 40],
    ],  
    [
        [2024, 130],
        [2040, 199],
        [2075, 221],
    ],  
    [
        [2670, 100],
        [2710, 188],
        [2700, 212],
    ],  
    [
        [2878, 312],
        [2865, 423],
        [2877, 487],
    ],  
    [
        [2343, 460],
        [2368, 661],
        [2343, 697],
    ],  
    [
        [1909, 1035],
        [1988, 1297],
        [1963, 1356],
    ],  
])
input_labels = np.array([
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
])

input_points = input_points / scale_factor

print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

# Select the best mask

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)


# For each instance, select its best mask
ind = np.argmax(scores, axis=-1)
masks = masks[np.arange(masks.shape[0]), ind]
scores = scores[np.arange(scores.shape[0]), ind]
logits = logits[np.arange(logits.shape[0]), ind]

print("Masks", masks.shape)  # (number_of_masks) x H x W

show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels, borders=False)

