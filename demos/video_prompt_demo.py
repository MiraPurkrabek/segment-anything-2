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

def draw_mask(mask, ax, random_color=False, borders = True, obj_id=None):
    
    cmap = plt.get_cmap("tab20")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])

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

def show_masks(image, masks, frame_idx, obj_ids, point_coords=None, box_coords=None, input_labels=None, borders=True):
    plt.close("all")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    H, W = image.shape[:2]
    merged_mask = np.zeros((H, W, 4), dtype=np.float32)

    for i, (mask, obj_id) in enumerate(zip(masks, obj_ids)):
        # breakpoint()
        ax = plt.gca()
        drawed_mask = draw_mask(mask, ax, borders=borders, obj_id=obj_id)

        # Merge masks such that the final mask is the union of all masks but each with its own color
        merged_mask = np.maximum(merged_mask, drawed_mask)
        
    # Draw the merged mask
    ax = plt.gca()
    ax.imshow(merged_mask, alpha=1.0)

    plt.axis('off')
    # plt.show()
    # Save the image
    save_dir = os.path.join(this_dir, "video_results")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "frame_{:06d}.jpg".format(frame_idx)), bbox_inches='tight', pad_inches=0)

this_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(this_dir, "..", "notebooks", "videos", "SKV_test_video_1080")

# # Resize image
# image_height, image_width = image.shape[:2]
max_width = 1080
scale_factor = 3840 / max_width
# # if image_width > 640:
# #     image = np.array(Image.fromarray(image).resize((max_width, int(max_width * image_height / image_width))))
# #     scale_factor = image_width / max_width

# # Print image size
# print("Image shape", image.shape)

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = os.path.join(this_dir, "..", "checkpoints", "sam2_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Pre-load sorted video frames in format *_%06d.jpg
video_frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".jpg")])
video_frames = [os.path.join(video_dir, f) for f in video_frames]

inference_state = predictor.init_state(
    video_path=video_dir,
    async_loading_frames=True,
    offload_state_to_cpu=True,
    offload_video_to_cpu=True,
)

predictor.reset_state(inference_state)

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

prompts = {}
init_frame_idx = 0
for i, (points, labels) in enumerate(zip(input_points, input_labels)):
    prompts[i] = points, labels

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=init_frame_idx,
        obj_id=i,
        points=points,
        labels=labels,
    )

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# render the segmentation results every few frames
vis_frame_stride = 1
for out_frame_idx in range(0, len(video_frames), vis_frame_stride):
    image = Image.open(os.path.join(video_dir, video_frames[out_frame_idx]))
    image = np.array(image.convert("RGB"))
    obj_ids = np.array(list(video_segments[out_frame_idx].keys()))
    out_mask = np.array(list(video_segments[out_frame_idx].values()))
    show_masks(image, out_mask, out_frame_idx, obj_ids, borders=False)

