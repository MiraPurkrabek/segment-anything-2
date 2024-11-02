# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor

import numpy as np
from scipy.ndimage import label

def get_connected_components(mask):
    """
    Finds connected components in a binary mask image with 8-connectivity.

    Parameters:
    - mask (np.array): A binary mask image where 1 represents foreground.

    Returns:
    - tuple of (labels, areas):
      - labels (np.array): Array with the same shape as `mask`, where each connected component has a unique label.
      - areas (dict): Dictionary where keys are component labels and values are the areas of the components.
    """
    # Check if the mask is torch tensor. If so, remember that and convert it to numpy array.
    is_tensor = False
    if isinstance(mask, torch.Tensor):
        is_tensor = True
        device = mask.device
        mask = mask.detach().cpu().numpy()

    # Using scipy's label function to find connected components
    struct = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])  # Defining the structure for 8-connectivity

    # Apply connected components labeling, mask-by-mask
    out_shape = mask.squeeze().shape
    if len(out_shape) == 2:
        out_shape = (1,) + out_shape
    labeled_array_all = np.zeros(out_shape, dtype=int)
    area_map_all = np.zeros(out_shape, dtype=int)



    for i in range(mask.shape[0]):
        labeled_array, num_features = label(mask[i].squeeze(), structure=struct)
        
        # breakpoint()
        unique_labels, counts = np.unique(labeled_array, return_counts=True)
        # Make 'areas' 2D array where each cell corresponds to the area of the component with the same label
        area_map = np.zeros_like(labeled_array)
        # Map counts back to their respective labels
        for li, count in zip(unique_labels, counts):
            area_map[labeled_array == li] = count

        labeled_array_all[i] = labeled_array
        area_map_all[i] = area_map

    # Convert the labeled array to torch tensor if the input was a torch tensor
    if is_tensor:
        labeled_array_all = torch.from_numpy(labeled_array_all)
        labeled_array_all = labeled_array_all.to(device)
        area_map_all = torch.from_numpy(area_map_all)
        area_map_all = area_map_all.to(device)

    labeled_array_all = labeled_array_all.reshape(mask.shape)
    area_map_all = area_map_all.reshape(mask.shape)

    return area_map_all, area_map_all

def union_find(labels):
    # Path compression
    root = np.arange(labels.size)
    change = True
    while change:
        change = False
        zroot = root[labels]
        if not np.all(zroot == root):
            root = zroot
            change = True
    return root

def _get_connected_components(image):
    is_tensor = False
    if isinstance(image, torch.Tensor):
        is_tensor = True
        device = image.device
        image = image.detach().cpu().numpy().squeeze()
    
    H, W = image.shape
    labels = np.arange(H * W).reshape(H, W)  # Initial labels
    
    # Structured array to consider the connectivity (8-connected)
    struct = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    
    # Connected components labeling using structure
    from scipy.ndimage import label
    labeled_image, num_features = label(image, structure=struct)
    
    # Flatten the labels for union-find
    labels_flat = labeled_image.flatten()
    # Union-find to flatten the labels hierarchy
    unique_labels = union_find(labels_flat)
    
    # Reshape back to the original image shape
    new_labels = unique_labels[labels_flat].reshape(H, W)
    
    # Normalize labels
    unique_labels, normalized_labels = np.unique(new_labels, return_inverse=True)

    breakpoint()

    if is_tensor:
        normalized_labels = torch.from_numpy(normalized_labels)
        normalized_labels = normalized_labels.to(device)
        unique_labels = torch.from_numpy(unique_labels)
        unique_labels = unique_labels.to(device)

    return normalized_labels.reshape(H, W), len(unique_labels)


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        # from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        # try:
        if self.max_hole_area > 0:
            # Holes are those connected components in background with area <= self.fill_hole_area
            # (background regions are those with mask scores <= self.mask_threshold)
            labels, areas = get_connected_components(
                mask_flat <= self.mask_threshold
            )
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with a small positive mask score (10.0) to change them to foreground.
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

        if self.max_sprinkle_area > 0:
            labels, areas = get_connected_components(
                mask_flat > self.mask_threshold
            )
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with negative mask score (-10.0) to change them to background.
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        # except Exception as e:
        #     # Skip the post-processing step if the CUDA kernel fails
        #     warnings.warn(
        #         f"{e}\n\nSkipping the post-processing step due to the error above. You can "
        #         "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
        #         "functionality may be limited (which doesn't affect the results in most cases; see "
        #         "https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md).",
        #         category=UserWarning,
        #         stacklevel=2,
        #     )
        #     masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
