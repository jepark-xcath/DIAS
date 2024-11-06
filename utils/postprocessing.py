import cv2
import numpy as np 
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_opening, disk

def remove_smalls(raw_data, min_size=64):
    """
    Remove small objects from the binary mask.
    Args:
        raw_data (np.array): Binary mask.
        min_size (int): Minimum size of the objects to keep.
    Returns:
        np.array: Binary mask with small objects removed.
    """
    return remove_small_objects(
        raw_data.astype(np.bool_), min_size=min_size).astype(np.uint8)

def remove_small_vessles(
    blood_vessel_image: np.array, kernel: int = 2
) -> np.array:
    """
    Remove small vessels from a binary blood vessel image.
    Args:
        blood_vessel_image (np.array): Binary blood vessel image.
        min_width (int): Minimum width of small vessels to remove.
    Returns:
        np.array: Binary blood vessel image with small vessels removed.
    """
    binary_image = ~binary_opening(blood_vessel_image, disk(kernel))
    binary_image = blood_vessel_image - binary_image
    return (binary_image > 0).astype(np.uint8)

def keep_largest_connected_component(data):
    """
    Keep the largest connected component in the binary mask.
    Args:
        data (np.array): Binary mask.
    Returns:
        np.array: Binary mask with only the largest connected component.
    """
    label_image, nums = label(data, connectivity=data.ndim, return_num=True)
    regions = regionprops(label_image)
    if nums > 1:
        # Find the largest component by area
        largest_region = max(regions, key=lambda r: r.area)
        # Create a mask with only the largest component
        data = (label_image == largest_region.label).astype(np.uint8) 
    return data

def get_connect_components(data, min_size=None):
    """
    Keep the largest connected component in the binary mask.
    Args:
        data (np.array): Binary mask.
    Returns:
        np.array: Colored mask for each connected component.
    """
    label_image, nums = label(data, connectivity=1, return_num=True)
    regions = regionprops(label_image)
    colored_data = np.zeros_like(data, dtype=np.uint8)
    if nums > 1: 
        regions = sorted(regions, key=lambda r: r.area, reverse=False)
        count = 1
        for region in regions:
            if min_size is not None and region.area < min_size:
                continue
            colored_data[label_image == region.label] = count
            count += 1
        # Normalize the colored data
        colored_data = (colored_data / count * 255).astype(np.uint8)
        # Apply colormap using OpenCV
        data = cv2.applyColorMap(colored_data, cv2.COLORMAP_HSV)
        # Ensure the background remains black
        data[colored_data == 0] = [0, 0, 0]
    return data
