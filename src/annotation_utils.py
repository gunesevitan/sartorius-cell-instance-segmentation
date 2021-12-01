import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure
from shapely.geometry import Polygon
import pycocotools.mask as mask_utils


def decode_rle_mask(rle_mask, shape, fill_holes=False, is_coco_encoded=False):

    """
    Decode run-length encoded mask string into 2d binary mask array

    Parameters
    ----------
    rle_mask (str): Run-length encoded mask string
    shape (tuple): Height and width of the mask
    fill_holes (bool): Whether to fill holes in masks or not
    is_coco_encoded (bool): Whether the mask is encoded with pycocotools or not

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d mask
    """

    if is_coco_encoded:
        # Decoding RLE encoded mask of string
        mask = np.uint8(mask_utils.decode({'size': shape, 'counts': rle_mask}))
    else:
        # Decoding RLE encoded mask of integers
        rle_mask = rle_mask.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
        starts -= 1
        ends = starts + lengths

        mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
        for start, end in zip(starts, ends):
            mask[start:end] = 1

        mask = mask.reshape(shape[0], shape[1])

    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    return mask


def encode_rle_mask(mask):

    """
    Encode 2d binary mask array into run-length encoded mask string

    Parameters
    ----------
    mask [numpy.ndarray of shape (height, width)]: 2d mask

    Returns
    -------
    rle_mask (str): Run-length encoded mask string
    """

    mask = mask.flatten()
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def is_broken(mask, horizontal_line_threshold=50):

    """
    Check whether the masks have perfect horizontal lines with given threshold

    Parameters
    ----------
    mask [numpy.ndarray of shape (height, width)]: 2d mask
    horizontal_line_threshold (int): Number of horizontal line pixels

    Returns
    -------
    is_broken (bool): Whether the mask annotation is broken or not
    """

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = contours[0][:, 0]
    diff = c - np.roll(c, 1, 0)
    targets = (diff[:, 1] == 0) & (np.abs(diff[:, 0]) >= horizontal_line_threshold)

    if np.any(targets):
        if np.all(c[targets][:, 1] == 0) or np.all(c[targets][:, 1] == 519):
            return False
        else:
            return True
    else:
        return False


def binary_to_multi_class_mask(binary_masks):

    """
    Encode multiple 2d binary masks into a single 2d multi-class segmentation mask

    Parameters
    ----------
    binary_masks [numpy.ndarray of shape (n_objects, height, width)]: 2d binary masks

    Returns
    -------
    multi_class_mask [numpy.ndarray of shape (height, width)]: 2d multi-class mask
    """

    multi_class_mask = np.zeros((binary_masks.shape[1], binary_masks.shape[2]))
    for i, binary_mask in enumerate(binary_masks):
        non_zero_idx = binary_mask == 1
        multi_class_mask[non_zero_idx] = i + 1

    return multi_class_mask


def mask_to_bounding_box(mask):

    """
    Get bounding box from a binary mask

    Parameters
    ----------
    mask [numpy.ndarray of shape (height, width)]: 2d binary mask

    Returns
    -------
    bounding_box [list of shape (4)]: Bounding box of the object
    """

    non_zero_idx = np.where(mask == 1)
    bounding_box = [
        np.min(non_zero_idx[1]),
        np.min(non_zero_idx[0]),
        np.max(non_zero_idx[1]),
        np.max(non_zero_idx[0])
    ]

    return bounding_box


def mask_to_polygon(mask):

    """
    Get polygon from a binary mask

    Parameters
    ----------
    mask [numpy.ndarray of shape (height, width)]: 2d binary mask

    Returns
    -------
    polygons [list of shape (n_objects)]: Polygons of the objects
    """

    contours = measure.find_contours(
        mask,
        level=0.5,
        fully_connected='low',
        positive_orientation='low'
    )
    segmentations = []
    polygons = []

    for obj in contours:

        obj -= 1
        obj = np.flip(obj, axis=1)

        if len(obj) > 2:

            polygon = Polygon(obj)
            polygon = polygon.simplify(tolerance=0.5, preserve_topology=True)
            polygons.append(polygon)

            if polygon.is_empty is False:
                try:
                    segmentation = np.array(polygon.exterior.coords).ravel().tolist()
                    segmentation = np.clip(segmentation, a_min=0, a_max=np.max(mask.shape) - 1).tolist()
                    segmentations.append(segmentation)
                except:
                    continue

    return segmentations, polygons


def polygon_to_mask(polygon, shape):

    """
    Get binary mask from a polygon

    Parameters
    ----------
    polygon [list of shape (n_points)]: Polygon

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: 2d binary mask
    """

    # Convert numpy.array to list of tuple pairs of X and Y coordinates
    points = np.array(polygon).reshape(-1).reshape(-1, 2)
    points = [(point[0], point[1]) for point in points]
    mask = Image.new('L', (shape[1], shape[0]), 0)

    # Draw mask from the polygon
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    mask = np.array(mask).astype(np.uint8)

    return mask
