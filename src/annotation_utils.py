import numpy as np
from scipy import ndimage


def decode_rle_mask(rle_mask, shape, fill_holes=False):

    """
    Decode run-length encoded mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded mask string
    shape (tuple): Height and width of the mask
    fill_holes (bool): Whether to fill holes in masks or not

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d mask
    """

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


def binary_to_multi_class_mask(binary_masks):

    """
    Encode multiple 2d binary segmentation masks into a single 2d multi-class segmentation mask

    Parameters
    ----------
    binary_masks [numpy.ndarray of shape (n_objects, height, width)]: 2d binary segmentation masks

    Returns
    -------
    multi_class_mask [numpy.ndarray of shape (height, width)]: 2d multi-class segmentation mask
    """

    multi_class_mask = np.zeros((binary_masks.shape[1], binary_masks.shape[2]))
    for i, binary_mask in enumerate(binary_masks):
        non_zero_idx = binary_mask == 1
        multi_class_mask[non_zero_idx] = i + 1

    return multi_class_mask


def mask_to_bounding_box(mask):

    """
    Get bounding box from a segmentation mask

    Parameters
    ----------
    mask [numpy.ndarray of shape (height, width)]: 2d segmentation mask

    Returns
    -------
    bounding_box [list of shape (4)]: Bounding box of the instance segmentation mask
    """

    non_zero_idx = np.where(mask == 1)
    bounding_box = [
        np.min(non_zero_idx[1]),
        np.min(non_zero_idx[0]),
        np.max(non_zero_idx[1]),
        np.max(non_zero_idx[0])
    ]

    return bounding_box
