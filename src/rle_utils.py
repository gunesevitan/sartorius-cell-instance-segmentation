import numpy as np


def _decode_rle_mask(rle_mask, shape):

    """
    Decode run-length encoded segmentation mask string into 2d array

    Parameters
    ----------
    rle_mask (str): Run-length encoded segmentation mask string
    shape (tuple): Height and width of the mask

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
    """

    rle_mask = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle_mask[0:][::2], rle_mask[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape(shape[0], shape[1])


def get_mask(df, image_id, shape):

    """
    Decode all run-length encoded segmentation masks of a given image and merge them into a 2d array

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, >= 2)]: Dataframe with id and annotation columns
    image_id (str): Image ID (filename)
    shape (tuple): Height and width of the mask

    Returns
    -------
    mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
    """

    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    rle_masks = df.loc[df['id'] == image_id, 'annotation'].values
    for rle_mask in rle_masks:
        mask += _decode_rle_mask(rle_mask=rle_mask, shape=shape)

    mask[mask >= 1] = 1

    return mask
