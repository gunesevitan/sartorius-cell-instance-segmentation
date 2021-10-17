import cv2
import matplotlib.pyplot as plt

import settings
import rle_utils


def visualize_image_from_dataframe(df, image_id, visualize_mask, path=None):

    """
    Visualize image and segmentation mask

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, >= 2)]: Training dataframe
    image_id (str): Image ID (filename)
    visualize_mask (bool): Whether to visualize segmentation mask over image or not
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image_path = df.loc[df['id'] == image_id, 'id'].values[0]
    cell_type = df.loc[df['id'] == image_id, 'cell_type'].values[0]
    plate_time = df.loc[df['id'] == image_id, 'plate_time'].values[0]
    sample_date = df.loc[df['id'] == image_id, 'sample_date'].values[0]
    sample_id = df.loc[df['id'] == image_id, 'sample_id'].values[0]
    elapsed_timedelta = df.loc[df['id'] == image_id, 'elapsed_timedelta'].values[0]

    image = cv2.imread(f'{settings.DATA_PATH}/train/{image_path}.png')

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image)
    if visualize_mask:
        mask = rle_utils.get_mask(df=df, image_id=image_id, shape=(image.shape[0], image.shape[1]))
        ax.imshow(mask, alpha=0.4)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(f'{image_path} - {cell_type} - {plate_time} - {sample_date} - {sample_id} - {elapsed_timedelta}', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
