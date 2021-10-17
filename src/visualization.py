import cv2
import matplotlib.pyplot as plt

import settings
import rle_utils


def visualize_image(df, image_id, visualize_mask, path=None):

    """
    Visualize raw image and segmentation mask

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

    image = cv2.imread(f'{settings.DATA_PATH}/train_images/{image_path}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image, cmap='gray')
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


def visualize_image_and_transforms(df, image_id, transforms=None, path=None):

    """
    Visualize raw and transformed image and segmentation mask

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, >= 2)]: Training dataframe
    image_id (str): Image ID (filename)
    transforms (albumentations.Compose): Transformation to apply image and mask
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image_path = df.loc[df['id'] == image_id, 'id'].values[0]
    image = cv2.imread(f'{settings.DATA_PATH}/train_images/{image_path}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = rle_utils.get_mask(df=df, image_id=image_id, shape=(image.shape[0], image.shape[1]))

    transformed = transforms(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    fig, axes = plt.subplots(figsize=(32, 16), ncols=2)
    axes[0].imshow(image, cmap='gray')
    axes[0].imshow(mask, alpha=0.4)
    axes[1].imshow(transformed_image, cmap='gray')
    axes[1].imshow(transformed_mask, alpha=0.4)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    axes[0].set_title(f'{image_path} Raw Image', size=20, pad=15)
    axes[1].set_title(f'{image_path} Transformed Image', size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
