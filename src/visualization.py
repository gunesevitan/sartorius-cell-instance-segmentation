import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import settings
import mask_utils


def visualize_image(df, image_id, visualize_mask=True, path=None):

    """
    Visualize raw image and segmentation masks

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
        mask = mask_utils.decode_and_add_rle_masks(df=df, image_id=image_id, shape=image.shape)
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


def visualize_transforms(df, image_id, transforms=None, path=None):

    """
    Visualize raw and transformed image and segmentation mask

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, >= 2)]: Training dataframe
    image_id (str): Image ID (filename)
    transforms (albumentations.Compose): Transformations to apply image, mask and bounding boxes
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image_path = df.loc[df['id'] == image_id, 'id'].values[0]
    raw_image = cv2.imread(f'{settings.DATA_PATH}/train_images/{image_path}.png')
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    raw_mask = mask_utils.decode_and_add_rle_masks(df=df, image_id=image_id, shape=raw_image.shape)

    encoded_masks = df.loc[df['id'] == image_id, 'annotation'].values
    masks = []
    boxes = []
    labels = []

    for encoded_mask in encoded_masks:
        decoded_mask = mask_utils.decode_rle_mask(rle_mask=encoded_mask, shape=raw_image.shape)
        bounding_box = mask_utils.get_bounding_box(decoded_mask)
        masks.append(decoded_mask)
        boxes.append(bounding_box)
        labels.append(1)

    transformed = transforms(image=raw_image, masks=masks, bboxes=boxes, labels=labels)
    transformed_image = transformed['image']
    transformed_masks = np.any(np.stack(transformed['masks']), axis=0)

    fig, axes = plt.subplots(figsize=(32, 16), ncols=2)
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].imshow(raw_mask, alpha=0.4)
    axes[1].imshow(transformed_image, cmap='gray')
    axes[1].imshow(transformed_masks, alpha=0.4)

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


def visualize_learning_curve(training_losses, validation_losses, title, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses [array-like of shape (n_epochs)]: Array of training losses computed after every epoch
    validation_losses [array-like of shape (n_epochs)]: Array of validation losses computed after every epoch
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 8), dpi=100)
    sns.lineplot(
        x=np.arange(1, len(training_losses) + 1),
        y=training_losses,
        ax=ax,
        label='train_loss'
    )
    sns.lineplot(
        x=np.arange(1, len(validation_losses) + 1),
        y=validation_losses,
        ax=ax,
        label='val_loss'
    )
    ax.set_xlabel('Epochs', size=15, labelpad=12.5)
    ax.set_ylabel('Loss', size=15, labelpad=12.5)
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.legend(prop={'size': 18})
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)
