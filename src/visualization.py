import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import settings
import annotation_utils
import inference_utils


def _draw_bounding_box(image, bounding_box):

    """
    Draw given bounding box on the given image

    Parameters
    ----------
    image [numpy.ndarray of shape (height, width, depth)]: Image
    bounding_box [numpy.ndarray of shape (4)]: Single bounding box coordinates in PASCAL VOC format
    label (str): Label of the single bounding box

    Returns
    -------
    image [np.ndarray of shape (height, width, channel)]: Image with bounding box and its label drawn on it
    """

    x_min, y_min, x_max, y_max = np.uint16(bounding_box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
    return image


def visualize_image(df, image_id, visualize_mask=True, path=None):

    """
    Visualize raw image along with segmentation masks

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
        mask = annotation_utils.decode_and_add_rle_masks(df=df, image_id=image_id, shape=image.shape)
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
    Visualize raw and transformed image along with segmentation masks

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
    raw_mask = annotation_utils.decode_and_add_rle_masks(df=df, image_id=image_id, shape=raw_image.shape)

    encoded_masks = df.loc[df['id'] == image_id, 'annotation'].values
    masks = []
    boxes = []
    labels = []

    for encoded_mask in encoded_masks:
        decoded_mask = annotation_utils.decode_rle_mask(rle_mask=encoded_mask, shape=raw_image.shape)
        bounding_box = annotation_utils.get_bounding_box(decoded_mask)
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


def visualize_predictions(df, image_id, model, nms_iou_thresholds, score_thresholds, label_threshold, transforms=None, path=None):

    """
    Visualize transformed image along with ground-truth and predicted segmentation masks

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, >= 2)]: Training dataframe
    image_id (str): Image ID (filename)
    model (torch.nn.Module): Model used for inference
    nms_iou_threshold (float): Threshold for non-maximum suppression (0 <= nms_iou_threshold <= 1)
    score_threshold (float): Threshold for confidence scores (0 <= score_threshold <= 1)
    label_threshold (float): Threshold for converting probabilities to labels (0 <= label_threshold <= 1)
    transforms (albumentations.Compose): Transformations to apply image, mask and bounding boxes
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    image_path = df.loc[df['id'] == image_id, 'id'].values[0]
    image = cv2.imread(f'{settings.DATA_PATH}/train_images/{image_path}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_image = transforms(image=image)['image']

    ground_truth_masks = []
    for mask in df.loc[df['id'] == image_id, 'annotation'].values:
        decoded_mask = annotation_utils.decode_rle_mask(rle_mask=mask, shape=image.shape)
        ground_truth_masks.append(decoded_mask)
    ground_truth_masks = np.stack(ground_truth_masks)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    prediction = inference_utils.predict_single_image(
        image=transformed_image,
        model=model,
        device=device,
        nms_iou_thresholds=nms_iou_thresholds,
        score_thresholds=score_thresholds,
        verbose=True
    )
    prediction_masks = prediction['masks'].reshape(-1, transformed_image.shape[1], transformed_image.shape[2])
    prediction_masks = np.uint8(prediction_masks > label_threshold)
    average_precision = inference_utils.get_average_precision(
        ground_truth_masks=ground_truth_masks,
        prediction_masks=prediction_masks,
        verbose=True
    )
    prediction_mask = np.any(prediction_masks == 1, axis=0).astype(np.uint8)
    ground_truth_mask = np.any(ground_truth_masks == 1, axis=0).astype(np.uint8)

    fig, axes = plt.subplots(figsize=(32, 16), ncols=2)
    axes[0].imshow(image, cmap='gray')
    axes[0].imshow(ground_truth_mask, alpha=0.4)
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(prediction_mask, alpha=0.4)

    for i in range(2):
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=15, pad=10)
        axes[i].tick_params(axis='y', labelsize=15, pad=10)

    axes[0].set_title(f'{image_path} Ground-truth Segmentation Mask', size=20, pad=15)
    axes[1].set_title(f'{image_path} Prediction Segmentation Mask AP: {average_precision:.4f}', size=20, pad=15)

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
