import numpy as np
import cv2
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2


class Scale(ImageOnlyTransform):

    def apply(self, image, **kwargs):

        """
        Scale pixel values between 0 and 1

        Parameters
        ----------
        image [numpy.ndarray of shape (height, width)]: 2d grayscale image

        Returns
        -------
        image [numpy.ndarray of shape (height, width)]: 2d grayscale image divided by max 8 bit integer
        """

        return np.float32(image / 255.)


class ToRGB(ImageOnlyTransform):

    def apply(self, image, **kwargs):

        """
        Convert 2d grayscale image to 2d RGB image

        Parameters
        ----------
        image [numpy.ndarray of shape (height, width)]: 2d grayscale image

        Returns
        -------
        image [numpy.ndarray of shape (height, width, channel)]: 2d grayscale image with channel dimension
        """

        image = np.stack([image] * 3)
        return np.moveaxis(image, 0, -1)


def get_instance_segmentation_transforms(**transform_parameters):

    """
    Get transforms for classification dataset

    Parameters
    ----------
    transform_parameters (dict): Dictionary of transform parameters

    Returns
    -------
    transforms (dict): Transforms of training and test sets
    """

    bbox_params = {'format': 'pascal_voc',  'label_fields': ['labels'], 'min_area': 0, 'min_visibility': 0}

    train_transforms = A.Compose([
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        A.RandomRotate90(p=transform_parameters['random_rotate_90_probability']),
        A.ShiftScaleRotate(
            shift_limit=transform_parameters['shift_limit'],
            scale_limit=transform_parameters['scale_limit'],
            rotate_limit=transform_parameters['rotate_limit'],
            p=transform_parameters['shift_scale_rotate_probability'],
            border_mode=cv2.BORDER_REFLECT
        ),
        A.RandomBrightnessContrast(
            brightness_limit=transform_parameters['brightness_limit'],
            contrast_limit=transform_parameters['contrast_limit'],
            brightness_by_max=True,
            p=transform_parameters['brightness_contrast_probability']
        ),
        Scale(always_apply=True),
        ToRGB(always_apply=True),
        ToTensorV2(always_apply=True)
    ], bbox_params=A.BboxParams(**bbox_params))

    val_transforms = A.Compose([
        Scale(always_apply=True),
        ToRGB(always_apply=True),
        ToTensorV2(always_apply=True)
    ], bbox_params=A.BboxParams(**bbox_params))

    test_transforms = A.Compose([
        Scale(always_apply=True),
        ToRGB(always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    instance_segmentation_transforms = {'train': train_transforms, 'val': val_transforms, 'test': test_transforms}
    return instance_segmentation_transforms


def get_classification_transforms(**transform_parameters):

    """
    Get transforms for classification dataset

    Parameters
    ----------
    transform_parameters (dict): Dictionary of transform parameters

    Returns
    -------
    transforms (dict): Transforms of training and test sets
    """

    train_transforms = A.Compose([
        A.HorizontalFlip(p=transform_parameters['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_parameters['vertical_flip_probability']),
        ToRGB(always_apply=True),
        A.Normalize(mean=transform_parameters['normalize']['mean'], std=transform_parameters['normalize']['std'], always_apply=True),
        A.CoarseDropout(
            max_holes=transform_parameters['coarse_dropout']['max_holes'],
            max_height=transform_parameters['coarse_dropout']['max_height'],
            max_width=transform_parameters['coarse_dropout']['max_width'],
            min_holes=transform_parameters['coarse_dropout']['min_holes'],
            min_height=None,
            min_width=None,
            fill_value=0,
            mask_fill_value=None,
            p=transform_parameters['coarse_dropout']['probability']
        ),
        ToTensorV2(always_apply=True)
    ])

    val_transforms = A.Compose([
        ToRGB(always_apply=True),
        A.Normalize(mean=transform_parameters['normalize']['mean'], std=transform_parameters['normalize']['std'], always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    classification_transforms = {'train': train_transforms, 'val': val_transforms}
    return classification_transforms
