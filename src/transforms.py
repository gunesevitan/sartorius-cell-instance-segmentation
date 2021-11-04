import numpy as np
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


def get_instance_segmentation_transforms(**kwargs):

    bbox_params = {'format': 'pascal_voc',  'label_fields': ['labels'], 'min_area': 0, 'min_visibility': 0}

    train_transforms = A.Compose([
        A.HorizontalFlip(p=kwargs['horizontal_flip_probability']),
        A.VerticalFlip(p=kwargs['vertical_flip_probability']),
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


def get_classification_transforms(**kwargs):

    train_transforms = A.Compose([
        A.HorizontalFlip(p=kwargs['horizontal_flip_probability']),
        A.VerticalFlip(p=kwargs['vertical_flip_probability']),
        ToRGB(always_apply=True),
        A.Normalize(mean=kwargs['normalize']['mean'], std=kwargs['normalize']['std'], always_apply=True),
        A.CoarseDropout(
            max_holes=kwargs['coarse_dropout']['max_holes'],
            max_height=kwargs['coarse_dropout']['max_height'],
            max_width=kwargs['coarse_dropout']['max_width'],
            min_holes=kwargs['coarse_dropout']['min_holes'],
            min_height=None,
            min_width=None,
            fill_value=0,
            mask_fill_value=None,
            p=kwargs['coarse_dropout']['probability']
        ),
        ToTensorV2(always_apply=True)
    ])

    val_transforms = A.Compose([
        ToRGB(always_apply=True),
        A.Normalize(mean=kwargs['normalize']['mean'], std=kwargs['normalize']['std'], always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    classification_transforms = {'train': train_transforms, 'val': val_transforms}
    return classification_transforms
