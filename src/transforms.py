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

    return {'train': train_transforms, 'val': val_transforms, 'test': test_transforms}


def get_classification_transforms(**kwargs):

    train_transforms = A.Compose([
        A.HorizontalFlip(p=kwargs['horizontal_flip_probability']),
        A.VerticalFlip(p=kwargs['vertical_flip_probability']),
        Scale(always_apply=True),
        ToRGB(always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    val_test_transforms = A.Compose([
        Scale(always_apply=True),
        ToRGB(always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    return {'train': train_transforms, 'val_test': val_test_transforms}
