import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import settings
import annotation_utils


class InstanceSegmentationDataset(Dataset):

    def __init__(self, images, masks=None, labels=None, transforms=None, dataset='raw'):

        self.images = images
        self.masks = masks
        self.labels = labels
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.images))

        Returns
        -------
        image [torch.FloatTensor of shape (channel, height, width)]: Image
        target (dict or None):
        Dictionary contains:
            - masks [torch.FloatTensor of shape (n_objects, height, width)]: Segmentation masks
            - boxes [torch.FloatTensor of shape (n_objects, 4)]: Bounding boxes in VOC format
            - labels [torch.LongTensor of shape (n_objects)]: Labels of objects
            - image_id [torch.LongTensor of shape (1)]: Image identifier used during evaluation
            - area [torch.FloatTensor of shape (n_objects)]: Areas of the bounding boxes
            - iscrowd [torch.UInt8Tensor of shape (n_objects)]: Instances with iscrowd=True will be ignored during evaluation
        """

        if self.dataset == 'raw':
            image = cv2.imread(f'{settings.DATA_PATH}/train_images/{self.images[idx]}.png')
        elif self.dataset == 'livecell':
            image = cv2.imread(f'{settings.DATA_PATH}/livecell_images/{self.images[idx]}.tif')
        else:
            image = None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if (self.masks is not None) and (self.labels is not None):

            masks = []
            boxes = []
            labels = []

            for mask in self.masks[idx]:
                decoded_mask = annotation_utils.decode_rle_mask(rle_mask=mask, shape=image.shape, fill_holes=False, is_coco_encoded=(self.dataset == 'livecell'))
                bounding_box = annotation_utils.mask_to_bounding_box(decoded_mask)
                masks.append(decoded_mask)
                boxes.append(bounding_box)
                labels.append(self.labels[idx])

            if self.transforms is not None:
                transformed = self.transforms(image=image, masks=masks, bboxes=boxes, labels=labels)
                image = transformed['image']
                masks = transformed['masks']
                boxes = transformed['bboxes']
                labels = transformed['labels']

            image_id = torch.tensor([idx])
            boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
            labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

            target = {
                'masks': masks,
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd
            }

            return image, target

        else:

            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed['image']
            else:
                image = np.stack([image] * 3)
                image = torch.as_tensor(image, dtype=torch.float32)

            return image


class SemanticSegmentationDataset(Dataset):

    def __init__(self, images, masks=None, transforms=None, dataset='raw'):

        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.images))

        Returns
        -------
        image [torch.FloatTensor of shape (channel, height, width)]: Image
        mask [torch.FloatTensor of shape (channel, height, width)]: Semantic segmentation mask
        """

        if self.dataset == 'raw':
            image = cv2.imread(f'{settings.DATA_PATH}/train_images/{self.images[idx]}.png')
        elif self.dataset == 'livecell':
            image = cv2.imread(f'{settings.DATA_PATH}/livecell_images/{self.images[idx]}.tif')
        else:
            image = None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.masks is not None:

            masks = []

            for mask in self.masks[idx]:
                decoded_mask = annotation_utils.decode_rle_mask(rle_mask=mask, shape=image.shape, fill_holes=False, is_coco_encoded=(self.dataset == 'livecell'))
                masks.append(decoded_mask)

            masks = np.stack(masks)
            mask = np.any(masks > 1, axis=0)

            if self.transforms is not None:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                image = torch.unsqueeze(image, 0)
                mask = torch.as_tensor(mask, dtype=torch.float)
                mask = torch.unsqueeze(mask, 0)

            return image, mask

        else:

            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed['image']
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                image = torch.unsqueeze(image, 0)

            return image


class ClassificationDataset(Dataset):

    def __init__(self, images, image_directory, targets=None, transforms=None):

        self.images = images
        self.image_directory = image_directory
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.images))

        Returns
        -------
        image [torch.FloatTensor of shape (channel, height, width)]: Image
        target [torch.LongTensor of shape (1)]: Cell type
        """

        image = cv2.imread(f'{settings.DATA_PATH}/{self.image_directory}/{self.images[idx]}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']
        else:
            image = torch.as_tensor(image, dtype=torch.float)
            image = torch.unsqueeze(image, 0)

        if self.targets is not None:
            target = self.targets[idx]
            target = torch.as_tensor(target, dtype=torch.long)
            return image, target
        else:
            return image
