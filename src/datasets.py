import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import settings
import mask_utils


class InstanceSegmentationDataset(Dataset):

    def __init__(self, images, masks=None, transforms=None):

        self.images = images
        self.masks = masks
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
        image [torch.FloatTensor of shape (depth, height, width)]: Image
        target (dict or None):
        Dictionary contains:
            - masks [torch.FloatTensor of shape (n_objects, height, width)]: Segmentation masks
            - boxes [torch.FloatTensor of shape (n_objects, 4)]: Bounding boxes in VOC format
            - labels [torch.Int64Tensor of shape (n_objects)]: Labels of objects
            - image_id [torch.Int64Tensor of shape (1)]: Image identifier used during evaluation
            - area [torch.FloatTensor of shape (n_objects)]: Areas of the bounding boxes
            - iscrowd [torch.UInt8Tensor of shape (n_objects)]: Instances with iscrowd=True will be ignored during evaluation
        """

        image = cv2.imread(f'{settings.DATA_PATH}/train_images/{self.images[idx]}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.masks is not None:

            masks = []
            boxes = []
            labels = []

            for mask in self.masks[idx]:
                decoded_mask = mask_utils.decode_rle_mask(rle_mask=mask, shape=image.shape)
                bounding_box = mask_utils.get_bounding_box(decoded_mask)
                masks.append(decoded_mask)
                boxes.append(bounding_box)
                labels.append(1)

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

    def __init__(self, images, masks=None, transforms=None):

        self.images = images
        self.masks = masks
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
        image [torch.FloatTensor of shape (depth, height, width)]: Image
        mask [torch.FloatTensor of shape (depth, height, width)]: Semantic segmentation mask
        """

        image = cv2.imread(f'{settings.DATA_PATH}/train_images/{self.images[idx]}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.masks is not None:

            masks = []

            for mask in self.masks[idx]:
                decoded_mask = mask_utils.decode_rle_mask(rle_mask=mask, shape=image.shape)
                masks.append(decoded_mask)

            mask = np.any(np.stack(masks), axis=0)

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
