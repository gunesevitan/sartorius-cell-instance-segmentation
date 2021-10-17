import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import settings


class SegmentationDataset(Dataset):

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
        mri [torch.FloatTensor of shape (channel, depth, height, width)]: 4D mpMRI
        mask [torch.FloatTensor of shape (channel, depth, height, width)]: 4D segmentation mask
        """

        image = cv2.imread(f'{settings.DATA_PATH}/train_images/{self.images[idx]}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.masks == 'npy':

            mask = np.load(f'{settings.DATA_PATH}/train_masks/{self.images[idx]}.npy')

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

        elif self.masks is None:

            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed['image']
            else:
                image = torch.as_tensor(image, dtype=torch.float)
                image = torch.unsqueeze(image, 0)

            return image
