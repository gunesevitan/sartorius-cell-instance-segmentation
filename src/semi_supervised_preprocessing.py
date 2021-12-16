from tqdm import tqdm
from glob import glob
import numpy as np
import cv2

import settings

if __name__ == '__main__':

    train_image_paths = glob(f'{settings.DATA_PATH}/train_images/*.png')
    train_semi_supervised_image_paths = glob(f'{settings.DATA_PATH}/train_semi_supervised_images/*.png')

    train_images = []
    for train_image_path in tqdm(train_image_paths):
        image = cv2.imread(train_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_images.append(image)

    train_semi_supervised_images = []
    for train_semi_supervised_image_path in tqdm(train_semi_supervised_image_paths):
        image = cv2.imread(train_semi_supervised_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        train_semi_supervised_images.append(image)

    duplicates = []
    for train_semi_supervised_image in tqdm(train_semi_supervised_images):
        is_duplicate = np.any(np.all(train_semi_supervised_image == train_images, axis=(1, 2)))
        duplicates.append(is_duplicate)
