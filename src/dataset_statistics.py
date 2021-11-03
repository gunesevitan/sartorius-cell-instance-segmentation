from glob import glob
from tqdm import tqdm
import numpy as np
import cv2

import settings


# Mean and standard deviations below are computed with the code below
TRAIN_STATISTICS = {'mean': 127.96482685978934, 'std': 13.705081432885466}
TRAIN_TEST_STATISTICS = {'mean': 127.96497968626201, 'std': 13.686623357801329}
TRAIN_TEST_SEMI_SUPERVISED_STATISTICS = {'mean': 127.9755108670509, 'std': 12.206579996359626}


if __name__ == '__main__':

    train_images = glob(f'{settings.DATA_PATH}/train_images/*.png')
    test_images = glob(f'{settings.DATA_PATH}/test_images/*.png')
    train_semi_supervised_images = glob(f'{settings.DATA_PATH}/train_semi_supervised_images/*.png')

    images = []
    for image in tqdm(train_images + test_images + train_semi_supervised_images):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)

    images = np.stack(images)
    image_mean = np.mean(images)
    image_std = np.std(images)
    print(f'{images.shape[0]} Images - Mean: {image_mean} - Standard Deviation: {image_std}')
