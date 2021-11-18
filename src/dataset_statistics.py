from glob import glob
from tqdm import tqdm
import numpy as np
import cv2

import settings


# Competition raw datasets' mean and standard deviations
COMPETITION_TRAIN_STATISTICS = {'mean': 127.96482685978934, 'std': 13.705081432885466}
COMPETITION_TRAIN_TEST_STATISTICS = {'mean': 127.96497968626201, 'std': 13.686623357801329}
COMPETITION_TRAIN_TEST_SEMI_SUPERVISED_STATISTICS = {'mean': 127.9755108670509, 'std': 12.206579996359626}
COMPETITION_TRAIN_STATISTICS_SCALED = {'mean': 0.5018228504305459, 'std': 0.05374541738386454}
COMPETITION_TRAIN_TEST_STATISTICS_SCALED = {'mean': 0.5018234497500464, 'std': 0.05367303277569167}
COMPETITION_TRAIN_TEST_SEMI_SUPERVISED_STATISTICS_SCALED = {'mean': 0.5018395185470581, 'std': 0.04786895215511322}
# LIVECELL dataset's mean and standard deviations
LIVECELL_STATISTICS = {'mean': 128.01926491911536, 'std': 11.172353318295597}
LIVECELL_PAPER_STATISTICS = {'mean': 128, 'std': 11.58}
LIVECELL_STATISTICS_SCALED = {'mean': 0.5020363330161387, 'std': 0.043813150267825875}
LIVECELL_PAPER_STATISTICS_SCALED = {'mean': 0.5019607843137255, 'std': 0.05146666666666667}


if __name__ == '__main__':

    DATASET = 'livecell'

    if DATASET == 'competition':
        train_images = glob(f'{settings.DATA_PATH}/train_images/*.png')
        test_images = glob(f'{settings.DATA_PATH}/test_images/*.png')
        train_semi_supervised_images = glob(f'{settings.DATA_PATH}/train_semi_supervised_images/*.png')
        images = train_images + test_images + train_semi_supervised_images
    elif DATASET == 'livecell':
        images = glob(f'{settings.DATA_PATH}/livecell_images/*.tif')
    else:
        images = []

    pixel_sums = 0
    pixel_squared_sums = 0

    for image in tqdm(images):

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.float32(image) / 255.

        pixel_sums += np.sum(image, axis=(0, 1))
        pixel_squared_sums += np.sum(image ** 2, axis=(0, 1))

    pixel_count = len(images) * 520 * 704
    mean = pixel_sums / pixel_count
    var = (pixel_squared_sums / pixel_count) - (mean ** 2)
    std = np.sqrt(var)

    print(f'{len(images)} Images - Mean: {mean} - Standard Deviation: {std}')
