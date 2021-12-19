import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

import settings
import annotation_utils


if __name__ == '__main__':

    df = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
    df = df.loc[~df['annotation'].isnull(), :]
    df = df.groupby('id')['annotation_filled'].agg(lambda x: list(x)).reset_index()

    Path(f'{settings.DATA_PATH}/train_images_clean').mkdir(parents=True, exist_ok=True)

    for idx in tqdm(df['id'].index):

        image_id = df.loc[idx, 'id']
        image = cv2.imread(f'{settings.DATA_PATH}/train_images/{image_id}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        masks = []
        for rle_mask in df.loc[idx, 'annotation_filled']:
            decoded_mask = annotation_utils.decode_rle_mask(
                rle_mask,
                shape=image.shape,
                fill_holes=False,
                is_coco_encoded=False
            )
            masks.append(decoded_mask)
        masks = np.stack(masks)

        broken_masks = []
        for mask in masks:
            broken_masks.append(annotation_utils.is_broken(mask, horizontal_line_threshold=50))

        if bool(np.any(broken_masks)) is False:
            # Copy image from train_images to train_images_clean
            shutil.copy2(f'{settings.DATA_PATH}/train_images/{image_id}.png', f'{settings.DATA_PATH}/train_images_clean/{image_id}.png')
        else:
            # Remove objects with broken masks from the image and save it to train_images_clean
            foreground = np.any(masks > 0, axis=0)
            image_background_mean = image[~foreground].mean()

            for broken_mask in masks[np.array(broken_masks)]:
                # Use dilation to make masks slightly larger because of the white outline
                broken_mask = cv2.dilate(broken_mask, kernel=np.ones((3, 3), dtype=np.uint8))
                image[broken_mask > 0] = np.random.randint(
                    low=image_background_mean - 3,
                    high=image_background_mean + 3,
                    size=broken_mask.sum()
                )

            cv2.imwrite(f'{settings.DATA_PATH}/train_images_clean/{image_id}.png', image)
