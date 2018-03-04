"""Preprocess: our raw images by resizing grayscale."""

import os
import cv2
import numpy as np
import math

def main():
    for data_type in ['test', 'train']:
        # the folders to contain our data
        raw_filepath = os.path.join('data', data_type, 'raw')
        proc_filepath = os.path.join('data', data_type, 'proc')

        for label in os.listdir(raw_filepath):
            raw_label_filepath = os.path.join(raw_filepath, label)
            proc_label_filepath = os.path.join(proc_filepath, label)

            num_images = len(os.listdir(raw_label_filepath))

            # np.arrays that we will fill with out image/label data
            images = np.zeros((num_images, 784), dtype=np.float32)
            labels = np.zeros((num_images, 1), dtype=np.int32)

            # process each digit 0-9 one at a time
            for image_name in os.listdir(raw_label_filepath):
                if image_name == '.gitkeep':
                    continue

                # load the image as a grayscale
                img = cv2.imread(os.path.join(raw_label_filepath, image_name), cv2.IMREAD_GRAYSCALE)

                # resize
                img = cv2.resize(img, (138, 138))  # allow stretch for resize
                img = img[10:-10][10:-10]  # trim images

                # save processed images
                if not os.path.exists(proc_label_filepath):
                    os.makedirs(proc_label_filepath)
                cv2.imwrite(os.path.join(proc_label_filepath, image_name), img)

    print('Done!')

if __name__ == '__main__':
    main()
