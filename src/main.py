"""
Transforms text image to dataframe
"""
import os

import cv2
import numpy as np

from preprocessing import get_file_path, remove_noise, remove_colorful_components, \
    generate_dataframe

if __name__ == '__main__':
    X1_KERNEL = np.ones((1, 1), np.uint8)

    file_path = get_file_path()

    image = cv2.imread(file_path)
    image = cv2.medianBlur(image, 3)
    image = cv2.blur(image, (3, 3))
    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = remove_colorful_components(image)
    image = cv2.blur(image, (5, 5))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = remove_noise(image, X1_KERNEL)

    dataframe = generate_dataframe(image)
    with open(os.path.join(os.path.dirname(file_path), 'output.csv'), 'w') as file:
        file.write(dataframe.to_string())
