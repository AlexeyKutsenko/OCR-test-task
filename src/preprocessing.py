"""
Images pre-processing helpers
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import pytesseract


def is_valid_filepath(parser=None, filepath=None):
    """
    Checks if the file path is a valid
    """
    if not os.path.exists(filepath):
        parser.error('File {} does not exists'.format(filepath))
    return filepath


def generate_dataframe(img=None):
    """
    Generate a dataframe from pytesseract output.
    """
    tesseract_output = pytesseract.image_to_data(img,
                                                 lang='swe+eng',
                                                 output_type=pytesseract.Output.DICT)
    zipped_output = zip(tesseract_output.get('left'),
                        tesseract_output.get('top'),
                        tesseract_output.get('width'),
                        tesseract_output.get('height'),
                        tesseract_output.get('text'))

    img_height = img.shape[0]
    img_width = img.shape[1]

    data = ((left / img_width,
             (top + height) / img_height,
             (left + width) / img_width,
             top / img_height,
             text)
            for left, top, width, height, text in zipped_output if text)

    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'text']
    return pd.DataFrame(data, columns=columns)


def remove_noise(src=None, kernel=None):
    """
    Removes thin dots and lines from the image (kernel determs how thin they must be)
    """
    src = cv2.dilate(src, kernel, iterations=1)
    src = cv2.erode(src, kernel, iterations=1)
    # smoothing edges
    return cv2.bilateralFilter(src, 9, 75, 75)


def get_file_path():
    """
    Parses arguments for finding source image
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath',
                        metavar='filepath',
                        help='Path to image file',
                        type=lambda filepath: is_valid_filepath(parser, filepath))
    args = parser.parse_args()
    return args.filepath


def remove_colorful_components(img=None):
    """
    Removes non-black components from the image
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.threshold(img, 105, 255, cv2.THRESH_BINARY)[1]
    img = cv2.erode(img, np.ones((1, 2), np.uint8), iterations=1)
    return img
