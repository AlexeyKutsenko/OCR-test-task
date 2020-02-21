"""
Tests
"""
from argparse import ArgumentParser
from unittest.mock import patch

import cv2
import pytest
from pandas.tests.extension.numpy_.test_numpy_nested import np

from preprocessing import is_valid_filepath, generate_dataframe


def test_is_valid_filepath_function():
    """
    Tests is_valid_filepath function
    """
    parser = ArgumentParser()
    with pytest.raises(SystemExit):
        is_valid_filepath(parser, 'wrong_file_path')

    with patch('os.path.exists') as mock_exists:
        mocked_file_path = 'wrong_file_path'
        mock_exists.return_value = True
        file_path = is_valid_filepath(parser, mocked_file_path)
        assert file_path is mocked_file_path


def test_generate_csv():
    """
    Tests generate_csv
    """
    image_weight = image_height = 200
    left = 10
    top = 100
    image = np.zeros((image_weight, image_height, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, 'XXX', (left, top), font, 3, (255, 255, 255), 5, cv2.LINE_AA)

    output = generate_dataframe(img=image)
    assert len(output) == 1

    row = output.iloc[0]
    assert left / image_weight < row.x_min
    assert top / image_height < row.y_min
