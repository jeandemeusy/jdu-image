import cv2 as cv
import numpy as np

from typing import Union, List
from pathlib import Path


class ImageStructure:
    """General purpose high level opencv image class. All operation on instances are inplace."""

    def __init__(self, input: Union[str, np.ndarray]):
        """Initialisation of the class.

        Parameters
        ----------
        input : str, np.ndarray
            Either the path to an image file, or a numpy array.
        """
        if not isinstance(input, str) and not isinstance(input, np.ndarray):
            raise TypeError("Bad type")

        if isinstance(input, str):
            self.data = self.__load_image(input)

        elif isinstance(input, np.ndarray):
            self.data = self.__convert_array(input)

    @property
    def shape(self) -> List[int]:
        """Shape of the image."""
        return self.data.shape

    @property
    def width(self) -> int:
        """Width of the image"""
        return self.shape[1]

    @property
    def height(self) -> int:
        """Height of the image"""
        return self.shape[0]

    @property
    def dim(self) -> int:
        """Number of dimensions of the image."""
        return len(self.shape)

    def __load_image(self, path: str) -> np.ndarray:
        """Loads the image from a file, as a color image (BGR).

        Parameters
        ----------
        path: str
            path to input image

        Returns
        -------
        np.ndarray
            Loaded image
        """
        if not Path(path).exists():
            raise ValueError(f"image at {path} not found")

        return cv.imread(path, cv.IMREAD_COLOR)

    def __convert_array(self, array: np.ndarray) -> np.ndarray:
        """Converts an array. For the moments only checks the array type and returns it.

        Parameters
        ----------
        array: np.ndarray
            Input array

        Returns
        -------
        np.ndarray
            Converted array
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Wrong image type for conversion")

        return array
