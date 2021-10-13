from __future__ import annotations
from typing import List

import cv2 as cv

from ._imageStructure import ImageStructure
from ._generalMixin import GeneralMixin
from ._thresholdMixin import ThresholdMixin
from ._morphologicalMixin import MorphologicalMixin
from ._circleMixin import CircleMixin
from ._colorMixin import ColorMixin
from ._outputMixin import OutputMixin
from ._toClassifyMixin import ToClassifyMixin


class Image(
    ImageStructure,
    GeneralMixin,
    ThresholdMixin,
    MorphologicalMixin,
    CircleMixin,
    ColorMixin,
    OutputMixin,
    ToClassifyMixin,
):
    """
    High level image processing class to handle import, transforms, display and saving of images. All transformations are opencv/numpy based so that the Python to C++ conversion is easy.
    """

    def __init__(self, input):
        super().__init__(input)

    def deepcopy(self) -> Image:
        """Creates a deep copy of the image, independant from the source.

        Returns
        -------
        Image
            Deepcopy of the input image
        """
        return Image(self.data)

    def channels(self):
        chans = cv.split(self.data)
        return [Image(c) for c in chans]

    def blend(self, other: Image, alpha: float) -> None:
        """Blends two images together.

        Parameters
        ----------
        other: Image
            Image to blend with
        alpha: float
            Transparency coefficient

        Returns
        -------
        Image
            Blended image
        """
        if not isinstance(other, Image):
            raise TypeError("Wrong input image type")
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be in [0, 1]")

        img1 = self.deepcopy()
        img2 = other.deepcopy()

        img1.to_BGR()
        img2.to_BGR()

        self.data = cv.addWeighted(img1.data, 1 - alpha, img2.data, alpha, 0.0)

    def crop(self, tl: List[int], br: List[int], inplace: bool = True) -> None or Image:
        """Crops the image from top-left (tl) to bottom-right (br) corners.

        Parameters
        ----------
        tl : list of int
            Top left corner of the ROI (2 elements array)
        br : list of int
            Bottom right corner of the ROI (2 elements array)
        inplace: bool, optional
            flag to make to operation inplace (default is True)
        """

        if tl[0] < 0 or tl[1] < 0:
            raise ValueError("Wrong tl")
        if br[0] > self.height or br[1] > self.width:
            raise ValueError("Wrong br")
        if tl[0] >= br[0] or tl[1] >= br[1]:
            raise ValueError("br must be bigger than tl")

        cropped = self.data[tl[0] : br[0] + 1, tl[1] : br[1] + 1]
        if inplace:
            self.data = cropped
        else:
            return Image(cropped)

    def crop2(
        self, tl: List[int], height: int, width: int, inplace: bool = True
    ) -> None or Image:
        """Crops the image from top_left (tl) with given height and width.

        Parameters
        ----------
        tl : list of int
            Top left corner of the ROI (2 elements array)
        height: int
            height of the cropped image
        width: int
            width of the cropped image
        inplace: bool, optional
            flag to make to operation inplace (default is True)
        """
        if tl[0] < 0 or tl[1] < 0:
            raise ValueError("Wrong tl")

        cropped = self.data[tl[0] : tl[0] + height, tl[1] : tl[1] + width]
        if inplace:
            self.data = cropped
        else:
            return Image(cropped)
