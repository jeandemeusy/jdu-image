import cv2 as cv
import numpy as np

from typing import List


class GeneralMixin:
    def channel(self, c: int) -> None:
        """Selects one channel among the image's.

        Parameters
        ----------
        c : int
            Channels index to access
        """
        if self.dim != 2:
            channels = cv.split(self.data)
            self.data = channels[c]

    def split(self, direction: str = "h", position: int or str = "mid") -> None:
        """Split the image.

        Parameters
        ----------
        direction: {"h","horizontal","v","vertical"}
            split direction
        position int or str
            index where to split, or "mid" if the split has to be centered (default is "mid")
        """
        if direction not in ["h", "horizontal", "v", "vertical"]:
            raise ValueError("Wrong orientation")

        tl1 = [0, 0]
        br2 = [self.height, self.width]

        if direction in ["h", "horizontal"]:
            if position == "mid":
                position = self.height // 2
            br1 = [position, self.width]

            tl2 = [position, 0]

        if direction in ["v", "vertical"]:
            if position == "mid":
                position = self.width // 2
            br1 = [self.height, position]
            tl2 = [0, position]

        first = self.crop(tl1, br1, False)
        secon = self.crop(tl2, br2, False)

        return first, secon

    def negate(self) -> None:
        """Inverts the image. Only works on 2D images."""
        if self.dim != 2:
            raise ValueError("Negation only on 2D images")

        self.data = ~self.data

    def blur(self, size: int = 5, method: str = "gauss") -> None:
        """Blurs the image. The method and filter size can be chosen.
        Parameters
        ----------
        size : int, optional
            Size of the filter (default is 5)
        method: { "gauss", "average", "median","bilateral"}
            Blurring methods
        """
        if method not in ["gauss", "average", "median", "bilateral"]:
            raise ValueError("Unexpected method")
        if size < 3:
            raise ValueError("Size too small, must be bigger than 3.")
        if size % 2 == 0:
            raise ValueError("Size must be odd")

        if method == "gauss":
            self.data = cv.GaussianBlur(self.data, (size, size), 0)
        elif method == "average":
            self.data = cv.blur(self.data, (size, size))
        elif method == "median":
            self.data = cv.medianBlur(self.data, size)
        elif method == "bilateral":
            self.data = cv.bilateralFilter(self.data, size, 100, 100)

    def gaussian_blur(self, size: int, sigma: float) -> None:
        """Applies gaussian blur to the image.

        Parameters
        ---------
        size: int
            size of the kernel
        sigma: float
            standard deviation of the kernel
        """
        self.data = cv.GaussianBlur(self.data, (size, size), sigma)

    def resize(self, param: str, value: float, inter: int = cv.INTER_AREA) -> None:
        """Resizes the image. When changing height (respectively width), width (respectively height) change so that the ratio stays the same.

        Parameters
        ----------
        param: { "height", "width", "ratio" }
            Which output dimensions will be set
        value: float
            Output dimensions specified by param value
        inter: int
            Interpolation method (default is cv.INTER_AREA)
        """
        if param not in ["width", "height", "ratio"]:
            raise ValueError("Unexpected parameter")
        if value <= 0:
            raise ValueError("Value must be bigger than 0")

        dim = None
        (h, w) = self.shape[:2]

        if param == "height":
            r = int(value) / float(h)
            dim = (int(w * r), int(value))
        elif param == "width":
            r = int(value) / float(w)
            dim = (int(value), int(h * r))
        elif param == "ratio":
            dim = (int(w * value), int(h * value))
        else:
            dim = (w, h)

        self.data = cv.resize(self.data, dim, interpolation=inter)

    def full_resize(self, dim: List[int], inter: int = cv.INTER_AREA) -> None:
        """Resize the image to the given dimensions, without keeping aspect ratio.

        Parameters
        ----------
        dim: list of int
            Desired image dimensions
        inter: int
            openCV interpolation method (default is cv.INTER_AREA)
        """
        self.data = cv.resize(self.data, dim, interpolation=inter)

    def rotate90(self):
        """rotate the image at 90degrees clockwise"""
        self.data = cv.rotate(self.data, cv.ROTATE_90_CLOCKWISE)

    def rotate(self, angle: float, center: List[int]):
        """Rotate the image wtha given angle.

        Parameters
        ----------
        angle: float
            Angle of rotation in degrees
        center: list of int
            (x,y) rotation center coordinate
        """
        rot_mat = cv.getRotationMatrix2D(center, 1.0 * angle, 1.0)
        self.data = cv.warpAffine(
            self.data, rot_mat, self.shape[1::-1], flags=cv.INTER_NEAREST
        )

    def equalize_hist(self, rate: float) -> None:
        """Equalizes the histogram of intensities of the image. It increases the contrast of the image. Only works on 2D images.

        Parameters
        ----------
        rate: float
            Proportion of the histogram to remove to the left and right
        """
        if self.dim != 2:
            raise ValueError("Equalize only on 2D images")

        self.data = cv.equalizeHist(self.data.astype(np.uint8))

        hist = np.cumsum(np.histogram(self.data, 255)[0])
        lowest_value = np.where((rate * hist[-1]) <= hist)[0][0]
        highest_value = np.where(((1 - rate) * hist[-1]) >= hist)[0][-1]

        self.data[self.data < lowest_value] = lowest_value
        self.data[self.data > highest_value] = highest_value

    def distance_transform(self) -> None:
        """Computes distances transformation, i.e. for each black pixel, computes the shortest distances to a white pixel. Only works on 2D images."""
        if self.dim != 2:
            raise ValueError("Distance transform only on 2D images")

        self.data = cv.distanceTransform(self.data, cv.DIST_L2, 3)
        cv.normalize(self.data, self.data, 0, 1, cv.NORM_MINMAX)
        self.data = np.uint8(self.data * 255)
