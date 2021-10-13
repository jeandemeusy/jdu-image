import cv2 as cv
import numpy as np


class MorphologicalMixin:
    def sharpen(self) -> None:
        """Sharpens the image with 3x3 filter. Only works on 2D images."""
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.data = cv.filter2D(self.data, -1, filter)

    def open(self, size: int = (5, 5), element: int = cv.MORPH_ELLIPSE) -> None:
        """Performs morphological opening on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Size of the kernel (default is (5, 5))
        element: int, optional
            Structural element (default is cv.MORPH_ELLIPSE)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        kernel = cv.getStructuringElement(element, size)
        self.data = cv.morphologyEx(self.data, cv.MORPH_OPEN, kernel)

    def close(self, size: int = (5, 5), element: int = cv.MORPH_ELLIPSE) -> None:
        """Performs morphological closing on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Size of the kernel (default is (5, 5)
            )
        element: int, optional
            Structural element (default is cv.MORPH_ELLIPSE)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        kernel = cv.getStructuringElement(element, size)
        self.data = cv.morphologyEx(self.data, cv.MORPH_CLOSE, kernel)

    def dilate(self, size: int = (5, 5), element: int = cv.MORPH_ELLIPSE) -> None:
        """Performs morphological dilatation on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Size of the kernel (default is (5, 5))
        element: int, optional
            Structural element (default is cv.MORPH_ELLIPSE)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        kernel = cv.getStructuringElement(element, size)
        self.data = cv.morphologyEx(self.data, cv.MORPH_DILATE, kernel)

    def erode(self, size: int = (5, 5), element: int = cv.MORPH_ELLIPSE) -> None:
        """Performs morphological erosion on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Size of the kernel (default is (5,5))
        element: int, optional
            Structural element (default is cv.MORPH_ELLIPSE)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        kernel = cv.getStructuringElement(element, size)
        self.data = cv.morphologyEx(self.data, cv.MORPH_ERODE, kernel)

    def tophat(self, size: int = 5, element: int = cv.MORPH_ELLIPSE) -> None:
        """Performs morphological tophat on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Size of the kernel (default is 5)
        element: int, optional
            Structural element (default is cv.MORPH_ELLIPSE)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        kernel = cv.getStructuringElement(element, (size, size))
        self.data = cv.morphologyEx(self.data, cv.MORPH_TOPHAT, kernel)

    def algebric_open(self, size: int = 5, step: int = 5) -> None:
        """Performs morphological algebric opening on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Structural element size
        step: int, optional
            Angle step
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        result = np.zeros(self.shape, dtype=np.uint8)

        for a in range(0, 180, step):
            kernel = line_strel(size=size, angle=a)

            temp = cv.morphologyEx(self.data, cv.MORPH_OPEN, kernel).astype(np.uint8)
            result = np.maximum(result, temp)

        self.data = result

    def algebric_dilate(self, size: int = 5, step: int = 5) -> None:
        """Performs morphological algebric dilatation on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Structural element size
        step: int, optional
            Angle step
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        result = np.zeros(self.shape, dtype=np.uint8)

        for a in range(0, 180, step):
            kernel = line_strel(size=size, angle=a)

            temp = cv.morphologyEx(self.data, cv.MORPH_DILATE, kernel).astype(np.uint8)
            result = np.maximum(result, temp)

        self.data = result

    def blackhat(self, size: int = 5, element: int = cv.MORPH_ELLIPSE) -> None:
        """Performs morphological blackhat on the image. Only works on 2D images.

        Parameters
        ----------
        size: int, optional
            Size of the kernel (default is 5)
        element: int, optional
            Structural element (default is cv.MORPH_ELLIPSE)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        kernel = cv.getStructuringElement(element, (size, size))
        self.data = cv.morphologyEx(self.data, cv.MORPH_BLACKHAT, kernel)

    def gabor(self) -> None:
        """Applies gabor filter to the image."""
        ksize = 21
        thetas = [-45]

        filters = []
        for a in thetas:
            kernel = cv.getGaborKernel([ksize, ksize], 40, a, 25, 1)
            filters.append(kernel)

        result = np.zeros(self.shape, dtype=np.uint8)
        for kernel in filters:
            imgfiltered = cv.filter2D(self.data, -1, kernel)
            result = np.maximum(result, imgfiltered)

        self.data = result

    def edges(self, thres1: int = 100, thres2: int = 200) -> None:
        """Finds the edges on the image with Canny algorithm.

        Parameters
        ----------
        thres1: int, optional
            Low threshold (default is 100)
        thres2: int, optional
            High threshold (default is 200)
        """
        self.data = cv.Canny(image=self.data, threshold1=thres1, threshold2=thres2)

    def sobel(self) -> None:
        self.data = cv.Sobel(src=self.data, ddepth=cv.CV_8UC1, dx=1, dy=1, ksize=3)


def line_strel(size: int, angle: float) -> np.ndarray:
    """Creates a linear structural element, with given length and rotation angle.

    Parameters
    ----------
    size: int
        Length of the structural element when laying flat
    angle: float
        Rotation angle of the line in degrees

    Returns
    -------
    np.ndarray
        Square array containing the linear structural element
    """
    if size % 2 != 1:
        raise ValueError("Size must be odd")

    line = np.zeros((size, size))
    line[line.height // 2, :] = 1

    center = (size // 2, size // 2)

    tform = cv.getRotationMatrix2D(center, angle, 1)
    kernel = cv.warpAffine(line, tform, line.shape)

    return (kernel * 255).astype(np.uint8)
