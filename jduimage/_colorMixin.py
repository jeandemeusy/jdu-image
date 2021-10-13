import cv2 as cv
import numpy as np


class ColorMixin:
    def BGR_to_LAB(self) -> None:
        """Converts the image from BGR to LAB colorspace."""
        self.__colorspace(cv.COLOR_BGR2LAB)

    def LAB_to_BGR(self) -> None:
        """Converts the image from LAB to BGR colorspace."""
        self.__colorspace(cv.COLOR_LAB2BGR)

    def BGR_to_HSV(self) -> None:
        """Converts the image from BGR to HSV colorspace."""
        self.__colorspace(cv.COLOR_BGR2HSV)

    def HSV_to_BGR(self) -> None:
        """Converts the image from HSV to BGR colorspace."""
        self.__colorspace(cv.COLOR_HSV2BGR)

    def BGR_to_GRAY(self) -> None:
        """Converts the image from BGR to GRAY colorspace."""
        self.__colorspace(cv.COLOR_BGR2GRAY)

    def to_BGR(self) -> None:
        """Assures that img is a three channels image.

        Parameters
        ----------
        img: Image
            Input to convert to three channels image
        """
        if self.dim == 2:
            self.data = cv.cvtColor(self.data, cv.COLOR_GRAY2BGR)

    def to_GRAY(self) -> None:
        """Assures that img is a two channels image.

        Parameters
        ----------
        img: Image
            Input to convert to three channels image
        """
        if self.dim == 3:
            self.BGR_to_GRAY()

    def to_8UC1(self) -> None:
        """Convert the image to 8UC1."""
        if isinstance(self.data[0, 0], np.uint8):
            return

        max_val = np.max(self.data)
        min_val = np.min(self.data)

        temp = (self.data - min_val) / (max_val - min_val) * 255.0

        self.data = temp.astype(np.uint8)

    def to_16UC1(self) -> None:
        """Convert the image to 16UC1."""
        if isinstance(self.data[0, 0], np.uint8):
            return

        max_val = np.max(self.data)
        min_val = np.min(self.data)

        temp = (self.data - min_val) / (max_val - min_val) * 65535.0

        self.data = temp.astype(np.uint16)

    def kmeans(
        self, K: int, maxiter: int = 10, epsilon: float = 1.0, equalize: bool = False
    ) -> None:
        """Computes k-means on an image.

        Parameters
        ----------
        K: int
            Number of clusters to create
        maxiter: int, optional
            Maximum iterations for kmeans (default is 10)
        epsilon: float, optional
            Correctly assigned pixel rate (default is 1.0)
        equalize: bool, optional
            Flag to indicate if an histogram equalization needs to be applied (default is False)
        """
        if self.dim == 2:
            self.to_BGR()
        Z = self.data.reshape((-1, 3))

        Z = np.float32(Z)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, maxiter, epsilon)
        _, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)

        if equalize:
            self.data = label.flatten().reshape(self.shape[:2]).astype(np.uint8)
        else:
            self.data = center[label.flatten()].reshape(self.shape).astype(np.uint8)

    def __colorspace(self, code: int) -> None:
        """Converts the image to a given colorspace.

        Parameters
        ----------
        code: int
            OpenCV color conversion code.
        """
        if self.dim != 3:
            raise ValueError("Colorspace conversion requires 3 channels.")
        if self.shape[2] != 3:
            raise ValueError(f"Excepting 3 channels, found {self.shape[2]}")

        self.data = cv.cvtColor(self.data, code)
