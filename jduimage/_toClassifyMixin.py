import cv2 as cv
import numpy as np


class ToClassifyMixin:
    def integral(self) -> None:
        """Computes the integral of the image."""
        if self.dim == 3:
            self.BGR_to_GRAY()

        self.data = self.data.astype(np.float32) / 255.0

        self.data = cv.integral(self.data)

    def vertical_shift(self, offset: int) -> None:
        """Shift the image verticaly by a given offset."""
        new_indexes = (np.arange(self.width) + offset) % self.width

        if self.dim == 3:
            self.data = self.data[:, new_indexes, :]
        elif self.dim == 2:
            self.data = self.data[:, new_indexes]

    def horizontal_shift(self, offset: int) -> None:
        """Shift the image horizontaly by a given offset."""
        new_indexes = (np.arange(self.height) + offset) % self.height

        if self.dim == 3:
            self.data = self.data[new_indexes, :, :]
        elif self.dim == 2:
            self.data = self.data[new_indexes, :]
