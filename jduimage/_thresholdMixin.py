import cv2 as cv


class ThresholdMixin:
    def adaptive_threshold(
        self, blursize: int = 5, blocksize: int = 21, c: int = 0
    ) -> None:
        """Threshold image with respect to local background (21pxl square), by first blurring the image. Only works on 2D images.

        Parameters
        ----------
        blursize: int, optional
            Size of blur filter (default is 5)
        blocksize: int, optional
            Mask size where threshold is computed (default is 21)
        c: int, optional
            Obscure parameter (default is 0)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        if blursize > 0:
            self.blur(size=blursize, method="gauss")

        self.data = cv.adaptiveThreshold(
            self.data,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY_INV,
            blocksize,
            c,
        )

    def binarize(self, threshold: int = -1) -> None:
        """Binarizes the image with Otsu's method. Only works on 2D images.

        Parameters
        ----------
        threshold: int, optional
            Binarisation threshold. If set to -1, otsu binarisation is applied (default is -1)
        """
        if self.dim != 2:
            raise ValueError("Only on 2D images")

        if threshold == -1:
            _, self.data = cv.threshold(
                self.data, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )
        else:
            _, self.data = cv.threshold(self.data, threshold, 255, cv.THRESH_BINARY)
