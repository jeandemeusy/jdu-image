import cv2 as cv
from pathlib import Path


class OutputMixin:
    def show(self, label: str = "", height: int = 0, width: int = 0) -> None:
        """Displays the image to the screen.

        Parameters
        ----------
        label: str, optional
            Title of the window
        height: int, optional
            Height of the image for display (default is 0)
        width: int, optional
            Width of the image for display (default is 0)
        """

        display = self.deepcopy()

        if height != 0:
            display.resize("height", height)
        elif width != 0:
            display.resize("width", width)

        cv.imshow(label, display.data)
        cv.waitKey(0)

    def save(self, path: str) -> None:
        """
        Saves the image to a file. All folder creation is handled by the method.

        Parameters
        ----------
        path: str
            Path to output image. Can be absolute or relative. Recognised file types are {"jpg","jpeg","png"}
        """
        if path.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            raise ValueError("Unrecognised image file type")

        folder = "."
        substrings = path.replace("\\", "/").split("/")
        folder = "/".join(substrings[0:-1])

        if folder and not Path(folder).exists():
            Path(folder).mkdir(parents=True, exist_ok=True)

        cv.imwrite(path, self.data)
