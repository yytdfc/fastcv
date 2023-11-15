import sh
import cv2
import numpy as np
from PIL import Image

def view(im):
    if isinstance(im, np.ndarray):
        im = im[:, :, ::-1]
        raw = cv2.imencode(".jpg", im)[1].tobytes()
    elif isinstance(im, Image.Image):
        im = np.array(im)
        im = im[:, :, ::-1]
        raw = cv2.imencode(".jpg", im)[1].tobytes()
    elif isinstance(im, bytes):
        raw = im

    print(sh.imgcat(_in=raw), end="")
