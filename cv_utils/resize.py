import cv2
import numpy as np
from PIL import Image


_PIL_INTERPOLATION_METHODS = {
    'nearest': Image.NEAREST,
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS,
    'linear': Image.LINEAR,
    'box': Image.BOX,
    'hamming': Image.HAMMING,
    'cubic': Image.CUBIC,
}

_CV2_INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'linear': cv2.INTER_LINEAR,
    'box': cv2.INTER_AREA,
    'hamming': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
}


def resize(img, size=(0, 0), short=0, long=0, scale=0, factor=1, mode="linear", pillow=False):
    epilson = 1e-3
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
        mode = _CV2_INTERPOLATION_METHODS[mode]
    elif isinstance(img, Image.Image):
        w, h = img.size
        mode = _PIL_INTERPOLATION_METHODS[mode]
    else:
        raise TypeError("img must be np.ndarray or PIL.Image.Image")
    mi, ma = min(h, w), max(h, w)
    if short:
        h = round(short / mi * h / factor - epilson) * factor
        w = round(short / mi * w / factor - epilson) * factor
    elif long:
        h = round(long / ma * h / factor - epilson) * factor
        w = round(long / ma * w / factor - epilson) * factor
    elif scale:
        h = round(scale * h / factor - epilson) * factor
        w = round(scale * w / factor - epilson) * factor
    elif size != (0, 0):
        sh, sw = size
        h = round(sh / factor - epilson) * factor
        w = round(sw / factor - epilson) * factor
    else:
        h = round(h / factor - epilson) * factor
        w = round(w / factor - epilson) * factor

    if isinstance(img, np.ndarray):
        return cv2.resize(img, (w, h), interpolation=mode)
    else:
        return img.resize((w, h), resample=mode)
