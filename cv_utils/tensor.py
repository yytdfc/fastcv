import cv2
import numpy as np


def to_tensor(img):
    img = np.array(img).transpose(2, 0, 1).reshape(1, 3, nh, nw)
    img = img.astype(np.float32) / 255.0
    return img


def from_tensor(img):
    pass
