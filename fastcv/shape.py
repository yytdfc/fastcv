import numpy as np
from PIL import Image
import cv2
import torch

def shape(x):
    if isinstance(x, torch.Tensor):
        return x.shape
    elif isinstance(x, np.ndarray):
        return x.shape
    elif isinstance(x, Image.Image):
        w, h = x.size
        return (h, w)
    elif isinstance(x, cv2.UMat):
        return x.shape
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")