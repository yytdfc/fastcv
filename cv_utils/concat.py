import math
import numpy as np
from PIL import Image
import cv2

from .draw_text import draw_text

def concat(imgs, direc="auto", mode=cv2.INTER_AREA, crop=None, labels=None, reshape=True):
    if len(imgs) == 0:
        return imgs
    if isinstance(imgs[0], np.ndarray):
        type = "np"
    elif isinstance(imgs[0], Image.Image):
        type = "pil"
        imgs = [np.array(i) for i in imgs]

    h, w, _ = imgs[0].shape
    if reshape:
        reshaped_imgs = [
            im if im.shape[0] == h and im.shape[1] == w else cv2.resize(im, (w, h), interpolation=mode)
            for im in imgs
        ]
    else:
        reshaped_imgs = imgs
    if crop is not None:
        x, y, w, h = crop
        crop_imgs = [i[y:y+h, x:x+w].copy() for i in reshaped_imgs]
        reshaped_imgs = [cv2.rectangle(i, (x, y), (x + w, y + h), (0, 0, 255), 1) for i in reshaped_imgs]
        if labels is not None:
            crop_imgs = [draw_text(i, l, color=(0, 0, 255)) for i, l in zip(crop_imgs, labels)]
    if labels is not None:
        reshaped_imgs = [draw_text(i, l, color=(0, 0, 255)) for i, l in zip(reshaped_imgs, labels)]

    if (direc == "auto" and (h >= w)) or direc == "x":
        concat_img = np.concatenate(reshaped_imgs, 1)
    else:
        concat_img = np.concatenate(reshaped_imgs, 0)
    if crop is not None:
        crop_concat_img = np.concatenate(crop_imgs, 1)
        return concat_img, crop_concat_img
    if type == "pil":
        concat_img = Image.fromarray(concat_img)
    return concat_img

def get_fine_product(n):
    for i in range(int(n ** 0.5) + 1, 1, -1):
        if n % i == 0:
            j = n // i
            if 0.4999 <= j / i <= 2.0001:
                return max(j, i), min(j, i)
            break
    j = math.ceil(math.sqrt(n))
    i = math.ceil(n / j)
    return j, i

def grid_concat(images, nx=0, ny=0, board=None, reshape=True):
    if nx == 0 and ny == 0:
        nx, ny = get_fine_product(len(images))
    elif nx == 0:
        nx = math.ceil(len(images) / ny)
    elif ny == 0:
        ny = math.ceil(len(images) / nx)

    if len(images) < nx * ny:
        images.extend([np.zeros_like(images[0]) + 127] * (nx * ny - len(images)))
    rows = [
        concat(images[i * nx : (i + 1) * nx], direc="x", reshape=reshape)
        for i in range(ny)
    ]
    im = concat(rows, direc="y", reshape=reshape)
    if board is not None:
        h, w, _ = im.shape
        im = cv2.rectangle(im, (0, 0), (w, h), (255, 0, 0), board)
    return im
