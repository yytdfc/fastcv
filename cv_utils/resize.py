import cv2


def resize(img, size=(0, 0), short=0, long=0, scale=0, factor=1, mode=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    mi, ma = min(h, w), max(h, w)
    if short:
        h = round(short / mi * h / factor) * factor
        w = round(short / mi * w / factor) * factor
    elif long:
        h = round(long / ma * h / factor) * factor
        w = round(long / ma * w / factor) * factor
    elif scale:
        h = round(scale * h / factor) * factor
        w = round(scale * w / factor) * factor
    else:
        sh, sw = size
        h = round(sh / factor) * factor
        w = round(sw / factor) * factor

    return cv2.resize(img, (w, h), mode)
