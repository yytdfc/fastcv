import numpy as np
import cv2


def adasharp(img0):
    img0 = img0.astype(np.float32) / 255.
    out = img0.copy()
    img = img0[:, :, 1].copy()
    kernel = np.ones((3,3), np.float32) / 9
    mean = cv2.filter2D(img, -1, kernel)
    hpass = img - mean + 0.5
    tmp = np.expand_dims(2 * hpass, 2) + img0 - 1.0
    out = (np.clip(tmp, 0, 1) * 0.7) + img0 * 0.3
    return np.clip(out * 255, 0, 255).round().astype(np.uint8)

    # mean = cv2.filter2D(img, -1, kernel)
    # hpass = img - mean + 0.5
    # tmp = np.expand_dims(2 * hpass, 2) + img0 - 1.0
    # out = (np.clip(tmp, 0, 1) * 0.7) + img0 * 0.3


    # tmp = 2 * (img - mean) + img

    # a = 0.7
    # tmp = (3 * img - 2 * mean) * a + (1 - a) * img
    # 2 a (img - mean) + img


def swf(img, filter=None):
    size = 3
    filter = np.ones((size, size), dtype=np.float32) / (size ** 2)
    out = cv2.filter2D(img, -1, filter)
    return out
    def softmax(imgs, d):
        print(imgs.max(), imgs.min())
        exp = np.exp(imgs)
        return exp / np.sum(exp, d)

    def side_filter(img, filter):
        k = filter.shape[0] // 2
        l, r, u, d = [filter.copy() for _ in range(4)]
        u[:k] = 0
        d[-k:] = 0
        l[:, :k] = 0
        r[:, -k:] = 0
        ul, ur, dl, dr = [l.copy(), r.copy(), l.copy(), r.copy()]
        ul[:k] = 0
        dl[-k:] = 0
        ur[:k] = 0
        dr[-k:] = 0
        filters = []
        img_01 = img.astype(np.float32) / 255.
        for f in [u, d, l, r, ul, ur, dl, dr]:
            f /= f.sum()
            f [k, k] -= 1
            filters.append(f)
            # print(f)
            # print(cv2.filter2D(img_01, -1, f).max())
            # input()
        # print(img_01.min(), img_01.max())
        # print(diffs.min(), diffs.max())
        # symbol = diffs >= 0
        # out = np.min(np.abs(diffs), 0) * symbol #  + img
        # print(out.shape)

        # ======================================================================
        # diffs = np.concatenate([np.expand_dims(cv2.filter2D(img_01, -1, f), 0) for f in filters], 0)
        diffs = np.concatenate([np.expand_dims(cv2.filter2D(img_01 * 255, -1, f), 0) for f in filters], 0)
        # print(diffs)
        soft = softmax(1 / (np.abs(diffs) + 1), 0)
        print(soft.shape, soft.max(), soft.min())
        diff = (soft * diffs).sum(0)
        out = (img + diff)#  * 255.
        # ======================================================================
        # diffs = np.concatenate([np.expand_dims(cv2.filter2D(img, -1, f), 0) for f in filters], 0)
        # mins = np.expand_dims(np.argmin(np.abs(diffs), 0), 0)
        # out = np.take_along_axis(diffs, mins, axis=0)[0]
        # out = out + img
        # ======================================================================

        # out = diffs[mins]
        # print(mins.shape)
        return out
    out = np.clip(side_filter(img.astype(np.float32), filter), 0, 255).round().astype(np.uint8)

    return out
    return img



if __name__ == "__main__":
    import os
    im = cv2.imread("/Volumes/chenfu/dataset/deblur/new_train/portrait/hr/versa-02526.png")
    # out = cv2.bilateralFilter(im, 25, 0.3, 7)
    # out = swf(im)
    out = cv2.GaussianBlur(im,(3,3),sigmaX=4)
    # out = im
    import fastcv
    out = fastcv.resize(out, scale=0.5)
    # out = swf(out)
    cv2.imwrite("out.png", out)
    os.system("open out.png")


