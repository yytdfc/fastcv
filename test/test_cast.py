from fastcv import cast
import numpy as np
from PIL import Image

if __name__ == "__main__":
    x_pil = Image.open("test.jpg")
    # cast from tensor to numpy
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    x = cast(x, type="tensor", dtype="fp32", min_max="auto", device="cpu")
    print(x.dtype, x.shape, x.min(), x.max())
    x = cast(x, type="numpy", dtype="fp32", min_max="auto")
    print(x.dtype, x.shape, x.min(), x.max())

    exit()
    # cast from tensor to pil
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x = cast(x, type="tensor", dtype="fp32", min_max="auto", device="cuda")
    x = cast(x, type="pil", dtype="fp32", min_max="auto")
    print(x.mode, x.size, np.array(x).min(), np.array(x).max())

    # cast from tensor to cv2
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x = cast(x, type="tensor", dtype="fp32", min_max="auto", device="cuda")
    x = cast(x, type="cv2", dtype="fp32", min_max="auto")
    print(x.dtype, x.shape, x.min(), x.max())

    # cast from numpy to tensor
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x = cast(x, type="numpy", dtype="fp32", min_max="auto")
    x = cast(x, type="tensor", dtype="fp32", min_max="auto", device="cuda")
    print(x.dtype, x.shape, x.min(), x.max())

    # cast from numpy to pil
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x = cast(x, type="numpy", dtype="fp32", min_max="auto")
    x = cast(x, type="pil", dtype="fp32", min_max="auto")
    print(x.mode, x.size, np.array(x).min(), np.array(x).max())

    # cast from numpy to cv2
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x = cast(x, type="numpy", dtype="fp32", min_max="auto")
    x = cast(x, type="cv2", dtype="fp32", min_max="auto")
    print(x.dtype, x.shape, x.min(), x.max())

