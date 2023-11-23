import cv2
import numpy as np
from PIL import Image
import torch

# type: tensor, numpy / np, pil, cv2
# dtype: auto, fp32, fp16
# from_range: (0, 255), (-1.0, 1.0), (0.0, 1.0)
# to_range: (0, 255), (-1.0, 1.0), (0.0, 1.0)
# device: cuda, cpu
# color: bgr, rgb, gray
# layout: nchw, nhwc, hwc, chw, hw


def _cast_tensor(x, dtype="fp32", from_range="auto", to_range="auto", device="cuda", color="rgb", layout="nchw"):
    if isinstance(x, torch.Tensor):
        if dtype == "auto":
            if x.dtype == torch.float16:
                dtype = "fp16"
            else:
                dtype = "fp32"
        if to_range == "auto":
            if x.dtype == torch.float16:
                to_range = [-1.0, 1.0]
            else:
                to_range = [0.0, 1.0]
        if color == "rgb":
            if layout == "nchw":
                x = x[:, [2, 1, 0], :, :]
            elif layout == "nhwc":
                x = x[:, :, :, [2, 1, 0]]
            elif layout == "chw":
                x = x[:, [2, 1, 0], :]
            elif layout == "hwc":
                x = x[:, :, [2, 1, 0]]
            else:
                raise ValueError("layout must be nchw, nhwc, chw or hwc")
        elif color == "bgr":
            if layout == "nchw":
                x = x[:, [0, 1, 2], :, :]
            elif layout == "nhwc":
                x = x[:, :, :, [0, 1, 2]]
            elif layout == "chw":
                x = x[:, [0, 1, 2], :]
            elif layout == "hwc":
                x = x[:, :, [0, 1, 2]]
            else:
                raise ValueError("layout must be nchw, nhwc, chw or hwc")
        elif color == "gray":
            if layout == "nchw":
                x = x[:, [0], :, :]
            elif layout == "nhwc":
                x = x[:, :, :, [0]]
            elif layout == "chw":
                x = x[:, [0], :]
            elif layout == "hwc":
                x = x[:, :, [0]]
            else:
                raise ValueError("layout must be nchw, nhwc, chw or hwc")
        else:
            raise ValueError("color must be rgb, bgr or gray")
        if dtype == "fp16":
            x = x.float()
            x = x.clamp(*to_range)
            x = x.add(1.0).div(2.0)
            x = x.mul(65535.0)
            x = x.to(torch.uint16)
        return x.to(device)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        return _cast_tensor(x, dtype, to_range, device, color, layout)
    elif isinstance(x, Image.Image):
        x = torch.from_numpy(np.array(x))
        return _cast_tensor(x, dtype, to_range, device, color, layout)
    elif isinstance(x, cv2.UMat):
        x = torch.from_numpy(np.array(x))
        return _cast_tensor(x, dtype, to_range, device, color, layout)
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")


def _cast_numpy(x, dtype="fp32", to_range="auto", device="cuda", color="rgb", layout="nchw"):
    if isinstance(x, np.ndarray):
        if dtype == "auto":
            if x.dtype == np.float16:
                dtype = "fp16"
            else:
                dtype = "fp32"
        if to_range == "auto":
            if x.dtype == np.float16:
                to_range = [-1.0, 1.0]
            else:
                to_range = [0.0, 1.0]
        if color == "rgb":
            if layout == "nchw":
                x = x[:, [2, 1, 0], :, :]
            elif layout == "nhwc":
                x = x[:, :, :, [2, 1, 0]]
            elif layout == "chw":
                x = x[:, [2, 1, 0], :]
            elif layout == "hwc":
                x = x[:, :, [2, 1, 0]]
            else:
                raise ValueError("layout must be nchw, nhwc, chw or hwc")
        elif color == "bgr":
            if layout == "nchw":
                x = x[:, [0, 1, 2], :, :]
            elif layout == "nhwc":
                x = x[:, :, :, [0, 1, 2]]
            elif layout == "chw":
                x = x[:, [0, 1, 2], :]
            elif layout == "hwc":
                x = x[:, :, [0, 1, 2]]
            else:
                raise ValueError("layout must be nchw, nhwc, chw or hwc")
        elif color == "gray":
            if layout == "nchw":
                x = x[:, [0], :, :]
            elif layout == "nhwc":
                x = x[:, :, :, [0]]
            elif layout == "chw":
                x = x[:, [0], :]
            elif layout == "hwc":
                x = x[:, :, [0]]
            else:
                raise ValueError("layout must be nchw, nhwc, chw or hwc")
        else:
            raise ValueError("color must be rgb, bgr or gray")
        if dtype == "fp16":
            x = x.astype(np.float32)
            x = np.clip(x, *to_range)
            x = x + 1.0
            x = x / 2.0
            x = x * 65535.0
            x = x.astype(np.uint16)
        return x
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    elif isinstance(x, Image.Image):
        x = np.array(x)
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    elif isinstance(x, cv2.UMat):
        x = np.array(x)
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    
def _cast_pil(x, dtype="fp32", to_range="auto", device="cuda", color="rgb", layout="nchw"):
    if isinstance(x, Image.Image):
        return x
    elif isinstance(x, np.ndarray):
        x = Image.fromarray(x)
        return _cast_pil(x, dtype, to_range, device, color, layout)
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return _cast_pil(x, dtype, to_range, device, color, layout)
    elif isinstance(x, cv2.UMat):
        x = np.array(x)
        return _cast_pil(x, dtype, to_range, device, color, layout)
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")


def cast(x, type="tensor", dtype="fp32", to_range="auto", device="cuda", color="rgb", layout="nchw"):
    if type == "tensor":
        return _cast_tensor(x, dtype, to_range, device, color, layout)
    elif type == "numpy":
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    elif type == "pil":
        return _cast_pil(x, dtype, to_range, device, color, layout)
    elif type == "cv2":
        return _cast_cv2(x, dtype, to_range, device, color, layout)
    else:
        raise ValueError("type must be tensor, numpy, pil or cv2")