from PIL import Image
import numpy as np
import torch
from .resize import resize


def to_tensor(im):
    return torch.tensor(im.transpose(2, 0, 1)).unsqueeze(0).float() / 255.


def from_tensor(t):
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
    t = t.squeeze(0).transpose(1, 2, 0)
    return np.clip((t * 255.0).round(), 0, 255).astype(np.uint8)


def convert_to_onnx(net, x, path, input_names, output_names, **kargs):
    torch.onnx.export(
        net,
        x,
        path,
        input_names=input_names,
        output_names=output_names,
        verbose=False,
        opset_version=11,
        **kargs,
    )
    print("Succeed convert to", path)


def softmax2d(t):
    t = torch.exp(t)
    t = t / torch.sum(t, 1)
    return t


def get_image(img, long=0, short=0, factor=16):
    img = Image.open(img).convert("RGB")
    img = np.array(img)
    img = resize(img, long=long, short=short, factor=factor)
    # img = to_tensor(img)
    return img


def get_mask(mask, h, w):
    mask = Image.open(mask).convert("P")
    mask = np.array(mask)
    mask = Image.fromarray(mask)
    mask = mask.point(lambda x: 255 if x != 0 else 0)
    mask = mask.resize((w, h), Image.BILINEAR)
    mask = np.array(mask).reshape(1, 1, h, w)
    mask = mask.astype(np.float32) / 255.0
    return mask


def get_multimask(mask, h, w):
    mask = Image.open(mask).convert("P")
    mask = np.array(mask)
    mask = Image.fromarray(mask)
    # mask = mask.point(lambda x: 255 if x != 0 else 0)
    mask = mask.resize((w, h), Image.NEAREST)
    mask = np.array(mask).reshape(1, 1, h, w)
    # mask = mask.astype(np.float32) / 255.0
    return mask


def iou(self, mask, alpha):
    assert mask.shape == alpha.shape
    a_b = (alpha > 0.5).astype(np.int64)
    m_b = (mask > 0.5).astype(np.int64)
    sum = a_b + m_b
    inter = (sum == 2).astype(np.float32).sum()
    union = (sum > 0).astype(np.float32).sum()
    return inter / union
