from functools import partial

import cv2
import torch

from .resize import resize
from .converter import to_tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, resize_args):
        super().__init__()
        if isinstance(imgs[0], str):
            pass

        self.imgs = imgs
        self.resize_args = resize_args

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        im = cv2.imread(img)[:,:,::-1]
        im = resize(im, **self.resize_args)
        return to_tensor(im.copy()).squeeze(0)


def dataloader(imgs, resize_args=None, **kargs):
    if resize_args is None:
        resize_args = {}

    dataset = Dataset(imgs, resize_args)
    xkargs = {
        "batch_size": 1,
        "num_workers": 1,
        "drop_last": False,
        "prefetch_factor": 2,
        "pin_memory": False,
        "shuffle": False
    }
    xkargs.update(kargs)
    dataloader = torch.utils.data.DataLoader(dataset, **xkargs)
    return dataloader

