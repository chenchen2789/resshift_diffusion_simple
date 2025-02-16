
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import random
import numpy as np
# from torchvision.datasets import CIFAR10
# from torchvision import transforms

class DataloaderSimpleTest(Dataset):
    """
    输出维度[b, c, h, w]，颜色 rgb， 图像值域 [-1, 1]
    """
    def __init__(self, opt):
        self.paths = opt['paths']
        self.data_paths = []
        self.sf = opt['sf']
        self.ds = 1.0/(self.sf)
        self.gt_size = opt['gt_size']
        for path in self.paths:
            self.data_paths.extend(glob.glob(path + '/*.png') + glob.glob(path + '/*.jpg') + glob.glob(path + '/*.JPEG'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.data_paths[index])
        img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
        w,h,_ = img.shape
        i, j  = random.randint(0, w - 1 - self.gt_size), random.randint(0, h - 1 - self.gt_size)
        gt = img[i:i+self.gt_size, j:j+self.gt_size, ::-1]/255.0

        lq = cv2.resize(gt, dsize=None, fx=self.ds, fy=self.ds)


        gt = np.transpose(gt, (2, 0, 1))
        lq = np.transpose(lq, (2, 0, 1))

        gt = torch.from_numpy(gt.copy())
        lq = torch.from_numpy(lq.copy())

        gt = (gt-0.5)/0.5
        lq = (lq-0.5)/0.5

        gt = gt.float()
        lq = lq.float()
        return {'gt': gt, 'lq': lq}



