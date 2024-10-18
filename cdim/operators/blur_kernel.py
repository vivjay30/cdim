import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn


class BlurKernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.padding = self.kernel_size // 2

        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k).float()
            k = k.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kernel_size, kernel_size)
            k = k.repeat(3, 1, 1, 1)  # Shape (3, 1, kernel_size, kernel_size)
            self.register_buffer('kernel', k)
        else:
            raise ValueError(f"Unknown blur type {self.blur_type}")

    def forward(self, x):
        x = F.pad(x, [self.padding]*4, mode='reflect')
        x = F.conv2d(x, self.kernel, groups=3)
        return x
