import torch
from cdim.operators import register_operator
from cdim.operators.blur_kernel import BlurKernel

@register_operator(name='gaussian_blur')
class GaussianBlurOperator:
    def __init__(self, kernel_size, intensity, device='cuda'):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = BlurKernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)

    def __call__(self, data, **kwargs):
        return self.conv(data)
