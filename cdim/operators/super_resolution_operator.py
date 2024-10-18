import torch

from cdim.operators import register_operator
from cdim.operators.resizer import Resizer


@register_operator(name='super_resolution')
class SuperResolutionOperator:
    def __init__(self, scale=4, in_shape=(1, 3, 256, 256), device='cpu'):
        self.device = device
        self.down_sample = Resizer(in_shape, 1/scale).to(device)

    def __call__(self, data, **kwargs):
        return self.down_sample(data)

