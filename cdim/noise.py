# Code based on https://github.com/DPS2022/diffusion-posterior-sampling
from abc import ABC, abstractmethod
import torch

__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def __call__(self, data):
        pass

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        # Important! We scale sigma by 2 because the config assumes images are in [0, 1]
        # but actually this model uses images in [-1, 1]
        self.sigma = 2 * sigma
        self.name = 'gaussian'
    
    def __call__(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate
        self.name = 'poisson'

    def __call__(self, data):
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)


@register_noise(name='bimodal')
class BimodalNoise(Noise):
    def __init__(self, value):
        self.value = value
        self.name = 'bimodal'

    def __call__(self, data):
        noise = self.sample_noise_distribution(data)
        return data + noise.to(data.device)

    def sample_noise_distribution(self, data):
        return (torch.randint(low=0, high=2, size=data.shape) * 2 - 1) * self.value

