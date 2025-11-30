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
        # For compatibility with Gaussian code paths that check noise_function.sigma
        # This is not used for Poisson - the actual variance is signal-dependent
        self.sigma = None  # Signal that this is heteroscedastic noise

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
    
    def get_counts(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert observed measurement y (in [-1, 1] range) to count space.
        
        For Poisson noise, y was generated as:
            y = poisson(x * 255 * rate) / (255 * rate)
        So the expected count is:
            count = y * 255 * rate
        """
        # Convert from [-1, 1] to [0, 1]
        y_01 = (y + 1.0) / 2.0
        # Convert to count space
        counts = y_01 * 255.0 * self.rate
        return counts
    
    def get_weights(self, y: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """
        Compute Pearson weights W = 1 / Var(y) for each observation.
        
        For Poisson noise, Var(count) = count, so in the normalized space:
            Var(y) = count / (255 * rate)^2
        
        The weight is the inverse variance. We add eps to avoid division by zero
        for very dark pixels.
        
        Returns weights in the measurement space (normalized).
        """
        counts = self.get_counts(y)
        # Variance in count space is the count itself
        # Variance in normalized space is count / (255 * rate)^2
        # Weight = 1 / Var = (255 * rate)^2 / count
        scale = 255.0 * self.rate
        weights = (scale ** 2) / (counts + eps)
        return weights
    
    def sum_counts(self, y: torch.Tensor, operator=None) -> float:
        """
        Compute sum of counts (used in mean formula).
        For Poisson, this is sum(y_counts) = sum((y+1)/2 * 255 * rate).
        
        If operator has 'select' method, only compute over observed pixels.
        """
        if operator is not None and hasattr(operator, 'select'):
            y_selected = operator.select(y).flatten()
        else:
            y_selected = y.flatten()
        counts = self.get_counts(y_selected)
        return counts.sum().item()


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

