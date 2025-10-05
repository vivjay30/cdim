import torch
import torch.nn.functional as F
from cdim.operators import register_operator
from cdim.fastmri import fft2c_new, ifft2c_new

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])

def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator:
    def __init__(self, oversample=2.0, device='cuda'):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def __call__(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude