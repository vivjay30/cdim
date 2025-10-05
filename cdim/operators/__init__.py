# Code based on https://github.com/DPS2022/diffusion-posterior-sampling
from abc import ABC, abstractmethod

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")

        original_init = cls.__init__

        # Wrap the original __init__ to inject the `name` attribute.
        def new_init(self, *args, **kwargs):
            self.name = name  # Set the name attribute
            original_init(self, *args, **kwargs)  # Call the original __init__

        cls.__init__ = new_init  # Replace the class's __init__ with the wrapped version.
        __OPERATOR__[name] = cls  # Register the class.
        return cls
    return wrapper

def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


# Import everything to make sure they register
from .random_box_masker import RandomBoxMasker
from .random_pixel_masker import RandomPixelMasker
from .identity_operator import IdentityOperator
from .super_resolution_operator import SuperResolutionOperator
from .gaussian_blur_operator import GaussianBlurOperator
from .phase_retrieval_operator import PhaseRetrievalOperator
