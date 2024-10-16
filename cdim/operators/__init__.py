# Code based on https://github.com/DPS2022/diffusion-posterior-sampling
from abc import ABC, abstractmethod

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


# Import everything to make sure they register
from .random_box_masker import RandomBoxMasker
from .identity_operator import IdentityOperator
