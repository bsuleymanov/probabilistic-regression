import hydra
from hydra.utils import instantiate, call, get_method

from types import FunctionType
from typing import Callable, Mapping


def get_callable_object_name(obj):
    if isinstance(obj, FunctionType):
        return obj.__name__
    elif isinstance(obj, Callable):
        return obj.__repr__()[:-2]
    else:
        raise TypeError(f"{type(obj)} is not callable.")


def instantiate_from_config(config):
    if isinstance(config, Mapping) and "_target_" in config.keys():
        return instantiate(config)
    elif isinstance(config, str):
        return get_method(config)
    else:
        raise TypeError(f"{type(object)} can't be used as a config type.")