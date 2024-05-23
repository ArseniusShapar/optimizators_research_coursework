from functools import wraps
from typing import Callable, Any, Sequence

from domain import Vector


def add_information_and_counter(name: str, formula: Callable[[Any], Any], ops: Sequence[Vector] = None,
                                min_value: float = None,
                                limits: list[Vector, Vector] = None):
    def decorator(func: Callable[[Any], Any]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.count += 1
            return func(*args, **kwargs)

        wrapper.count = 0
        wrapper.name = name
        wrapper.formula = formula
        wrapper.ops = ops
        wrapper.min_value = min_value
        wrapper.limits = limits

        return wrapper

    return decorator
