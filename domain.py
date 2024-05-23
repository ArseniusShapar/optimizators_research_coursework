from typing import Sequence, Callable

import numpy as np

n: int = 2

Vector = np.ndarray[float, (n, 1)]
Trajectory = np.ndarray[float]
TargetFunction = Callable[[Vector], float]

def vec(*values: float) -> Vector:
    return np.array(values).astype(float).reshape((n, 1))


def diff(func: Callable[[Vector], float], i: int, x: Vector, h: float = 0.001, method='central',
         f0: float = None) -> float:
    dx = np.zeros((n, 1))
    dx[i][0] = h

    if method == 'left':
        f0 = f0 if f0 is not None else func(x - dx)
        return (f0 - func(x - dx)) / h
    elif method == 'central':
        return (func(x + dx) - func(x - dx)) / (2 * h)
    elif method == 'right':
        f0 = f0 if f0 is not None else func(x - dx)
        return (func(x + dx) - f0) / h
    else:
        raise ValueError(f'Invalid method: {method}')


def gradient(func: Callable[[Vector], float], x: Vector, h: float = 0.001, method='central', f0: float = None) -> Vector:
    if method in ('left', 'right'):
        f0 = f0 if f0 is not None else func(x)
        grad = [diff(func, i, x, h, method, f0) for i in range(n)]
    elif method == 'central':
        grad = [diff(func, i, x, h) for i in range(n)]
    else:
        raise ValueError(f'Invalid method: {method}')
    return vec(*grad)
