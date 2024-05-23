from typing import Callable, Any

import numpy as np
from numpy.linalg import norm

from domain import Vector, Trajectory, TargetFunction
from domain import gradient


def SGD_decreasing_learning_rate(func: TargetFunction, x0: Vector, lr: float = 0.01,
                                 e: float = 0.01) -> tuple[Vector, Trajectory]:
    x = x0.copy()
    f2 = func(x)
    trajectory = [x.copy()]

    while True:
        grad = gradient(func, x, f0=f2)

        if norm(grad) <= e:
            break

        v = lr * grad
        x = x - v
        trajectory.append(x.copy())
        f1, f2 = f2, func(x)

        if f2 > f1:
            lr /= 2

    return x, np.asarray(trajectory)


def SGD_base(func: TargetFunction,
             x0: Vector,
             e: float = 0.01,
             optimizer: Callable[[Any], Vector] = None,
             *optimizer_arguments) -> tuple[Vector, Trajectory]:
    x = x0.copy()
    trajectory = [x.copy()]
    grad = np.fabs(x0) + 100

    i = 0
    while True:
        i += 1
        grad = gradient(func, x)

        if (norm(grad) < e) or (i >= 10_000):
            break

        dx = optimizer(grad, *optimizer_arguments)
        x = x - dx
        trajectory.append(x.copy())

    return x, np.asarray(trajectory)


def SGD(func: TargetFunction,
        x0: Vector,
        e: float = 0.01,
        lr: float = 0.01) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr):
        dx = lr * grad
        return dx

    return SGD_base(func, x0, e, _optimizer, lr)


def Momentum(func: TargetFunction,
             x0: Vector,
             e: float = 0.01,
             lr: float = 0.01,
             gamma: float = 0.9) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, gamma):
        v = _optimizer.v

        v = gamma * v + (1 - gamma) * lr * grad
        dx = v

        _optimizer.v = v
        return dx

    _optimizer.v = 0

    return SGD_base(func, x0, e, _optimizer, lr, gamma)


def Nesterov(func: TargetFunction, x0: Vector, e: float = 0.01, lr: float = 0.1, gamma: float = 0.9) -> tuple[
    Vector, Trajectory]:
    x = x0.copy()
    v = 0
    trajectory = [x.copy()]
    grad = gradient(func, x0)

    i = 0
    while (norm(grad) > e) and (i < 9_999):
        i += 1
        grad = gradient(func, x - gamma * v)
        v = gamma * v + (1 - gamma) * lr * grad
        x = x - v
        trajectory.append(x.copy())

    return x, np.asarray(trajectory)


def Adagrad(func: TargetFunction,
            x0: Vector,
            e: float = 0.01,
            lr: float = 0.01,
            eps: float = 0.01) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, eps):
        G = _optimizer.G

        G += grad ** 2
        v = lr / np.sqrt(G) * grad
        dx = v

        _optimizer.G = G
        return dx

    _optimizer.G = eps

    return SGD_base(func, x0, e, _optimizer, lr, eps)


def Prop(func: TargetFunction,
         x0: Vector,
         e: float = 0.01,
         lr: float = 0.01,
         gamma: float = 0.9,
         eps: float = 0.01,
         grad_func: Callable[[Vector], Vector] = None,
         mean_func: Callable[[Vector], Vector] = None) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, gamma, eps):
        EG = _optimizer.EG

        EG = gamma * EG + (1 - gamma) * grad_func(grad)
        v = lr / mean_func(EG + eps) * grad
        dx = v

        _optimizer.EG = EG
        return dx

    _optimizer.EG = 0

    return SGD_base(func, x0, e, _optimizer, lr, gamma, eps)


def RMSProp(func: TargetFunction,
            x0: Vector,
            e: float = 0.01,
            lr: float = 0.01,
            gamma: float = 0.9,
            eps: float = 0.01) -> tuple[Vector, Trajectory]:
    grad_func = lambda x: x ** 2
    mean_func = lambda x: np.sqrt(x)
    return Prop(func, x0, e, lr, gamma, eps, grad_func, mean_func)


def RMAProp(func: TargetFunction,
            x0: Vector,
            e: float = 0.01,
            lr: float = 0.01,
            gamma: float = 0.9,
            eps: float = 0.01) -> tuple[Vector, Trajectory]:
    grad_func = lambda x: x
    mean_func = lambda x: np.sqrt(x)
    return Prop(func, x0, e, lr, gamma, eps, grad_func, mean_func)


def MAProp(func: TargetFunction,
           x0: Vector,
           e: float = 0.01,
           lr: float = 0.01,
           gamma: float = 0.9,
           eps: float = 0.01) -> tuple[Vector, Trajectory]:
    grad_func = lambda x: x
    mean_func = lambda x: x
    return Prop(func, x0, e, lr, gamma, eps, grad_func, mean_func)


def ReverseRMSProp(func: TargetFunction,
                   x0: Vector,
                   e: float = 0.01,
                   lr: float = 0.01,
                   gamma: float = 0.9,
                   eps: float = 0.01) -> tuple[Vector, Trajectory]:
    grad_func = lambda x: x ** 2
    mean_func = lambda x: 1 / np.sqrt(x)
    return Prop(func, x0, e, lr, gamma, eps, grad_func, mean_func)


def Adadelta(func: TargetFunction,
             x0: Vector,
             e: float = 0.01,
             lr: float = 0.01,
             gamma: float = 0.9,
             eps: float = 0.01) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, gamma, eps):
        EG = _optimizer.EG
        D = _optimizer.D
        v = _optimizer.v

        D = gamma * D + (1 - gamma) * v ** 2
        EG = gamma * EG + (1 - gamma) * grad ** 2
        v = lr * np.sqrt(D + eps) / np.sqrt(EG + eps) * grad
        dx = v

        _optimizer.EG = EG
        _optimizer.D = D
        _optimizer.v = v
        return dx

    _optimizer.EG = 0
    _optimizer.D = 0
    _optimizer.v = 0

    return SGD_base(func, x0, e, _optimizer, lr, gamma, eps)


def Adam(func: TargetFunction,
         x0: Vector,
         e: float = 0.01,
         lr: float = 0.01,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 0.01) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, b1, b2, eps):
        t = _optimizer.t
        m = _optimizer.m
        v = _optimizer.v

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mm = m / (1 - b1 ** t)
        vv = v / (1 - b2 ** t)
        dx = lr * mm / np.sqrt(vv + eps)

        _optimizer.t = t + 1
        _optimizer.m = m
        _optimizer.v = v
        return dx

    _optimizer.t = 1
    _optimizer.m = 0
    _optimizer.v = 0

    return SGD_base(func, x0, e, _optimizer, lr, b1, b2, eps)


def Adamax(func: TargetFunction,
           x0: Vector,
           e: float = 0.01,
           lr: float = 0.01,
           b1: float = 0.9,
           b2: float = 0.999) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, b1, b2):
        t = _optimizer.t
        m = _optimizer.m
        v = _optimizer.v

        m = b1 * m + (1 - b1) * grad
        v = np.maximum(b2 * v, np.fabs(grad))
        mm = m / (1 - b1 ** t)
        dx = lr * mm / v

        _optimizer.t = t + 1
        _optimizer.m = m
        _optimizer.v = v
        return dx

    _optimizer.t = 1
    _optimizer.m = 0
    _optimizer.v = 0

    return SGD_base(func, x0, e, _optimizer, lr, b1, b2)


def Nadam(func: TargetFunction,
          x0: Vector,
          e: float = 0.01,
          lr: float = 0.01,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 0.01) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, b1, b2, eps):
        t = _optimizer.t
        m = _optimizer.m
        v = _optimizer.v

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mm = m / (1 - b1 ** t)
        vv = v / (1 - b2 ** t)
        dx = lr / np.sqrt(vv + eps) * (b1 * mm + (1 - b1) / (1 - b1 ** t) * grad)

        _optimizer.t = t + 1
        _optimizer.m = m
        _optimizer.v = v
        return dx

    _optimizer.t = 1
    _optimizer.m = 0
    _optimizer.v = 0

    return SGD_base(func, x0, e, _optimizer, lr, b1, b2, eps)


def AMSGrad(func: TargetFunction,
            x0: Vector,
            e: float = 0.01,
            lr: float = 0.01,
            b1: float = 0.9,
            b2: float = 0.999,
            eps: float = 0.01) -> tuple[Vector, Trajectory]:
    def _optimizer(grad, lr, b1, b2, eps):
        t = _optimizer.t
        m = _optimizer.m
        v = _optimizer.v
        vv = _optimizer.vv

        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        vv = np.maximum(vv, v)
        dx = lr * m / np.sqrt(vv + eps)

        _optimizer.t = t + 1
        _optimizer.m = m
        _optimizer.v = v
        _optimizer.vv = vv
        return dx

    _optimizer.t = 1
    _optimizer.m = 0
    _optimizer.v = 0
    _optimizer.vv = x0 - x0

    return SGD_base(func, x0, e, _optimizer, lr, b1, b2, eps)
