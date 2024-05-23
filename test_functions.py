import numpy as np
from numpy.linalg import norm

from domain import Vector, vec
from tools import add_information_and_counter


@add_information_and_counter('Parabolic #1',
                             lambda x1, x2: x1 ** 2 + x2 ** 2,
                             [vec(0, 0)],
                             0.0,
                             [vec(-10, 10), vec(-10, 10)])
def parabolic_1(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(parabolic_1.formula(x1, x2))


@add_information_and_counter('Parabolic',
                             lambda x1, x2: 2 * x1 ** 2 + x1 * x2 + x2 ** 2 - 3 * x1,
                             [vec(0.8571, -0.4285)],
                             -1.2857,
                             [vec(-10, 10), vec(-10, 10)])
def parabolic_2(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(parabolic_2.formula(x1, x2))


@add_information_and_counter('Rozenbrock',
                             lambda x1, x2: (1 - x1) ** 2 + 10 * (x2 - x1 ** 2) ** 2,
                             [vec(1, 1)],
                             0.0,
                             [vec(-5, 5), vec(-5, 5)])
def rozenbrock_1(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(rozenbrock_1.formula(x1, x2))


@add_information_and_counter('Rozenbrock a=100',
                             lambda x1, x2: (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2,
                             [vec(1, 1)],
                             0.0,
                             [vec(-10, 10), vec(-10, 10)])
def rozenbrock_2(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(rozenbrock_2.formula(x1, x2))


@add_information_and_counter('Sinusoid Schwefel',
                             lambda x1, x2: -x1 * np.sin(np.sqrt(np.fabs(x1))) - x2 * np.sin(np.sqrt(np.fabs(x2))),
                             [vec(420.9687, 420.9687)],
                             -837.9657,
                             [vec(-500, 500), vec(-500, 500)])
def sinusoid_schwefel(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(sinusoid_schwefel.formula(x1, x2))


@add_information_and_counter('Multi',
                             lambda x1, x2: -x1 * np.sin(np.sqrt(np.fabs(x1))) - x2 * np.sin(np.sqrt(np.fabs(x2))),
                             [vec(1.6288, 1.6288), vec(1.6288, -1.6288), vec(-1.6288, 1.6288), vec(-1.6288, -1.6288)],
                             -4.2539,
                             [vec(-2, 2), vec(-2, 2)])
def multi(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(multi.formula(x1, x2))


@add_information_and_counter('Root',
                             lambda x1, x2: -1 / (1 + abs((x1 + x2 * 1j) ** 6 - 1)),
                             [vec(0.5, -0.866),
                              vec(-0.5, 0.866),
                              vec(0.5, 0.866),
                              vec(-0.5, -0.866),
                              vec(1, 0),
                              vec(-1, 0)],
                             -1.0,
                             [vec(-3, 3), vec(-3, 3)])
def root(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(root.formula(x1, x2))


@add_information_and_counter('Shuffer',
                             lambda x1, x2: -0.5 + (np.sin(np.sqrt(x1 ** 2 + x2 ** 2)) - 0.5) /
                                            (1 + 0.001 * (x1 ** 2 + x2 ** 2)),
                             [vec(0, 0)],
                             -1.0,
                             [vec(-10, 10), vec(-10, 10)])
def shuffer(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(shuffer.formula(x1, x2))


@add_information_and_counter('Rastrigin',
                             lambda x1, x2: 20 - (10 * np.cos(2 * np.pi * x1) - x1 ** 2) -
                                            (10 * np.cos(2 * np.pi * x2) - x2 ** 2),
                             [vec(0, 0)],
                             0.0,
                             [vec(-5, 5), vec(-5, 5)])
def rastrigin(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(rastrigin.formula(x1, x2))


@add_information_and_counter('Trihumped',
                             lambda x1, x2: 2 * x1 ** 2 - 1.05 * x1 ** 4 + 1 / 6 * x1 ** 6 + x1 * x2 + x2 ** 2,
                             [vec(0, 0)],
                             0.0,
                             [vec(-5, 5), vec(-5, 5)])
def trihumped(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(trihumped.formula(x1, x2))


@add_information_and_counter('Bird',
                             lambda x1, x2: np.sin(x1) * np.exp((1 - np.cos(x2)) ** 2) + np.cos(x2) *
                                            np.exp((1 - np.sin(x1)) ** 2) - (x1 - x2) ** 2,
                             [vec(4.7124, 3.1416), vec(-1.5708, -3.1416)],
                             -106.7645,
                             [vec(-2 * np.pi, 2 * np.pi), vec(-2 * np.pi, 2 * np.pi)])
def bird(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(bird.formula(x1, x2))


@add_information_and_counter('Ackley',
                             lambda x1, x2: np.e - 20 * np.exp(-np.sqrt((x1 ** 2 + x2 ** 2) / 50)) -
                                            np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))),
                             [vec(0, 0)],
                             -20.0,
                             [vec(-3, 3), vec(-3, 3)])
def ackley(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(ackley.formula(x1, x2))


@add_information_and_counter('Bukin',
                             lambda x1, x2: 100 * np.sqrt(np.fabs(-x2 + 0.01 * x1 ** 2)) + 0.01 * np.fabs(x1 + 10),
                             [vec(-10, 1)],
                             0.0,
                             [vec(-15, 5), vec(-3, 3)])
def bukin(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(bukin.formula(x1, x2))


@add_information_and_counter('Schwefel #1',
                             lambda x1, x2: x1 ** 2 + (x1 + x2) ** 2,
                             [vec(0, 0)],
                             0.0,
                             [vec(-10, 10), vec(-10, 10)])
def schwefel_1(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(schwefel_1.formula(x1, x2))


@add_information_and_counter('Schwefel #2',
                             lambda x1, x2: np.fabs(x1) + np.fabs(x2) + np.fabs(x1) * np.fabs(x2),
                             [vec(0, 0)],
                             0.0,
                             [vec(-10, 10), vec(-10, 10)])
def schwefel_2(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(schwefel_2.formula(x1, x2))


@add_information_and_counter('Two extremum',
                             lambda x1, x2: 3 * x1 ** 2 + 4 * x2 ** 2 + 23 * np.cos(x1 - 0.5),
                             [vec(-2.0709, 0)],
                             -6.4892,
                             [vec(-6, 6), vec(-6, 6)])
def two_extremum(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(two_extremum.formula(x1, x2))


@add_information_and_counter('Grivanka',
                             lambda x1, x2: 1 + (x1 ** 2 + x2 ** 2) / 4000 - np.cos(x1) * np.cos(x2 / np.sqrt(2)),
                             [vec(0, 0)],
                             0.0,
                             [vec(-10, 10), vec(-10, 10)])
def grivanka(x: Vector) -> float:
    x1, x2 = x[0], x[1]
    return float(grivanka.formula(x1, x2))


all_functions = [parabolic_2, rozenbrock_1, trihumped, two_extremum, ackley]

if __name__ == '__main__':
    print(parabolic_2(vec(0.8571, -0.4285)))
