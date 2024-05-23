from typing import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from domain import Trajectory, TargetFunction, vec, Vector, gradient

alpha = 0.1
beta = 0.001

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def pprint(obj, q=3, end='\n'):
    if type(obj) in (float, np.float64):
        s = str(round(obj, q))
    else:
        s = str(obj.round(q))

    print(s, end=end)


def _borders(func: TargetFunction, trajectories: Sequence[Trajectory]) -> tuple[Vector, Vector]:
    # Drop outliers trajectories
    borders = []
    for trajectory in trajectories:
        X, Y = trajectory[:, 0, 0], trajectory[:, 1, 0]
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        borders.append((vec(x_min, x_max), vec(y_min, y_max)))
    scopes = [norm(v2 - v1) for v1, v2 in borders]
    min_scope = min(scopes)
    borders = [border for border, scope in zip(borders[::], scopes) if scope <= 10 ** 4 * min_scope]
    borders += [tuple(func.limits)]

    global_min_x = min([v1[0][0] for v1, v2 in borders])
    global_max_x = max([v1[1][0] for v1, v2 in borders])
    global_min_y = min([v2[0][0] for v1, v2 in borders])
    global_max_y = max([v2[1][0] for v1, v2 in borders])

    return vec(global_min_x, global_max_x), vec(global_min_y, global_max_y)


def build_graph(trajectories: Sequence[Trajectory], func: TargetFunction, labels: Sequence[str] = None,
                save: bool = False, animate: bool = False) -> None:
    a, b = _borders(func, trajectories)

    scope_x = a[1][0] - a[0][0]
    scope_x = 2.0 if scope_x == 0.0 else scope_x
    scope_y = b[1][0] - b[0][0]
    scope_y = 2.0 if scope_y == 0.0 else scope_y
    delta_x = beta * scope_x
    delta_y = beta * scope_y
    X_mesh = np.arange(a[0][0] - alpha * scope_x, a[1][0] + alpha * scope_x, delta_x)
    Y_mesh = np.arange(b[0][0] - alpha * scope_y, b[1][0] + alpha * scope_y, delta_y)
    X, Y = np.meshgrid(X_mesh, Y_mesh)

    Z = func.formula(X, Y)

    fig, ax = plt.subplots(figsize=[8.0, 6.0])
    Z_flat = np.sort(Z.flatten())
    step = len(Z_flat) // 30
    levels = np.concatenate([np.asarray([Z_flat[0], Z_flat[step // 2]]), Z_flat[step::step]])
    CS = ax.contour(X, Y, Z, levels=20)
    cbar = plt.colorbar(CS)
    cbar.remove()

    nphistory = []
    for trajectory in trajectories:
        nphistory.append([trajectory[:, 0, 0], trajectory[:, 1, 0]])

    lines = []
    for nph, label in zip(nphistory, labels):
        lines += ax.plot(nph[0], nph[1], label=label)

    for op in func.ops:
        plt.plot([op[0][0]], [op[1][0]], '*')

    plt.plot(trajectory[0][0], trajectory[0][1], '.', color='black')

    plt.xlim(a[0][0] - alpha * scope_x, a[1][0] + alpha * scope_x, delta_x)
    plt.ylim(b[0][0] - alpha * scope_y, b[1][0] + alpha * scope_y, delta_y)
    plt.legend(loc='best')
    plt.title(func.name)

    if save:
        plt.savefig(f'resources/{func.name}', dpi=300)
    else:
        plt.show()

    if animate:
        def _animate(i):
            for line, hist in zip(lines, nphistory):
                line.set_xdata(hist[0][:i])
                line.set_ydata(hist[1][:i])
            return lines

        def _init():
            for line, hist in zip(lines, nphistory):
                line.set_ydata(np.ma.array(hist[0], mask=True))
            return lines

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, 301), init_func=_init,
                                      interval=100, repeat_delay=0, blit=True, repeat=True)

        ani.save(f'resources/{func.name}.mp4', writer='ffmpeg_file', fps=5)


def vec_to_str(v: Vector) -> str:
    x1, x2 = v[0][0], v[1][1]
    return f'({x1:.4f}, {x2:.4f})'

def show_results(target_functions: Sequence[TargetFunction], optimizers, optimizers_params, labels: Sequence[str],
                 start_point: Vector) -> None:
    for func in target_functions:
        trajectories = []
        print(f'\nFunction: {func.name}')
        for optimizer in optimizers:
            params = optimizers_params[optimizer.__name__]
            x, trajectory = optimizer(func, start_point, **params)
            grad = norm(gradient(func, x))
            point_error = min([norm(x - op) for op in func.ops])
            value_error = func(x) - func.min_value
            print(
                f'{optimizer.__name__}: n = {len(trajectory)}, point error = {round(point_error, 4)}, value error = {round(value_error, 4)}')
            trajectories.append(trajectory)

        build_graph(trajectories, func, labels)
