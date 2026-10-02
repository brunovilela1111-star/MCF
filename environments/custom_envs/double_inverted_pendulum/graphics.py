# renderização visual do ambiente

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def initialize_double_pendulum_renderer(name: str):
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0.0, color="black")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(name)

    bar1 = ax.plot([], [], "-o", linewidth=4, markersize=8)[0]
    bar2 = ax.plot([], [], "-o", linewidth=4, markersize=8)[0]

    return fig, ax, bar1, bar2

def compute_double_pendulum_bars(
    state: np.ndarray,
    L1: float,
    L2: float,
):
    x = state.flatten()

    cart_x = x[0]
    theta1 = x[1]
    theta2 = x[2]

    line_1_x = np.array(
        [
            cart_x,
            cart_x + L1 * np.sin(theta1),
        ]
    )
    line_1_y = np.array(
        [
            0.0,
            L1 * np.cos(theta1),
        ]
    )

    line_2_x = np.array(
        [
            line_1_x[1],
            line_1_x[1] + L2 * np.sin(theta2),
        ]
    )
    line_2_y = np.array(
        [
            line_1_y[1],
            line_1_y[1] + L2 * np.cos(theta2),
        ]
    )

    line_1 = np.stack((line_1_x, line_1_y))
    line_2 = np.stack((line_2_x, line_2_y))

    return line_1, line_2

def render_double_pendulum(
    fig,
    ax,
    bar1,
    bar2,
    state: np.ndarray,
    L1: float,
    L2: float,
):
    line1, line2 = compute_double_pendulum_bars(
        state=state,
        L1=L1,
        L2=L2,
    )

    bar1.set_data(line1[0], line1[1])
    bar2.set_data(line2[0], line2[1])

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.001)

def close_double_pendulum_renderer(fig) -> None:
    if fig is not None:
        plt.close(fig)