import gpflow
import numpy as np
import scipy as sp

from robots.pendubot import Pendubot


def cost():



def run():
    start_state = np.array([np.pi / 4., 0., 0., 0.])
    pbot = Pendubot(start_state)
    target_point = np.array([0., 1., 0., 2.])

    fi = np.random.normal(0.0, 1.0, size=(1, 5))

    # sampling period for pbot
    dt_pbot = 0.010
    t_full = np.arange(0.0, 4.0, dt_pbot)
    points = np.zeros((t_full.shape[0], 4))

    # sampling period for PILCO
    dt_pilco = 1. / 20

    # PILCO algo:
    # Step1 Get some random moves
    pbot.step(t_full)
    pbot.get_points(points)

    kernel = gpflow.kernels.SquaredExponential(input_dim=5)
    m = gpflow.models.GPR(pbot.xu, pbot.delta_x, kernel)

    sp.optimize.minimize()

if __name__ == '__main__':
    run()
