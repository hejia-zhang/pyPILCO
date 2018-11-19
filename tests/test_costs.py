from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from costs import SaturatingCost


def run(x_target):
    cost = SaturatingCost(1, state_target=x_target)

    X = np.linspace(1, 100, 100)

    # tf.reset_default_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for x in X:
            cost_mean, cost_varaince = cost.compute_reward(np.array([[x]], dtype=np.float32),
                                                           np.array([[1]], dtype=np.float32))
            print(sess.run(cost_mean))


if __name__ == '__main__':
    config = dict(
        x_target=np.array([[0]], dtype=np.float32),
    )
    run(**config)
