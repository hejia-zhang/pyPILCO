import abc

import gpflow
import numpy as np
import tensorflow as tf


class Cost(gpflow.Parameterized):
    def __init__(self):
        gpflow.Parameterized.__init__(self)

    @abc.abstractmethod
    def compute_reward(self, state_mean, state_variance):
        pass


class SaturatingCost(Cost):
    """
    Efficient Reinforcement Learning using Gaussian Processes.
    p53
    """
    def __init__(self, state_dim, precision_matrix=None, state_target=None):
        Cost.__init__(self)
        self.state_dim = state_dim
        if precision_matrix is not None:
            self.precision_matrix = gpflow.Param(np.reshape(precision_matrix, (state_dim, state_dim)), trainable=False)
        else:
            self.precision_matrix = gpflow.Param(np.ones((state_dim, state_dim), dtype=np.float32), trainable=False)
        if state_target is not None:
            self.state_target = gpflow.Param(np.reshape(state_target, (1, state_dim)), trainable=False)
        else:
            self.state_target = gpflow.Param(np.zeros((1, state_dim)), trainable=False)

    @gpflow.params_as_tensors
    def compute_reward(self, state_mean, state_variance):
        sp = state_variance @ self.precision_matrix
        tilde_s_1 = self.precision_matrix @ tf.matrix_inverse(tf.eye(self.state_dim) + sp)
        tilde_s_2 = self.precision_matrix @ tf.matrix_inverse(tf.eye(self.state_dim) + 2 * sp)

        cost_mean = 1 - tf.exp(tf.constant(-0.5, dtype=np.float32) * tf.transpose(state_mean - self.state_target) @ tilde_s_1 @ (state_mean - self.state_target)) / tf.sqrt(tf.linalg.det(tf.eye(self.state_dim, dtype=np.float32) + sp))
        cost_sq_mean = tf.exp(-tf.transpose(state_mean - self.state_target) @ tilde_s_2 @ (state_mean - self.state_target)) / tf.sqrt(tf.linalg.det(tf.eye(self.state_dim, dtype=np.float32) + 2 * sp))

        cost_variance = cost_sq_mean - cost_mean @ cost_mean

        cost_mean.set_shape([1, 1])
        cost_variance.set_shape([1, 1])

        return cost_mean, cost_variance






