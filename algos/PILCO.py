import numpy as np


class PILCO(object):
    def __init__(self, env, policy, horizon):
        self.env = env
        self.policy = policy
        self.horizon = horizon

    def rollout(self):
        """
        Rollout given steps and collect new data.

        :param policy:
        :param timesteps:
        :return: X, delta_X
        """
        X = []
        Y = []

        x = self.env.reset()

        for timestep in range(self.horizon):
            u = self.policy(x)
            x_p, _, _, _ = self.env.step(u)
            X.append(np.hstack((x, u)))
            Y.append(x_p - x)
            x = x_p

            self.env.render()
        return np.stack(X), np.stack(Y)




