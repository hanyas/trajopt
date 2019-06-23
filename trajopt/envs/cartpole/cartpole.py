import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class Cartpole(gym.Env):

    def __init__(self):
        self.nb_xdim = 4
        self.nb_udim = 1

        self._sigma = 1.e-4 * np.eye(self.nb_xdim)

        self._dt = 0.01
        self._g = np.array([0., 2. * np.pi, 0., 0.])

        self._xw = - self._dt * 1. * np.array([1.e-1, 1.e1, 1.e-1, 1.e-1])
        self._uw = - self._dt * 1. * np.array([1.e-3])

        # x = [x, th, dx, dth]
        self._xmax = np.array([100., np.inf, 25., 25.])
        self._umax = 5.0

        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self.seed()
        self.reset()

    @property
    def sigma(self):
        return self._sigma

    def dynamics(self, x, u):
        # import from: https://github.com/JoeMWatson/input-inference-for-control/
        # x = [x, th, dx, dth]
        g = 9.81
        Mc = 0.37
        Mp = 0.127
        Mt = Mc + Mp
        l = 0.3365

        th = x[1]
        dth2 = np.power(x[3], 2)
        sth = np.sin(th)
        cth = np.cos(th)

        _num = - Mp * l * sth * dth2 + Mt * g * sth - u * cth
        _denom = l * ((4. / 3.) * Mt - Mp * cth ** 2)
        th_acc = _num / _denom
        x_acc = (Mp * l * sth * dth2 - Mp * l * th_acc * cth + u) / Mt

        xn = np.hstack((x[0] + self._dt * (x[2] + self._dt * x_acc),
                       x[1] + self._dt * (x[3] + self._dt * th_acc),
                       x[2] + self._dt * x_acc,
                       x[3] + self._dt * th_acc))
        return xn

    def reward(self, x, u, a):
        if a:
            return (x - self._g).T @ np.diag(self._xw) @ (x - self._g) + u.T @ np.diag(self._uw) @ u
        else:
            return u.T @ np.diag(self._uw) @ u

    def initialize(self):
        # mu, sigma
        return np.array([0., np.pi, 0., 0.]), 1.e-4 * np.eye(self.nb_xdim)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.state = self.dynamics(self.state, u)
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=self.sigma)
        return self.state, [], False, {}

    def reset(self):
        _mu_0, _sigma_0 = self.initialize()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return self.state
