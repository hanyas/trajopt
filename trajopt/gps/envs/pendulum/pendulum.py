import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class Pendulum(gym.Env):

    def __init__(self):
        self.nb_xdim = 2
        self.nb_udim = 1

        self._sigma = 1.e-4 * np.eye(self.nb_xdim)

        # damping
        self._k = 1.e-2

        self._dt = 0.025
        self._g = np.array([2. * np.pi, 0.])

        self._xw = - self._dt * 1. * np.array([1.e1, 1.e-1])
        self._uw = - self._dt * 1. * np.array([1.e-3])

        self._xmax = np.array([np.inf, 25.0])
        self._umax = 2.0

        self.low_state = - self._xmax
        self.high_state = self._xmax

        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)

        self.seed()
        self.reset()

    @property
    def sigma(self):
        return self._sigma

    def dynamics(self, x, u):
        # u = np.clip(u, -self.umax, self.umax)
        th, dth = x

        g, m, l = 9.80665, 1.0, 1.0

        dthn = dth + self._dt * (3. * g / (2 * l) * np.sin(th) +
                                 3. / (m * l ** 2) * (u - self._k * dth))
        # dthn = dth + self._dt * (g * l * m * np.sin(th) + u - self._k * dth)

        thn = th + dthn * self._dt

        xn = np.hstack((thn, dthn))
        return xn

    def reward(self, x, u, a):
        if a:
            return (x - self._g).T @ np.diag(self._xw) @ (x - self._g) + u.T @ np.diag(self._uw) @ u
        else:
            return u.T @ np.diag(self._uw) @ u

    def initialize(self):
        # mu, sigma
        return np.array([np.pi, 0.]), 1.e-4 * np.eye(self.nb_xdim)

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
