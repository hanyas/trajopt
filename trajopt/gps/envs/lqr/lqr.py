import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class _LQRBase:

    def __init__(self, dt, xw, uw, goal):
        self._dt = dt
        self._g = goal

        self._xw = xw
        self._uw = uw

        self.xmax = np.array([np.inf, np.inf])
        self.umax = np.inf

        self._A = np.array([[1., 1.e-2], [0., 1.]])
        self._B = np.array([[0.], [1.]])
        self._c = np.zeros((2, ))

        self.sigma = 1.e-4 * np.eye(2)

    def dyn(self, x, u):
        u = np.clip(u, -self.umax, self.umax)

        xn = np.einsum('kh,h->k', self._A, x) + \
             np.einsum('kh,h->k', self._B, u) + self._c

        xn = np.clip(xn, -self.xmax, self.xmax)
        return xn

    def rwrd(self, x, u, a):
        if a:
            return (x - self._g).T @ np.diag(self._xw) @ (x - self._g) + u.T @ np.diag(self._uw) @ u
        else:
            return u.T @ np.diag(self._uw) @ u

    def init(self):
        # mu, sigma
        return np.array([0., 0.]), 1.e-4 * np.eye(2)


class LQR(gym.Env):

    def __init__(self):

        self._dt = 0.01
        self._xw = - 1. * np.array([1.e1, 1.])
        self._uw = - 1. * np.array([1.e-3])
        self._g = np.array([1., 0])

        self._model = _LQRBase(self._dt, self._xw, self._uw, self._g)

        self.low_state = - self.model.xmax
        self.high_state = self.model.xmax

        self.action_space = spaces.Box(low=-self.model.umax,
                                       high=self.model.umax, shape=(1,))

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)

        self.seed()
        self.reset()

    @property
    def model(self):
        return self._model

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.state = self._model.dyn(self.state, u)
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=self.model.sigma)
        return self.state, [], False, {}

    def reset(self):
        _mu_0, _sigma_0 = self._model.init()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return self.state
