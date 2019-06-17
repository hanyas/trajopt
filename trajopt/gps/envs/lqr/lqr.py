import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class _LQRBase:

    def __init__(self, dt, xw, uw, goal=np.zeros((2, ))):
        self._dt = dt
        self._g = goal

        self._Rxx = np.diag(xw)
        self._rx = - 2. * np.diag(xw) @ self._g

        self._Ruu = np.diag(uw)
        self._ru = np.zeros((1, ))

        self._Rxu = np.zeros((2, 1))
        self._r0 = self._g.T @ np.diag(xw) @ self._g

        self._A = np.array([[1., 1.e-2], [0., 1.]])
        self._B = np.array([[0.], [1.]])
        self._c = np.zeros((2, ))

        self._sigma = 1.e-2 * np.eye(2)

        self.xmax = np.array([np.inf, np.inf])
        self.umax = np.inf

    def step(self, x, u):
        u = np.clip(u, -self.umax, self.umax)

        xn = np.einsum('kh,h->k', self._A, x) + \
             np.einsum('kh,h->k', self._B, u) + self._c

        xn = np.clip(xn, -self.xmax, self.xmax)
        xn = np.random.multivariate_normal(xn, self._sigma)

        return xn

    def rwrd(self, x=None, u=None):
        T = x.shape[1]
        _Rxx = np.zeros((2, 2, T + 1))
        _rx = np.zeros((2, T + 1))

        _Ruu = np.zeros((1, 1, T + 1))
        _ru = np.zeros((1, T + 1))

        _Rxu = np.zeros((2, 1, T + 1))
        _r0 = np.zeros((T + 1, ))

        for t in range(T):
            _Rxx[..., t] = self._Rxx
            _rx[..., t] = self._rx

            _Ruu[..., t] = self._Ruu
            _ru[..., t] = self._ru

            _Rxu[..., t] = self._Rxu
            _r0[..., t] = self._r0

        return _Rxx, _rx, _Ruu, _ru, _Rxu, _r0

    def dyn(self, x=None, u=None):
        T = x.shape[1] - 1
        _A = np.zeros((2, 2, T))
        _B = np.zeros((2, 1, T))
        _c = np.zeros((2, T))
        _sigma = np.zeros((2, 2, T))

        for t in range(T):
            _A[..., t] = self._A
            _B[..., t] = self._B
            _c[..., t] = self._c
            _sigma[..., t] = self._sigma

        return _A, _B, _c, _sigma

    def init(self):
        # mu, sigma
        return np.array([0., 0.]), 1.e-2 * np.eye(2)

class LQR(gym.Env):

    def __init__(self):

        self._dt = 0.01
        self._xw = - 1. * np.array([1.e4, 1.])
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
        self.state = self._model.step(self.state, u)
        return self.state, [], False, {'dyn': self._model.dyn(), 'rwrd': self._model.rwrd()}

    def reset(self):
        _mu_0, _sigma_0 = self._model.init()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return np.array(self.state)

    def render(self, mode='human'):
        return
