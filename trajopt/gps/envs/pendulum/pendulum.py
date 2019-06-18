import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian


class _PendulumBase:

    def __init__(self, dt, xw, uw, goal):
        self.nb_xdim = 2
        self.nb_udim = 1

        self._dt = dt
        self._g = goal

        self._Rxx = np.diag(xw)
        self._rx = - 2. * np.diag(xw) @ self._g

        self._Ruu = np.diag(uw)
        self._ru = np.zeros((self.nb_udim, ))

        self._Rxu = np.zeros((self.nb_xdim, self.nb_udim))
        self._r0 = self._g.T @ np.diag(xw) @ self._g

        self._sigma = 1.e-2 * np.eye(self.nb_xdim)

        self.xmax = np.array([np.inf, 25.0])
        self.umax = 2.0

        # damping
        self._k = 1.e-3

        self._dydx = jacobian(self.step, 0)
        self._dydu = jacobian(self.step, 1)

    def step(self, x, u):
        u = np.clip(u, -self.umax, self.umax)
        th, dth = x

        g, m, l = 9.80665, 1.0, 1.0

        dthn = dth + (g * l * m * np.sin(th) + u - self._k * dth) * self._dt
        thn = th + dthn * self._dt

        xn = np.hstack((thn, dthn))
        xn = np.clip(xn, -self.xmax, self.xmax)

        return xn

    def rwrd(self, x=None, u=None):
        T = x.shape[1]
        _Rxx = np.zeros((self.nb_xdim, self.nb_xdim, T + 1))
        _rx = np.zeros((self.nb_xdim, T + 1))

        _Ruu = np.zeros((self.nb_udim, self.nb_udim, T + 1))
        _ru = np.zeros((self.nb_udim, T + 1))

        _Rxu = np.zeros((self.nb_xdim, self.nb_udim, T + 1))
        _r0 = np.zeros((T + 1, ))

        for t in range(T):
            if t > T - 25:
                _Rxx[..., t] = self._Rxx
                _rx[..., t] = self._rx
                _Rxu[..., t] = self._Rxu
            else:
                _Rxx[..., t] = np.zeros((self.nb_xdim, self.nb_xdim))
                _rx[..., t] = np.zeros((self.nb_xdim, ))
                _Rxu[..., t] = np.zeros((self.nb_xdim, self.nb_udim))

            _Ruu[..., t] = self._Ruu
            _ru[..., t] = self._ru
            _r0[..., t] = self._r0

        return _Rxx, _rx, _Ruu, _ru, _Rxu, _r0

    def dyn(self, x=None, u=None):
        T = x.shape[1] - 1
        _A = np.zeros((self.nb_xdim, self.nb_xdim, T))
        _B = np.zeros((self.nb_xdim, self.nb_udim, T))
        _c = np.zeros((self.nb_xdim, T))
        _sigma = np.zeros((self.nb_xdim, self.nb_xdim, T))

        for t in range(T):
            _A[..., t] = self._dydx(x[..., t], u[..., t])
            _B[..., t] = self._dydu(x[..., t], u[..., t])
            _c[..., t] = x[..., t + 1] - _A[..., t] @ x[..., t] - _B[..., t] @ u[..., t]
            _sigma[..., t] = self._sigma

        return _A, _B, _c, _sigma

    def init(self):
        # mu, sigma
        return np.array([np.pi, 0.]), 1.e-4 * np.eye(2)


class Pendulum(gym.Env):

    def __init__(self):
        self._dt = 0.025
        self._xw = - self._dt * 1. * np.array([1.e1, 1.e-1])
        self._uw = - self._dt * 1. * np.array([1.e-3])
        self._g = np.array([2. * np.pi, 0.])

        self._model = _PendulumBase(self._dt, self._xw, self._uw, self._g)

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
        return self.state, [], False, {}

    def reset(self):
        _mu_0, _sigma_0 = self._model.init()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return self.state
