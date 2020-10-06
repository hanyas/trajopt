import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class LQR(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1

        self._dt = 0.1

        self._g = np.array([10., 10.])
        self._gw = np.array([1e1, 1e1])

        self._uw = np.array([1.])
        self._umax = np.inf
        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

        self._xmax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        # stochastic dynamics
        self._A = np.array([[1.1, 0.], [1.0, 1.0]])
        self._B = np.array([[1.], [0.]])
        self._c = - self._A @ self._g  # stable at goal

        self._sigma = 1e-8 * np.eye(self.dm_state)

        self.state = None
        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self._xmax

    @property
    def ulim(self):
        return self._umax

    @property
    def dt(self):
        return self._dt

    @property
    def goal(self):
        return self._g

    def dynamics(self, x, u):
        _u = np.clip(u, -self.ulim, self.ulim)

        def f(x, u):
            return self._A @ x + self._B @ u + self._c

        k1 = f(x, _u)
        k2 = f(x + 0.5 * self.dt * k1, _u)
        k3 = f(x + 0.5 * self.dt * k2, _u)
        k4 = f(x + self.dt * k3, _u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.clip(xn, -self.xlim, self.xlim)

        return xn

    def inverse_dynamics(self, x, u):
        _u = np.clip(u, -self.ulim, self.ulim)

        def f(x, u):
            return self._A @ x + self._B @ u + self._c

        k1 = f(x, _u)
        k2 = f(x - 0.5 * self.dt * k1, _u)
        k3 = f(x - 0.5 * self.dt * k2, _u)
        k4 = f(x - self.dt * k3, _u)

        xn = x - self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.clip(xn, -self.xlim, self.xlim)

        return xn

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self.ulim, self.ulim)
        _x = np.clip(x, -self.xlim, self.xlim)
        return self._sigma

    def cost(self, x, u, u_lst):
        return self.dt * ((x - self._g).T @ np.diag(self._gw) @ (x - self._g)
                          + u.T @ np.diag(self._uw) @ u)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # state-action dependent noise
        _sigma = self.noise(self.state, u)
        # evolve deterministic dynamics
        self.state = self.dynamics(self.state, u)
        # add noise
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=_sigma)
        return self.state, [], False, {}

    def reset(self):
        self.state = np.array([5., 5.])
        return self.state
