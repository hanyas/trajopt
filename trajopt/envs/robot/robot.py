import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class Robot(gym.Env):

    def __init__(self):
        self.dm_state = 4
        self.dm_act = 2

        self.dt = 0.025

        # car length
        self.l = 0.1

        self.x0 = np.array([5., 5., 0., 0.])
        self.g = np.array([0., 0., 0., 0.])

        self.gw = np.array([1e1, 1e1, 1., 1.])
        self.uw = np.array([1e-3, 1e-3])

        self.slew_rate = False
        self.umax = np.array([np.inf, np.inf])
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax)

        self.xmax = np.array([np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.sigma0 = 1e-4 * np.eye(self.dm_state)
        self.sigma = 1e-8 * np.eye(self.dm_state)

        self.state = None
        self.np_random = None

        self.seed()
        self.reset()

    @property
    def xlim(self):
        return self.xmax

    @property
    def ulim(self):
        return self.umax

    def dynamics(self, x, u):
        _u = np.clip(u, -self.ulim, self.ulim)

        def f(x, u):
            # x, y, th, v
            x, y, th, v = x
            a, phi = u
            return np.array([v * np.cos(th),
                             v * np.sin(th),
                             v * np.tan(phi) / self.l,
                             a])
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
        return self.sigma

    def cost(self, x, u, u_last, a):
        c = u.T @ np.diag(self.uw) @ u
        if a:
            c += a * (x - self.g).T @ np.diag(self.gw) @ (x - self.g)

        return self.dt * c

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

    def init(self):
        return self.x0, self.sigma0

    def reset(self):
        self.state = self.np_random.multivariate_normal(mean=self.x0, cov=self.sigma0)
        return self.state
