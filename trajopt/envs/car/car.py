import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class Car(gym.Env):

    def __init__(self):
        self.state_dim = 4
        self.belief_dim = 4
        self.act_dim = 2
        self.obs_dim = 2

        self.dt = 0.5

        self.dyn_sigma = 1e-8 * np.eye(self.state_dim)
        self.obs_sigma = 1e-8 * np.eye(self.obs_dim)

        self.goal = np.array([0., 0., 0., 0.])

        # car length
        self.l = 0.1

        # belief cost weights
        self.mu_w = 100. * np.array([1., 1., 1., 1.])
        self.sigma_w = 100. * np.array([1., 1., 1., 1.])

        # action cost weights
        self.act_w = np.array([1., 1.])

        self.xmax = np.array([np.inf, np.inf, np.inf, np.inf])
        self.zmax = np.array([np.inf, np.inf])
        self.umax = np.array([np.inf, np.inf])

        self.state_space = spaces.Box(low=-self.xmax,
                                       high=self.xmax)

        self.observation_space = spaces.Box(low=-self.zmax,
                                            high=self.zmax)

        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax)

        self.state = None
        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self.xmax

    @property
    def ulim(self):
        return self.umax

    def dynamics(self, x, u):
        _u = np.clip(u, -self.ulim, self.ulim)

        def f(x, u):
            return np.hstack([x[3] * np.cos(x[2]),
                              x[3] * np.sin(x[2]),
                              x[3] * np.tan(u[1]) / self.l,
                              u[0]])

        k1 = f(x, _u)
        k2 = f(x + 0.5 * self.dt * k1, _u)
        k3 = f(x + 0.5 * self.dt * k2, _u)
        k4 = f(x + self.dt * k3, _u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.clip(xn, -self.xlim, self.xlim)

        return xn

    def observe(self, x):
        return np.array([x[0], x[1]])

    def dyn_noise(self, x=None, u=None):
        _u = np.clip(u, -self.ulim, self.ulim)
        _x = np.clip(x, -self.xlim, self.xlim)
        return self.dyn_sigma

    def obs_noise(self, x=None):
        return self.obs_sigma +\
               np.array([[0.5 * (5. - x[0])**2, 0.],
                         [0., 0.]])

    # cost over belief
    def cost(self, mu_b, sigma_b, u):
        return (mu_b - self.goal).T @ np.diag(self.mu_w) @ (mu_b - self.goal) +\
               np.trace(np.diag(self.sigma_w) @ sigma_b) + \
               u.T @ np.diag(self.act_w) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # state-action dependent dynamics noise
        _sigma_dyn = self.dyn_noise(self.state, u)
        # evolve deterministic dynamics
        self.state = self.dynamics(self.state, u)
        # add dynamics noise
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=_sigma_dyn)

        # state-action dependent dynamics noise
        _sigma_obs = self.obs_noise(self.state)
        # observe state
        _z = self.observe(self.state)
        # add observation noise
        obs = self.np_random.multivariate_normal(mean=_z, cov=_sigma_obs)

        return obs, [], False, {}

    # initial belief
    def init(self):
        mu = np.array([2., 2., 0., 0.])
        sigma = 1. * np.eye(self.belief_dim)
        return mu, sigma

    def reset(self):
        self.state = np.array([0., 4., 0., 0.])
        return self.observe(self.state)
