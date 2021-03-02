import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from scipy.stats import beta


class LQRv1(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1

        self.dt = 0.01

        self.x0 = np.array([0., 0.])
        self.g = np.array([1., 0.])

        self.gw = np.array([1e2, 1e0])
        self.uw = np.array([1e-3])

        self.m, self.k, self.d = 1., 1e-2, 1e-1
        self.A = lambda m, k, d: np.array([[0.     , 1e0       ],
                                           [- k / m, -2 * d / m]])
        self.B = np.array([[0.], [1.]])
        self.c = np.array([0., 0.])

        self.umax = np.inf
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(1,))

        self.xmax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.sigma = 1e-8 * np.eye(self.dm_state)
        self.sigma0 = 1e-2 * np.eye(self.dm_state)

        self.perturb = False

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

        m, k, d = self.m, self.k, self.d
        if self.perturb:
            m += np.asscalar(0.5 * beta(2., 5.).rvs())

        def f(x, u):
            return self.A(m, k, d) @ x + self.B @ u + self.c

        k1 = f(x, _u)
        k2 = f(x + 0.5 * self.dt * k1, _u)
        k3 = f(x + 0.5 * self.dt * k2, _u)
        k4 = f(x + self.dt * k3, _u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
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
