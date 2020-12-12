import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


def wrap_angle(x):
    # wraps angle between [-pi, pi]
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Pendulum(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1

        self.dt = 0.01

        self.sigma = 1e-8 * np.eye(self.dm_state)

        # g = [th, thd]
        self.g = np.array([0., 0.])
        self.gw = np.array([1e1, 1e-1])

        # x = [th, thd]
        self.xmax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.uw = np.array([1e-5])
        self.umax = 2.5
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(1,))

        self.x0 = np.array([np.pi, 0.])
        self.sigma0 = 1e-4 * np.eye(self.dm_state)

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

        g, m, l, k = 10., 1., 1., 1e-3

        def f(x, u):
            th, dth = x
            return np.hstack((dth, - 3. * g / (2. * l) * np.sin(th + np.pi) +
                              3. / (m * l ** 2) * (u - k * dth)))

        k1 = f(x, _u)
        k2 = f(x + 0.5 * self.dt * k1, _u)
        k3 = f(x + 0.5 * self.dt * k2, _u)
        k4 = f(x + self.dt * k3, _u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.clip(xn, -self.xlim, self.xlim)

        return xn

    def inverse_dynamics(self, x, u):
        _u = np.clip(u, -self.ulim, self.ulim)

        g, m, l, k = 10., 1., 1., 1e-3

        def f(x, u):
            th, dth = x
            return np.hstack((dth, - 3. * g / (2. * l) * np.sin(th + np.pi) +
                              3. / (m * l ** 2) * (u - k * dth)))

        k1 = f(x, _u)
        k2 = f(x - 0.5 * self.dt * k1, _u)
        k3 = f(x - 0.5 * self.dt * k2, _u)
        k4 = f(x - self.dt * k3, _u)

        xn = x - self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.clip(xn, -self.xlim, self.xlim)

        return xn

    def features(self, x):
        return x

    def features_jacobian(self, x):
        J = jacobian(self.features, 0)
        j = self.features(x) - J(x) @ x
        return J, j

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self.ulim, self.ulim)
        _x = np.clip(x, -self.xlim, self.xlim)
        return self.sigma

    def cost(self, x, u, a):
        if a:
            x = np.hstack((wrap_angle(x[0]), x[1]))
            J, j = self.features_jacobian(getval(x))
            _x = J(getval(x)) @ x + j
            return (_x - self.g).T @ np.diag(self.gw) @ (_x - self.g)\
                   + u.T @ np.diag(self.uw) @ u
        else:
            return u.T @ np.diag(self.uw) @ u

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


class PendulumWithCartesianCost(Pendulum):

    def __init__(self):
        super(PendulumWithCartesianCost, self).__init__()

        # g = [cs_th, sn_th, dth]
        self.g = np.array([1., 0., 0.])
        self.gw = np.array([1e1, 1e1, 1e-1])

    def features(self, x):
        return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])
