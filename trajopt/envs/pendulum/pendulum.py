import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval

from scipy.stats import multivariate_normal
from scipy.stats import beta


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
        self.gw = np.array([1e4, 1e0])

        # x = [th, thd]
        self.xmax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.slew_rate = False
        self.uw = 1e-5 * np.ones((self.dm_act, ))
        self.umax = 10. * np.ones((self.dm_act, ))
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(1,))

        self.x0 = np.array([np.pi, 0.])
        self.sigma0 = 1e-4 * np.eye(self.dm_state)

        self.periodic = False
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

        g, m, l, k = 9.81, 1., 1., 0.025

        if self.perturb:
            m += 0.5 * beta(2., 5.).rvs()
            k += 1e-1 * beta(2., 4.).rvs()

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

        g, m, l, k = 9.81, 1., 1., 0.025

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

    def linearized(self, x, u, dist=None):
        _u = np.clip(u, -self.ulim, self.ulim)

        g, m, l, k = 9.81, 1., 1., 0.025

        def f(x, u):
            th, dth = x
            return np.hstack((dth, - 3. * g / (2. * l) * np.sin(th + np.pi) +
                              3. / (m * l ** 2) * (u - k * dth)))

        dfdx = jacobian(f, 0)
        dfdu = jacobian(f, 1)

        At = dfdx(x, _u)
        Bt = dfdu(x, _u)
        ct = f(x, _u) - At @ x - Bt @ _u

        dA = np.zeros((2, 2))
        if dist is not None:
            ABc = multivariate_normal(mean=dist['mu'], cov=dist['sigma']).rvs()
            dA = np.reshape(ABc[:self.dm_state ** 2], (self.dm_state, self.dm_state), order='F')
            dB = np.reshape(ABc[self.dm_state ** 2: self.dm_state ** 2 + self.dm_state * self.dm_act],
                            (self.dm_state, self.dm_act), order='F')
            dc = np.reshape(ABc[- self.dm_state:], (self.dm_state, ), order='F')
        else:
            dB = - 0.1 * beta(5., 1.).rvs(2)[:, None]
            dc = - 1. * beta(5., 1.).rvs(2)

        At += dA
        Bt += dB
        ct += dc

        def linf(x, u):
            return At @ x + Bt @ u + ct

        k1 = linf(x, _u)
        k2 = linf(x + 0.5 * self.dt * k1, _u)
        k3 = linf(x + 0.5 * self.dt * k2, _u)
        k4 = linf(x + self.dt * k3, _u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
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

    def cost(self, x, u, u_last, a):
        c = 0.

        if self.slew_rate:
            c += (u - u_last).T @ np.diag(self.uw) @ (u - u_last)
        else:
            c += u.T @ np.diag(self.uw) @ u

        if a:
            y = np.hstack((wrap_angle(x[0]), x[1])) if self.periodic else x
            J, j = self.features_jacobian(getval(y))
            z = J(getval(y)) @ y + j
            c += a * (z - self.g).T @ np.diag(self.gw) @ (z - self.g)

        return c

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

    def evolve(self, u, dist):
        # state-action dependent noise
        _sigma = self.noise(self.state, u)
        # evolve deterministic dynamics
        self.state = self.linearized(self.state, u, dist)
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
        self.gw = np.array([1e4, 1e4, 1e0])

    def features(self, x):
        return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])
