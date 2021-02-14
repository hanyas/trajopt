import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


def wrap_angle(x):
    # wraps angle between [-pi, pi]
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class DoublePendulum(gym.Env):

    def __init__(self):
        self.dm_state = 4
        self.dm_act = 2

        self.dt = 0.01

        self.sigma = 1e-8 * np.eye(self.dm_state)

        # g = [th1, th2, dth1, dth2]
        self.g = np.array([0., 0.,
                           0., 0.])
        self.gw = np.array([1e4, 1e4,
                            1e0, 1e0])

        # x = [th, dth]
        self.xmax = np.array([np.inf, np.inf,
                              np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.slew_rate = False
        self.uw = 1e-5 * np.ones((self.dm_act, ))
        self.umax = 10. * np.ones((self.dm_act, ))
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(2,))

        self.x0 = np.array([np.pi, 0.,
                            0., 0.])
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

        # Code from PolicySearchToolbox

        masses = np.array([1., 1.])
        lengths = np.array([1., 1.])
        friction = 0.025 * np.array([1., 1.])
        g = 9.81

        inertias = masses * (lengths ** 2 + 1e-4) / 3.0

        def f(x, u):
            th1, th2, dth1, dth2 = x
            th1 = th1 + np.pi  # downward position = PI

            u1, u2 = u

            I1, I2 = inertias
            l1, l2 = lengths
            m1, m2 = masses
            k1, k2 = friction

            l1CM = l1 / 2.
            l2CM = l2 / 2.

            s1, c1 = np.sin(th1), np.cos(th1)
            s2, c2 = np.sin(th2), np.cos(th2)

            h11 = I1 + I2 + l1CM * l1CM * m1 + l1 * l1 * m2\
                  + l2CM * l2CM * m2 + 2. * l1 * l2CM * m2 * c2
            h12 = I2 + l2CM * l2CM * m2 + l1 * l2CM * m2 * c2

            b1 = g * l1CM * m1 * s1 + g * l1 * m2 * s1\
                 + g * l2CM * m2 * c2 * s1\
                 - 2. * dth1 * dth2 * l1 * l2CM * m2 * s2\
                 - dth2 * dth2 * l1 * l2CM * m2 * s2\
                 + g * l2CM * m2 * c1 * s2

            h21 = I2 + l2CM * l2CM * m2 + l1 * l2CM * m2 * c2
            h22 = I2 + l2CM * l2CM * m2

            b2 = g * l2CM * m2 * c2 * s1\
                 + dth1 * dth1 * l1 * l2CM * m2 * s2\
                 + g * l2CM * m2 * c1 * s2

            u1 = u1 - k1 * dth1
            u2 = u2 - k2 * dth2

            det = h11 * h22 - h12 * h21

            ddth1 = (h22 * (u1 - b1) - h12 * (u2 - b2)) / det
            ddth2 = (h11 * (u2 - b2) - h21 * (u1 - b1)) / det

            return np.array([dth1, dth2, ddth1, ddth2])

        k1 = f(x, _u)
        k2 = f(x + 0.5 * self.dt * k1, _u)
        k3 = f(x + 0.5 * self.dt * k2, _u)
        k4 = f(x + self.dt * k3, _u)

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
            y = x
            # y = np.hstack((wrap_angle(x[0]), x[1]))
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

    def init(self):
        return self.x0, self.sigma0

    def reset(self):
        self.state = self.np_random.multivariate_normal(mean=self.x0, cov=self.sigma0)
        return self.state


class DoublePendulumWithCartesianCost(DoublePendulum):

    def __init__(self):
        super(DoublePendulumWithCartesianCost, self).__init__()

        # g = [cs_th, sn_th, dth]
        self.g = np.array([1., 0.,
                           1., 0.,
                           0., 0.])
        self.gw = np.array([1e4, 1e4,
                            1e4, 1e4,
                            1e0, 1e0])

    def features(self, x):
        return np.array([np.cos(x[0]), np.sin(x[0]),
                         np.cos(x[1]), np.sin(x[1]),
                         x[2], x[3]])
