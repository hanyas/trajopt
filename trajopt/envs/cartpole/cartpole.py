import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


def wrap_angle(x):
    # wraps angle between [-pi, pi]
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Cartpole(gym.Env):

    def __init__(self):
        self.dm_state = 4
        self.dm_act = 1

        self.dt = 0.01

        self.sigma = 1e-8 * np.eye(self.dm_state)

        # g = [x, th, dx, dth]
        self.g = np.array([0., 0., 0., 0.])
        self.gw = np.array([1e-1, 1e2, 1e0, 1e0])

        # x = [x, th, dx, dth]
        self.xmax = np.array([10., np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.uw = np.array([1e-5])
        self.umax = 5.0
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(1,))

        self.x0 = np.array([0., np.pi, 0., 0.])
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

        # Equations: http://coneural.org/florian/papers/05_cart_pole.pdf
        # x = [x, th, dx, dth]
        g = 9.81
        Mc = 0.37
        Mp = 0.127
        Mt = Mc + Mp
        l = 0.3365

        def f(x, u):
            th = x[1]
            dth2 = np.power(x[3], 2)
            sth = np.sin(th)
            cth = np.cos(th)

            _num = g * sth + cth * (- u - Mp * l * dth2 * sth) / Mt
            _denom = l * ((4. / 3.) - Mp * cth**2 / Mt)
            th_acc = _num / _denom

            x_acc = (u + Mp * l * (dth2 * sth - th_acc * cth)) / Mt

            return np.hstack((x[2], x[3], x_acc, th_acc))

        c1 = f(x, _u)
        c2 = f(x + 0.5 * self.dt * c1, _u)
        c3 = f(x + 0.5 * self.dt * c2, _u)
        c4 = f(x + self.dt * c3, _u)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)
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
            x = np.hstack((x[0], wrap_angle(x[1]),
                           x[2], x[3]))
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


class CartpoleWithCartesianCost(Cartpole):

    def __init__(self):
        super(CartpoleWithCartesianCost, self).__init__()

        # g = [x, cs_th, sn_th, dx, dth]
        self.g = np.array([0., 1., 0., 0., 0.])
        self.gw = np.array([1e-1, 1e2, 1e2, 1e0, 1e0])

    def features(self, x):
        return np.array([x[0],
                        np.cos(x[1]), np.sin(x[1]),
                        x[2], x[3]])
