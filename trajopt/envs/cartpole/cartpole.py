import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


def angle_normalize(x):
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Cartpole(gym.Env):

    def __init__(self):
        self.dm_state = 4
        self.dm_act = 1

        self._dt = 0.01

        self._sigma = 1e-4 * np.eye(self.dm_state)

        # g = [x, th, dx, dth]
        self._g = np.array([0., 0., 0., 0.])
        self._gw = np.array([1e-1, 1e2, 1e0, 1e0])

        # x = [x, th, dx, dth]
        self._xmax = np.array([10., np.inf, 5., 10.])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self._uw = np.array([1e-3])
        self._umax = 5.0
        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

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

        xn = np.hstack((xn[0], angle_normalize(xn[1]), xn[2], xn[3]))
        xn = np.clip(xn, -self.xlim, self.xlim)
        return xn

    def features(self, x):
        return x

    def features_jacobian(self, x):
        _J = jacobian(self.features, 0)
        _j = self.features(x) - _J(x) @ x
        return _J, _j

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self.ulim, self.ulim)
        _x = np.clip(x, -self.xlim, self.xlim)
        return self._sigma

    def cost(self, x, u, u_lst):
        _J, _j = self.features_jacobian(getval(x))
        _x = _J(getval(x)) @ x + _j
        return self.dt * ((_x - self._g).T @ np.diag(self._gw) @ (_x - self._g)
                          + (u - u_lst).T @ np.diag(self._uw) @ (u - u_lst))

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
        _low, _high = np.array([-0.1, np.pi - np.pi / 18., -0.1, -0.1]),\
                      np.array([0.1, np.pi + np.pi / 18., 0.1, 0.1])
        _x0 = self.np_random.uniform(low=_low, high=_high)
        self.state = np.hstack((_x0[0], angle_normalize(_x0[1]), _x0[2], _x0[3]))
        return self.state


class CartpoleWithCartesianCost(Cartpole):

    def __init__(self):
        super(CartpoleWithCartesianCost, self).__init__()

        # g = [x, cs_th, sn_th, dx, dth]
        self._g = np.array([0., 1., 0., 0., 0.])
        self._gw = np.array([1e-1, 1e2, 1e2, 1e0, 1e0])

    def features(self, x):
        return np.array([x[0],
                        np.cos(x[1]), np.sin(x[1]),
                        x[2], x[3]])
