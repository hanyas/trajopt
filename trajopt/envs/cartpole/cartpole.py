import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian


class Cartpole(gym.Env):

    def __init__(self):
        self.nb_xdim = 4
        self.nb_udim = 1

        self._dt = 0.01

        self._x0 = np.array([0., np.pi, 0., 0.])
        self._sigma_0 = 1.e-4 * np.eye(self.nb_xdim)

        self._sigma = 1.e-4 * np.eye(self.nb_xdim)

        # g = [x, th, dx, dth]
        self._g = np.array([0., 2. * np.pi, 0., 0.])
        self._gw = np.array([1.e-1, 1.e1, 1.e-1, 1.e-1])

        # x = [x, th, dx, dth]
        self._xmax = np.array([0., np.inf, 25., 25.])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self._uw = np.array([1.e-3])
        self._umax = 5.0
        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

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

    def init(self):
        # mu, sigma
        return self._x0, self._sigma_0

    def dynamics(self, x, u):
        _u = np.clip(u, -self._umax, self._umax)

        # import from: https://github.com/JoeMWatson/input-inference-for-control/
        # x = [x, th, dx, dth]
        g = 9.81
        Mc = 0.37
        Mp = 0.127
        Mt = Mc + Mp
        l = 0.3365

        th = x[1]
        dth2 = np.power(x[3], 2)
        sth = np.sin(th)
        cth = np.cos(th)

        _num = - Mp * l * sth * dth2 + Mt * g * sth - _u * cth
        _denom = l * ((4. / 3.) * Mt - Mp * cth ** 2)
        th_acc = _num / _denom
        x_acc = (Mp * l * sth * dth2 - Mp * l * th_acc * cth + _u) / Mt

        xn = np.hstack((x[0] + self._dt * (x[2] + self._dt * x_acc),
                       x[1] + self._dt * (x[3] + self._dt * th_acc),
                       x[2] + self._dt * x_acc,
                       x[3] + self._dt * th_acc))

        xn = np.clip(xn, -self._xmax, self._xmax)
        return xn

    def features(self, x):
        return x

    def features_jacobian(self, x):
        _J = jacobian(self.features, 0)
        _j = self.features(x) - _J(x) @ x
        return _J, _j

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self._umax, self._umax)
        _x = np.clip(x, -self._xmax, self._xmax)
        return self._sigma

    # xref is a hack to avoid autograd diffing through the jacobian
    def cost(self, x, u, a, xref):
        if a:
            _J, _j = self.features_jacobian(xref)
            _x = _J(xref) @ x + _j
            return (_x - self._g).T @ np.diag(self._gw) @ (_x - self._g) + u.T @ np.diag(self._uw) @ u
        else:
            return u.T @ np.diag(self._uw) @ u

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
        _mu_0, _sigma_0 = self.init()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return self.state


class CartpoleWithCartesianCost(Cartpole):

    def __init__(self):
        super(CartpoleWithCartesianCost, self).__init__()

        # g = [x, cs_th, sn_th, dx, dth]
        self._g = np.array([1e-1, 1., 0., 0., 0.])
        self._gw = np.array([1.e-1, 1.e1, 1.e-1, 1.e-1, 1.e-1])

    def features(self, x):
        return np.array([x[0],
                        np.cos(x[1]), np.sin(x[1]),
                        x[2], x[3]])
