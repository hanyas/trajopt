import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


class Pendulum(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1

        self._dt = 0.01

        self._sigma = 1.e-8 * np.eye(self.dm_state)

        # g = [th, thd]
        self._g = np.array([2. * np.pi, 0.])
        self._gw = np.array([1.e0, 1.e-1])

        # x = [th, thd]
        self._xmax = np.array([np.inf, 8.0])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self._uw = np.array([1.e-3])
        self._umax = 2.5
        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

        self.state = None
        self.np_random = None

        self.seed()

        _high, _low = np.array([0., -8.0]), np.array([2. * np.pi, 8.0])
        self._x0 = self.np_random.uniform(low=_low, high=_high)
        self._sigma_0 = 1.e-8 * np.eye(self.dm_state)

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
        u = np.clip(u, -self._umax, self._umax)

        g, m, l, k = 10., 1., 1., 1.e-3

        def f(x, u):
            th, dth = x
            return np.hstack((dth, 3. * g / (2. * l) * np.sin(th) +
                              3. / (m * l ** 2) * (u - k * dth)))

        k1 = f(x, u)
        k2 = f(x + 0.5 * self.dt * k1, u)
        k3 = f(x + 0.5 * self.dt * k2, u)
        k4 = f(x + self.dt * k3, u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.clip(xn, -self._xmax, self._xmax)

        return xn

    def inverse_dynamics(self, x, u):
        _u = np.clip(u, -self._umax, self._umax)

        g, m, l, k = 10., 1., 1., 1.e-3

        def f(x, u):
            th, dth = x
            return np.hstack((dth, 3. * g / (2. * l) * np.sin(th) +
                              3. / (m * l ** 2) * (u - k * dth)))

        k1 = f(x, _u)
        k2 = f(x - 0.5 * self.dt * k1, _u)
        k3 = f(x - 0.5 * self.dt * k2, _u)
        k4 = f(x - self.dt * k3, _u)

        xn = x - self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
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

    def cost(self, x, u, a):
        if a:
            _J, _j = self.features_jacobian(getval(x))
            _x = _J(getval(x)) @ x + _j
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


class PendulumWithCartesianCost(Pendulum):

    def __init__(self):
        super(PendulumWithCartesianCost, self).__init__()

        # g = [cs_th, sn_th, dth]
        self._g = np.array([1., 0., 0.])
        self._gw = np.array([1.e1, 1.e-1, 1.e-1])

    def features(self, x):
        return np.array([np.cos(x[0]), np.sin(x[0]), x[1]])


class PendulumWithCartesianObservation(Pendulum):

    def __init__(self):
        super(PendulumWithCartesianObservation, self).__init__()
        self.dm_state = 3

        self._x0 = np.array([-1., 0., 0.])
        self._sigma_0 = 1.e-8 * np.eye(self.dm_state)

        self._sigma = 1.e-8 * np.eye(self.dm_state)

        # g = [cs_th, sn_th, dth]
        self._g = np.array([1., 0., 0.])
        self._gw = np.array([1.e1, 1.e-1, 1.e-1])

        # x = [cs_th, sn_th, dth]
        self._xmax = np.array([1., 1., 25.0])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

    def dynamics(self, x, u):
        u = np.clip(u, -self._umax, self._umax)

        g, m, l, k = 10., 1., 1., 1.e-3

        # transfer to th/thd space
        cth, sth, dth = x
        _x = np.hstack((np.arctan2(sth, cth), dth))

        def f(x, u):
            th, dth = x
            return np.hstack((dth, 3. * g / (2. * l) * np.sin(th) +
                              3. / (m * l ** 2) * (u - k * dth)))

        k1 = f(_x, u)
        k2 = f(_x + 0.5 * self.dt * k1, u)
        k3 = f(_x + 0.5 * self.dt * k2, u)
        k4 = f(_x + self.dt * k3, u)

        _xn = _x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.array([np.cos(_xn[0]), np.sin(_xn[0]), _xn[1]])

        xn = np.clip(xn, -self._xmax, self._xmax)
        return xn

    def inverse_dynamics(self, x, u):
        u = np.clip(u, -self._umax, self._umax)

        g, m, l, k = 10., 1., 1., 1.e-3

        # transfer to th/thd space
        sth, cth, dth = x
        _x = np.hstack((np.arctan2(sth, cth), dth))

        def f(x, u):
            th, dth = x
            return np.hstack((dth, -3. * g / (2. * l) * np.sin(th + np.pi) +
                              3. / (m * l ** 2) * (u - k * dth)))

        k1 = f(_x, u)
        k2 = f(_x - 0.5 * self.dt * k1, u)
        k3 = f(_x - 0.5 * self.dt * k2, u)
        k4 = f(_x - self.dt * k3, u)

        _xn = _x - self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        xn = np.array([np.cos(_xn[0]), np.sin(_xn[0]), _xn[1]])

        xn = np.clip(xn, -self._xmax, self._xmax)
        return xn

    def features(self, x):
        return x
