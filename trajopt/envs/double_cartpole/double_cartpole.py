import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


def angle_normalize(x):
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class DoubleCartpole(gym.Env):

    def __init__(self):
        self.dm_state = 6
        self.dm_act = 1

        self._dt = 0.01

        self._sigma = 1e-4 * np.eye(self.dm_state)

        # x = [x, th1, th2, dx, dth1, dth2]
        self._g = np.array([0., 2. * np.pi, 0., 0., 0., 0.])
        self._gw = np.array([1e-1, 1e1, 1e1, 1e-1, 1e-1, 1e-1])

        self._xmax = np.array([10., np.inf, np.inf, 25., 25., 25.])
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

        # import from: https://github.com/JoeMWatson/input-inference-for-control/
        """
        http://www.lirmm.fr/~chemori/Temp/Wafa/double%20pendule%20inverse.pdf
        """

        # x = [x, th1, th2, dx, dth1, dth2]

        g = 9.81
        Mc = 0.37
        Mp1 = 0.127
        Mp2 = 0.127
        Mt = Mc + Mp1 + Mp2
        L1 = 0.3365
        L2 = 0.3365
        l1 = L1 / 2
        l2 = L2 / 2
        J1 = Mp1 * L1 / 12
        J2 = Mp2 * L2 / 12

        q = x[0]
        th1 = x[1]
        th2 = x[2]
        q_dot = x[3]
        th_dot1 = x[4]
        th_dot2 = x[5]

        sth1 = np.sin(th1)
        cth1 = np.cos(th1)
        sth2 = np.sin(th2)
        cth2 = np.cos(th2)
        sdth = np.sin(th1 - th2)
        cdth = np.cos(th1 - th2)

        # helpers
        l1_mp1_mp2 = Mp1 * l1 + Mp2 * L2
        l1_mp1_mp2_cth1 = l1_mp1_mp2 * cth1
        Mp2_l2 = Mp2 * l2
        Mp2_l2_cth2 = Mp2_l2 * cth2
        l1_l2_Mp2 = L1 * l2 * Mp2
        l1_l2_Mp2_cdth = l1_l2_Mp2 * cdth

        # inertia
        M11 = Mt
        M12 = l1_mp1_mp2_cth1
        M13 = Mp2_l2_cth2
        M21 = l1_mp1_mp2_cth1
        M22 = (l1 ** 2) * Mp1 + (L1 ** 2) * Mp2 + J1
        M23 = l1_l2_Mp2_cdth
        M31 = Mp2_l2_cth2
        M32 = l1_l2_Mp2_cdth
        M33 = (l2 ** 2) * Mp2 + J2

        # coreolis
        C11 = 0.
        C12 = -l1_mp1_mp2 * th_dot1 * sth1
        C13 = -Mp2_l2 * th_dot2 * sth2
        C21 = 0.
        C22 = 0.
        C23 = l1_l2_Mp2 * th_dot2 * sdth
        C31 = 0.
        C32 = -l1_l2_Mp2 * th_dot1 * sdth
        C33 = 0.

        # gravity
        G11 = 0.
        G21 = - (Mp1 * l1 + Mp2 * L1) * g * sth1
        G31 = - Mp2 * l2 * g * sth2

        # make matrices
        M = np.vstack((np.hstack((M11, M12, M13)), np.hstack((M21, M22, M23)), np.hstack((M31, M32, M33))))

        C = np.vstack((np.hstack((C11, C12, C13)), np.hstack((C21, C22, C23)), np.hstack((C31, C32, C33))))

        G = np.vstack((G11, G21, G31))

        action = np.vstack((_u, 0.0, 0.0))

        M_inv = np.linalg.inv(M)
        C_x_dot = np.dot(C, x[3:].reshape((-1, 1)))
        x_dot_dot = np.dot(M_inv, action - C_x_dot - G).squeeze()

        x_dot = x[3:] + x_dot_dot * self._dt
        x_pos = x[:3] + x_dot * self._dt

        xn = np.hstack((x_pos, x_dot))

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
        _low, _high = np.array([0., np.pi - np.pi / 18., np.pi - np.pi / 18., 0., -0.1, -0.1]),\
                      np.array([0., np.pi + np.pi / 18., np.pi + np.pi / 18., 0., 0.1, 0.1])
        _x0 = self.np_random.uniform(low=_low, high=_high)
        self.state = np.hstack((_x0[0],
                                angle_normalize(_x0[1]),
                                angle_normalize(_x0[2]),
                                _x0[3], _x0[4], _x0[5]))
        return self.state


class DoubleCartpoleWithCartesianCost(DoubleCartpole):

    def __init__(self):
        super(DoubleCartpoleWithCartesianCost, self).__init__()

        # g = [x, cs_th1, sn_th1, cs_th2, sn_th2, dx, dth1, dth2]
        self._g = np.array([0.,
                            1., 0.,
                            0., 0.,
                            0., 0., 0.])

        self._gw = np.array([1e-1,
                             1e1, 1e-1,
                             1e1, 1e-1,
                             1e-1, 1e-1, 1e-1])

    def features(self, x):
        return np.array([x[0],
                         np.cos(x[0]), np.sin(x[0]),
                         np.cos(x[1]), np.sin(x[1]),
                         x[2], x[3], x[4]])
