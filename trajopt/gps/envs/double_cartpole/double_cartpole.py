import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class _DoubleCartpoleBase:

    def __init__(self, dt, xw, uw, goal):
        self.nb_xdim = 6
        self.nb_udim = 1

        self._dt = dt
        self._g = goal

        self._xw = xw
        self._uw = uw

        # x = [x, th1, th2, dx, dth1, dth2]
        self.xmax = np.array([100., np.inf, np.inf, 25., 25., 25.])
        self.umax = 5.0

        self.sigma = 1.e-4 * np.eye(self.nb_xdim)

    def dynamics(self, x, u):
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

        action = np.vstack((u, 0.0, 0.0))

        M_inv = np.linalg.inv(M)
        C_x_dot = np.dot(C, x[3:].reshape((-1, 1)))
        x_dot_dot = np.dot(M_inv, action - C_x_dot - G).squeeze()

        x_dot = x[3:] + x_dot_dot * self._dt
        x_pos = x[:3] + x_dot * self._dt

        xn = np.hstack((x_pos, x_dot))
        return xn

    def reward(self, x, u, a):
        if a:
            return (x - self._g).T @ np.diag(self._xw) @ (x - self._g) + u.T @ np.diag(self._uw) @ u
        else:
            return u.T @ np.diag(self._uw) @ u

    def initialize(self):
        # mu, sigma
        return np.array([0., np.pi, 0., 0., 0., 0.]), 1.e-4 * np.eye(self.nb_xdim)


class DoubleCartpole(gym.Env):

    def __init__(self):
        self._dt = 0.01
        self._xw = - self._dt * 1. * np.array([1e-1, 1e1, 1e1, 1e-1, 1e-1, 1.e-1])
        self._uw = - self._dt * 1. * np.array([1.e-3])
        self._g = np.array([0., 2. * np.pi, 0., 0., 0., 0.])

        self._model = _DoubleCartpoleBase(self._dt, self._xw, self._uw, self._g)

        self.low_state = - self.model.xmax
        self.high_state = self.model.xmax

        self.action_space = spaces.Box(low=-self.model.umax,
                                       high=self.model.umax, shape=(1,))

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)

        self.seed()
        self.reset()

    @property
    def model(self):
        return self._model

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.state = self.model.dynamics(self.state, u)
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=self.model.sigma)
        return self.state, [], False, {}

    def reset(self):
        _mu_0, _sigma_0 = self._model.initialize()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return self.state
