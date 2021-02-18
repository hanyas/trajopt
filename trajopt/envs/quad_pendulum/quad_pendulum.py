import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval


def wrap_angle(x):
    # wraps angle between [-pi, pi]
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class QuadPendulum(gym.Env):

    def __init__(self):
        self.dm_state = 8
        self.dm_act = 4

        self.dt = 0.01

        self.sigma = 1e-8 * np.eye(self.dm_state)

        # g = [th1, th2, th3, th4, dth1, dth2, dth3, dth4]
        self.g = np.array([0., 0., 0., 0.,
                           0., 0., 0., 0.])
        self.gw = np.array([1e4, 1e4, 1e4, 1e4,
                            1e0, 1e0, 1e0, 1e0])

        # x = [th, dth]
        self.xmax = np.array([np.inf, np.inf, np.inf, np.inf,
                              np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax)

        self.slew_rate = False
        self.uw = 1e-5 * np.ones((self.dm_act, ))
        self.umax = 25. * np.ones((self.dm_act, ))
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(4,))

        self.x0 = np.array([np.pi, 0., 0., 0.,
                            0., 0., 0., 0.])
        self.sigma0 = 1e-4 * np.eye(self.dm_state)

        self.periodic = False

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
        # Code from Chris Atkeson's (http://www.cs.cmu.edu/~cga/kdc/dynamics-2d/dynamics4.c)

        masses = np.array([1., 1., 1., 1.])
        lengths = np.array([1., 1., 1., 1.])
        friction = 0.025 * np.array([1., 1., 1., 1.])
        G = 9.81

        inertias = masses * (lengths ** 2 + 1e-4) / 3.0

        def f(x, u):
            th1, th2, th3, th4,\
            dth1, dth2, dth3, dth4 = x
            th1 = th1 + np.pi  # downward position = PI

            u1, u2, u3, u4 = u

            I1, I2, I3, I4 = inertias
            l1, l2, l3, l4 = lengths
            m1, m2, m3, m4 = masses
            fr1, fr2, fr3, fr4 = friction

            l1cm = l1 / 2.
            l2cm = l2 / 2.
            l3cm = l3 / 2.
            l4cm = l4 / 2.

            s1, c1 = np.sin(th1), np.cos(th1)
            s2, c2 = np.sin(th2), np.cos(th2)
            s3, c3 = np.sin(th3), np.cos(th3)
            s4, c4 = np.sin(th4), np.cos(th4)

            s12 = s1 * c2 + c1 * s2
            c12 = c1 * c2 - s1 * s2
            s23 = s2 * c3 + c2 * s3
            c23 = c2 * c3 - s2 * s3
            s34 = s3 * c4 + c3 * s4
            c34 = c3 * c4 - s3 * s4

            s1234 = s12 * c34 + c12 * s34
            s123 = s12 * c3 + c12 * s3
            s234 = s2 * c34 + c2 * s34
            c234 = c2 * c34 - s2 * s34

            dth1_dth1 = dth1 * dth1
            dth2_dth2 = dth2 * dth2
            dth3_dth3 = dth3 * dth3
            dth4_dth4 = dth4 * dth4
            dth1_p_dth2_2 = (dth1 + dth2) * (dth1 + dth2)

            l4cm_m4 = l4cm * m4
            l3_l4cm_m4 = l3 * l4cm_m4
            l2_l4cm_m4 = l2 * l4cm_m4
            l2_l4cm_m4_c34 = l2_l4cm_m4 * c34
            l1_l4cm_m4 = l1 * l4cm_m4
            l3_m4 = l3 * m4
            l3cm_m3 = l3cm * m3
            l3cm_m3_l3_m4 = l3cm_m3 + l3_m4
            l2cm_m2 = l2cm * m2
            l2cm_m2_p_l2_m3_p_m4 = l2cm_m2 + l2 * (m3 + m4)
            l2_l3cm_m3_l3_m4 = l2 * l3cm_m3_l3_m4
            l1_l3cm_m3_l3_m4 = l1 * l3cm_m3_l3_m4
            a123d = dth1 + dth2 + dth3
            l1_l3cm_m3_l3_m4_s23 = l1_l3cm_m3_l3_m4 * s23
            l2_l4cm_m4_s34 = l2_l4cm_m4 * s34

            expr1 = G * (s123 * l3cm_m3_l3_m4 + s1234 * l4cm_m4)
            expr2 = (2 * a123d + dth4) * dth4 * l3_l4cm_m4 * s4
            expr3 = G * l2cm_m2_p_l2_m3_p_m4 * s12
            expr4a = 2 * dth1 * dth4 + 2 * dth2 * dth4 + 2 * dth3 * dth4 + dth4_dth4
            expr4b = 2 * dth1 * dth3 + 2 * dth2 * dth3 + dth3_dth3
            expr4 = (expr4b + expr4a) * l2_l4cm_m4_s34
            expr5a = dth1_dth1 * l1 * s234
            expr5 = l4cm_m4 * expr5a
            expr6 = expr4b * l2_l3cm_m3_l3_m4 * s3
            expr7 = l1 * l2cm_m2_p_l2_m3_p_m4
            expr8 = l1 * (m2 + m3 + m4)
            expr9a = 2 * dth1 * dth2 + dth2_dth2
            expr9 = (expr9a + expr4b)

            p = I4 + l4cm * l4cm_m4
            o = p + l3_l4cm_m4 * c4
            n = o + l2_l4cm_m4_c34
            m = n + l1_l4cm_m4 * c234

            t = u4 - fr4 * dth4 \
                - (l4cm_m4 * (a123d * a123d * l3 * s4 +
                              dth1_p_dth2_2 * l2 * s34 +
                              expr5a + G * s1234))

            l = o
            k = I3 + o + l3cm * l3cm_m3 + l3 * l3_m4 + l3_l4cm_m4 * c4
            j = k + l2_l3cm_m3_l3_m4 * c3 + l2_l4cm_m4_c34
            i = j + l1_l3cm_m3_l3_m4 * c23 + l1_l4cm_m4 * c234

            s = u3 - fr3 * dth3 \
                - ((dth1_p_dth2_2 * l2_l3cm_m3_l3_m4 * s3 + dth1_dth1 * l1_l3cm_m3_l3_m4_s23)
                    - expr2 + dth1_p_dth2_2 * l2_l4cm_m4_s34 + expr5 + expr1)

            h = n
            g = j

            f = j + I2 + l2cm * l2cm_m2 + (l2 * l2) * (m3 + m4)\
                + l2_l3cm_m3_l3_m4 * c3 + l2_l4cm_m4_c34

            e = f + i - j + expr7 * c2

            r = u2 - fr2 * dth2\
                - (dth1_dth1 * expr7 * s2
                   - expr6 + dth1_dth1 * l1_l3cm_m3_l3_m4_s23
                   - expr2 - expr4 + expr5 + expr3 + expr1)

            d = m
            c = i
            b = e
            a = 2 * e + I1 - f + (l1cm * l1cm) * m1 + l1 * expr8

            q = u1 - fr1 * dth1\
                - (- expr9a * expr7 * s2 - expr6 - expr9 * l1_l3cm_m3_l3_m4_s23
                   - expr2 - expr4 - (expr9 + expr4a) * l1_l4cm_m4 * s234
                   + expr3 + G * (l1cm * m1 + expr8) * s1 + expr1)

            det = (d * g * j * m - c * h * j * m - d * f * k * m + b * h * k * m + c * f * l * m - b * g * l * m - d * g * i * n +
                   c * h * i * n + d * e * k * n - a * h * k * n - c * e * l * n + a * g * l * n + d * f * i * o - b * h * i * o -
                   d * e * j * o + a * h * j * o + b * e * l * o - a * f * l * o - c * f * i * p + b * g * i * p + c * e * j * p -
                   a * g * j * p - b * e * k * p + a * f * k * p)

            ddth1 = q * (-(h * k * n) + g * l * n + h * j * o - f * l * o - g * j * p + f * k * p)\
                    + r * (d * k * n - c * l * n - d * j * o + b * l * o + c * j * p - b * k * p)\
                    + s * (-(d * g * n) + c * h * n + d * f * o - b * h * o - c * f * p + b * g * p)\
                    + t * (d * g * j - c * h * j - d * f * k + b * h * k + c * f * l - b * g * l)

            ddth2 = q * (h * k * m - g * l * m - h * i * o + e * l * o + g * i * p - e * k * p)\
                    + r * (-(d * k * m) + c * l * m + d * i * o - a * l * o - c * i * p + a * k * p)\
                    + s * (d * g * m - c * h * m - d * e * o + a * h * o + c * e * p - a * g * p)\
                    + t * (-(d * g * i) + c * h * i + d * e * k - a * h * k - c * e * l + a * g * l)

            ddth3 = q * (-(h * j * m) + f * l * m + h * i * n - e * l * n - f * i * p + e * j * p)\
                    + r * (d * j * m - b * l * m - d * i * n + a * l * n + b * i * p - a * j * p)\
                    + s * (-(d * f * m) + b * h * m + d * e * n - a * h * n - b * e * p + a * f * p)\
                    + t * (d * f * i - b * h * i - d * e * j + a * h * j + b * e * l - a * f * l)

            ddth4 = q * (g * j * m - f * k * m - g * i * n + e * k * n + f * i * o - e * j * o)\
                    + r * (-(c * j * m) + b * k * m + c * i * n - a * k * n - b * i * o + a * j * o)\
                    + s * (c * f * m - b * g * m - c * e * n + a * g * n + b * e * o - a * f * o)\
                    + t * (-(c * f * i) + b * g * i + c * e * j - a * g * j - b * e * k + a * f * k)

            ddth1 = ddth1 / det
            ddth2 = ddth2 / det
            ddth3 = ddth3 / det
            ddth4 = ddth4 / det

            return np.array([dth1, dth2, dth3, dth4,
                             ddth1, ddth2, ddth3, ddth4])

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
            y = np.hstack((wrap_angle(x[0]), wrap_angle(x[1]),
                           wrap_angle(x[2]), wrap_angle(x[3]),
                           x[4], x[5], x[6], x[7])) if self.periodic else x
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


class QuadPendulumWithCartesianCost(QuadPendulum):

    def __init__(self):
        super(QuadPendulumWithCartesianCost, self).__init__()

        self.g = np.array([1., 0.,
                           1., 0.,
                           1., 0.,
                           1., 0.,
                           0., 0., 0., 0.])
        self.gw = np.array([1e4, 1e4,
                            1e4, 1e4,
                            1e4, 1e4,
                            1e4, 1e4,
                            1e0, 1e0, 1e0, 1e0])

    def features(self, x):
        return np.array([np.cos(x[0]), np.sin(x[0]),
                         np.cos(x[1]), np.sin(x[1]),
                         np.cos(x[2]), np.sin(x[2]),
                         np.cos(x[3]), np.sin(x[3]),
                         x[4], x[5], x[6], x[7]])
