#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: objects.py
# @Date: 2019-06-22-15-01
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
from autograd import jacobian, hessian


class GaussianInTime:
    def __init__(self, nb_dim, nb_steps):
        self.nb_dim = nb_dim
        self.nb_steps = nb_steps

        self.mu = np.zeros((self.nb_dim, self.nb_steps))
        self.sigma = np.zeros((self.nb_dim, self.nb_dim, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = np.eye(self.nb_dim)

    @property
    def params(self):
        return self.mu, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma = values

    def sample(self, x):
        pass


class QuadraticStateValue:
    def __init__(self, nb_xdim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_steps = nb_steps

        self.V = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.v = np.zeros((self.nb_xdim, self.nb_steps, ))
        self.v0 = np.zeros((self.nb_steps, ))
        self.v0_softmax = np.zeros((self.nb_steps, ))


class QuadraticStateActionValue:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        self.Qxx = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.Quu = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        self.Qux = np.zeros((self.nb_udim, self.nb_xdim, self.nb_steps))

        self.qx = np.zeros((self.nb_xdim, self.nb_steps, ))
        self.qu = np.zeros((self.nb_udim, self.nb_steps, ))

        self.q0 = np.zeros((self.nb_steps, ))
        self.q0_common = np.zeros((self.nb_steps, ))
        self.q0_softmax = np.zeros((self.nb_steps, ))


class QuadraticReward:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim

        self.nb_steps = nb_steps

        self.Rxx = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.rx = np.zeros((self.nb_xdim, self.nb_steps))

        self.Ruu = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        self.ru = np.zeros((self.nb_udim, self.nb_steps))

        self.Rxu = np.zeros((self.nb_xdim, self.nb_udim, self.nb_steps))
        self.r0 = np.zeros((self.nb_steps, ))

        # activation of reward function
        self.a = np.zeros((self.nb_steps, ), dtype=np.int64)
        self.a[-1] = 1

    @property
    def params(self):
        return self.Rxx, self.rx, self.Ruu, self.ru, self.Rxu, self.r0

    @params.setter
    def params(self, values):
        self.Rxx, self.rx, self.Ruu, self.ru, self.Rxu, self.r0 = values


class AnalyticalQuadraticReward(QuadraticReward):
    def __init__(self, f, nb_xdim, nb_udim, nb_steps):
        super(AnalyticalQuadraticReward, self).__init__(nb_xdim, nb_udim, nb_steps)

        self.f = f
        self.drdxx = hessian(self.f, 0)
        self.drduu = hessian(self.f, 1)
        self.drdxu = jacobian(jacobian(self.f, 0), 1)

        self.drdx = jacobian(self.f, 0)
        self.drdu = jacobian(self.f, 1)

    def eval(self, x, u, t, const=False):
        res = 0.0
        # quadratic state
        res += np.einsum('k,kh,h->', x, self.Rxx[..., t], x)
        # quadratic action
        res += np.einsum('k,kh,h->', u, self.Ruu[..., t], u)
        # quadratic cross
        res += np.einsum('k,kh,h->', x, self.Rxu[..., t], u)
        # linear state
        res += np.einsum('k,h->', x, self.rx[..., t])
        # linear action
        res += np.einsum('k,h->', u, self.ru[..., t])
        if const:
            # constant term
            res += self.r0
        return res

    def diff(self, x, u):
        _x = x
        _u = np.hstack((u, np.zeros((self.nb_udim, 1))))

        for t in range(self.nb_steps):
            _in = tuple([_x[..., t], _u[..., t], self.a[t]])
            if t == self.nb_steps - 1:
                self.Rxx[..., t] = 0.5 * self.drdxx(*_in)
                self.rx[..., t] = self.drdx(*_in) - self.drdxx(*_in) @ _x[..., t]
            else:
                self.Rxx[..., t] = 0.5 * self.drdxx(*_in)
                self.Ruu[..., t] = 0.5 * self.drduu(*_in)
                self.Rxu[..., t] = 0.5 * self.drdxu(*_in)

                self.rx[..., t] = self.drdx(*_in) - self.drdxx(*_in) @ _x[..., t] - 0.5 * self.drdxu(*_in) @ _u[..., t]
                self.ru[..., t] = self.drdu(*_in) - self.drduu(*_in) @ _u[..., t] - 0.5 * x[..., t].T @ self.drdxu(*_in)

            self.r0[..., t] = self.f(*_in) - self.eval(_x[..., t], _u[..., t], t)


class LinearGaussianDynamics:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        self.A = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.B = np.zeros((self.nb_xdim, self.nb_udim, self.nb_steps))
        self.c = np.zeros((self.nb_xdim, self.nb_steps))
        self.sigma = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = 1e-8 * np.eye(self.nb_xdim)

    @property
    def params(self):
        return self.A, self.B, self.c, self.sigma

    @params.setter
    def params(self, values):
        self.A, self.B, self.c, self.sigma = values

    def sample(self, x, u):
        pass


class AnalyticalLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, f, sigma, nb_xdim, nb_udim, nb_steps):
        super(AnalyticalLinearGaussianDynamics, self).__init__(nb_xdim, nb_udim, nb_steps)

        self.f = f
        self.dfdx = jacobian(self.f, 0)
        self.dfdu = jacobian(self.f, 1)

        self._sigma = sigma

    def eval(self, x, u, t, const=False):
        xn = np.zeros((self.nb_xdim, ))
        # linear state
        xn += np.einsum('h,kh->k', x, self.A[..., t])
        # linear action
        xn += np.einsum('h,kh->k', u, self.B[..., t])
        # constant
        if const:
            # constant term
            xn += self.c
        return xn

    def diff(self, x, u):
        for t in range(self.nb_steps):
            self.A[..., t] = self.dfdx(x[..., t], u[..., t])
            self.B[..., t] = self.dfdu(x[..., t], u[..., t])
            self.c[..., t] = self.f(x[..., t], u[..., t]) - self.eval(x[..., t], u[..., t], t)
            self.sigma[..., t] = self._sigma


class LinearGaussianControl:
    def __init__(self, nb_xdim, nb_udim, nb_steps, init_ctl_sigma=1.):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        self.K = np.zeros((self.nb_udim, self.nb_xdim, self.nb_steps))
        self.kff = np.zeros((self.nb_udim, self.nb_steps))

        self.sigma = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = init_ctl_sigma * np.eye(self.nb_udim)

    @property
    def params(self):
        return self.K, self.kff, self.sigma

    @params.setter
    def params(self, values):
        self.K, self.kff, self.sigma = values

    def mean(self, x, t):
        return np.einsum('kh,h->k', self.K[..., t], x) + self.kff[..., t]

    def sample(self, x, t, stoch=True):
        mu = self.mean(x, t)
        if stoch:
            return np.random.multivariate_normal(mean=mu, cov=self.sigma[..., t])
        else:
            return mu
