#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: objects.py
# @Date: 2019-06-22-15-01
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
from autograd import jacobian, hessian


class QuadraticStateValue:
    def __init__(self, nb_xdim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_steps = nb_steps

        self.V = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.v = np.zeros((self.nb_xdim, self.nb_steps, ))


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


class QuadraticReward:
    def __init__(self, nb_xdim, nb_udim, nb_steps, activation):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim

        self.nb_steps = nb_steps

        self.Rxx = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.rx = np.zeros((self.nb_xdim, self.nb_steps))

        self.Ruu = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        self.ru = np.zeros((self.nb_udim, self.nb_steps))

        self.Rxu = np.zeros((self.nb_xdim, self.nb_udim, self.nb_steps))

        # activation of reward function
        self.a = activation

    @property
    def params(self):
        return self.Rxx, self.rx, self.Ruu, self.ru, self.Rxu

    @params.setter
    def params(self, values):
        self.Rxx, self.rx, self.Ruu, self.ru, self.Rxu = values


class AnalyticalQuadraticReward(QuadraticReward):
    def __init__(self, f, nb_xdim, nb_udim, nb_steps, activation):
        super(AnalyticalQuadraticReward, self).__init__(nb_xdim, nb_udim, nb_steps, activation)

        self.f = f
        self.drdxx = hessian(self.f, 0)
        self.drduu = hessian(self.f, 1)
        self.drdxu = jacobian(jacobian(self.f, 0), 1)

        self.drdx = jacobian(self.f, 0)
        self.drdu = jacobian(self.f, 1)

    def exact_eval(self, x, u):
        ret = 0.0
        for t in range(self.nb_steps):
            ret += self.f(x[..., t], u[..., t], self.a[..., t])
        return ret

    def diff(self, x, u):
        _x = x
        _u = np.hstack((u, np.zeros((self.nb_udim, 1))))

        for t in range(self.nb_steps):
            _in = tuple([_x[..., t], _u[..., t], self.a[t]])
            if t == self.nb_steps - 1:
                self.Rxx[..., t] = self.drdxx(*_in)
                self.rx[..., t] = self.drdx(*_in) - self.drdxx(*_in) @ _x[..., t]
            else:
                self.Rxx[..., t] = self.drdxx(*_in)
                self.Ruu[..., t] = self.drduu(*_in)
                self.Rxu[..., t] = self.drdxu(*_in)

                self.rx[..., t] = self.drdx(*_in) - self.drdxx(*_in) @ _x[..., t] -\
                                  self.drdxu(*_in) @ _u[..., t]
                self.ru[..., t] = self.drdu(*_in) - self.drduu(*_in) @ _u[..., t] -\
                                  x[..., t].T @ self.drdxu(*_in)


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

    def approx_eval(self, x, u, t, const=False):
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
            self.c[..., t] = self.f(x[..., t], u[..., t]) -\
                             self.approx_eval(x[..., t], u[..., t], t)
            self.sigma[..., t] = self._sigma


class LearnedLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        super(LearnedLinearGaussianDynamics, self).__init__(nb_xdim, nb_udim, nb_steps)

    def learn(self, data, pointwise=False):
        if pointwise:
            from mimo import distributions
            _hypparams = dict(M=np.zeros((self.nb_xdim, self.nb_xdim + self.nb_udim + 1)),
                              V=1.e6 * np.eye(self.nb_xdim + self.nb_udim + 1),
                              affine=True,
                              psi=np.eye(self.nb_xdim), nu=self.nb_xdim + 2)
            _prior = distributions.MatrixNormalInverseWishart(**_hypparams)

            for t in range(self.nb_steps):
                _data = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T, data['xn'][:, t, :].T))

                _model = distributions.BayesianLinearGaussian(_prior)
                _model = _model.MAP(_data)

                self.A[..., t] = _model.A[:, :self.nb_xdim]
                self.B[..., t] = _model.A[:, self.nb_xdim:self.nb_xdim + self.nb_udim]
                self.c[..., t] = _model.A[:, -1]
                self.sigma[..., t] = _model.sigma
        else:
            _obs = [data['x'][..., n].T for n in range(data['x'].shape[-1])]
            _input = [data['u'][..., n].T for n in range(data['u'].shape[-1])]

            from sds.rarhmm_ls import rARHMM
            rarhmm = rARHMM(nb_states=5, dim_obs=self.nb_xdim, dim_act=self.nb_udim)
            rarhmm.initialize(_obs, _input)
            rarhmm.em(_obs, _input, nb_iter=50, prec=1e-12, verbose=False)

            _mean_obs = np.mean(data['x'], axis=-1).T
            _mean_input = np.mean(data['u'], axis=-1).T
            _, _mean_z = rarhmm.viterbi([_mean_obs], [_mean_input])

            for t in range(self.nb_steps):
                self.A[..., t] = rarhmm.observations.A[_mean_z[0][t], ...]
                self.B[..., t] = rarhmm.observations.B[_mean_z[0][t], ...]
                self.c[..., t] = rarhmm.observations.c[_mean_z[0][t], ...]


class LinearControl:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        self.K = np.zeros((self.nb_udim, self.nb_xdim, self.nb_steps))
        self.kff = np.zeros((self.nb_udim, self.nb_steps))

    @property
    def params(self):
        return self.K, self.kff

    @params.setter
    def params(self, values):
        self.K, self.kff = values

    def apply(self, x, alpha, xref, uref, t):
        dx = x - xref[:, t]
        return uref[:, t] + alpha * self.kff[..., t] + self.K[..., t] @ dx
