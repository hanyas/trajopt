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


class QuadraticCost:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim

        self.nb_steps = nb_steps

        self.Cxx = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.cx = np.zeros((self.nb_xdim, self.nb_steps))

        self.Cuu = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        self.cu = np.zeros((self.nb_udim, self.nb_steps))

        self.Cxu = np.zeros((self.nb_xdim, self.nb_udim, self.nb_steps))

    @property
    def params(self):
        return self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu

    @params.setter
    def params(self, values):
        self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu = values


class AnalyticalQuadraticCost(QuadraticCost):
    def __init__(self, f, nb_xdim, nb_udim, nb_steps):
        super(AnalyticalQuadraticCost, self).__init__(nb_xdim, nb_udim, nb_steps)

        self.f = f
        self.dcdxx = hessian(self.f, 0)
        self.dcduu = hessian(self.f, 1)
        self.dcdxu = jacobian(jacobian(self.f, 0), 1)

        self.dcdx = jacobian(self.f, 0)
        self.dcdu = jacobian(self.f, 1)

    def finite_diff(self, x, u, a):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.nb_udim, 1))))

        for t in range(self.nb_steps):
            _in = tuple([x[..., t], _u[..., t], a[t]])
            if t == self.nb_steps - 1:
                self.Cxx[..., t] = self.dcdxx(*_in)
                self.cx[..., t] = self.dcdx(*_in) - self.dcdxx(*_in) @ x[..., t]
            else:
                self.Cxx[..., t] = self.dcdxx(*_in)
                self.Cuu[..., t] = self.dcduu(*_in)
                self.Cxu[..., t] = self.dcdxu(*_in)

                self.cx[..., t] = self.dcdx(*_in) - self.dcdxx(*_in) @ x[..., t] - self.dcdxu(*_in) @ _u[..., t]
                self.cu[..., t] = self.dcdu(*_in) - self.dcduu(*_in) @ _u[..., t] - x[..., t].T @ self.dcdxu(*_in)


class LinearDynamics:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        self.A = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.B = np.zeros((self.nb_xdim, self.nb_udim, self.nb_steps))

    @property
    def params(self):
        return self.A, self.B

    @params.setter
    def params(self, values):
        self.A, self.B = values

    def sample(self, x, u):
        pass


class AnalyticalLinearDynamics(LinearDynamics):
    def __init__(self, f, nb_xdim, nb_udim, nb_steps):
        super(AnalyticalLinearDynamics, self).__init__(nb_xdim, nb_udim, nb_steps)

        self.f = f
        self.dfdx = jacobian(self.f, 0)
        self.dfdu = jacobian(self.f, 1)

    def finite_diff(self, x, u):
        for t in range(self.nb_steps):
            self.A[..., t] = self.dfdx(x[..., t], u[..., t])
            self.B[..., t] = self.dfdu(x[..., t], u[..., t])


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

    def action(self, x, alpha, xref, uref, t):
        dx = x - xref[:, t]
        return uref[:, t] + alpha * self.kff[..., t] + self.K[..., t] @ dx
