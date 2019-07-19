#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: objects.py
# @Date: 2019-06-22-15-01
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.misc import flatten


class Gaussian:
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


class QuadraticBeliefValue:
    def __init__(self, dm_belief, nb_steps):
        self.dm_belief = dm_belief
        self.nb_steps = nb_steps

        self.S = np.zeros((self.dm_belief, self.dm_belief, self.nb_steps))
        self.s = np.zeros((self.dm_belief, self.nb_steps, ))
        self.tau = np.zeros((self.dm_belief, self.nb_steps, ))


class QuadraticCost:
    def __init__(self, dm_belief, dm_act, nb_steps):
        self.dm_belief = dm_belief
        self.dm_act = dm_act

        self.nb_steps = nb_steps

        self.Q = np.zeros((self.dm_belief, self.dm_belief, self.nb_steps))
        self.q = np.zeros((self.dm_belief, self.nb_steps))

        self.R = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        self.r = np.zeros((self.dm_act, self.nb_steps))

        self.P = np.zeros((self.dm_belief, self.dm_act, self.nb_steps))
        self.p = np.zeros((self.dm_belief * self.dm_belief, self.nb_steps))

    @property
    def params(self):
        return self.Q, self.q, self.R, self.r, self.P, self.p

    @params.setter
    def params(self, values):
        self.Q, self.q, self.R, self.r, self.P, self.p = values


class AnalyticalQuadraticCost(QuadraticCost):
    def __init__(self, f_cost, dm_belief, dm_act, nb_steps):
        super(AnalyticalQuadraticCost, self).__init__(dm_belief, dm_act, nb_steps)

        self.f = f_cost

        self.fQ = hessian(self.f, 0)
        self.fq = jacobian(self.f, 0)

        self.fR = hessian(self.f, 2)
        self.fr = jacobian(self.f, 2)

        self.fP = jacobian(jacobian(self.f, 0), 2)
        self.fp = jacobian(self.f, 1)

    def evalf(self, mu_b, sigma_b, u, a):
        return self.f(mu_b, sigma_b, u, a)

    def taylor_expansion(self, b, u, a):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))

        for t in range(self.nb_steps):
            _in = tuple([b.mu[..., t], b.sigma[..., t], _u[..., t], a[t]])

            self.Q[..., t] = self.fQ(*_in)
            self.q[..., t] = self.fq(*_in)

            self.R[..., t] = self.fR(*_in)
            self.r[..., t] = self.fr(*_in)

            self.P[..., t] = self.fP(*_in)
            self.p[..., t] = np.reshape(self.fp(*_in),
                                        (self.dm_belief * self.dm_belief), order='F')


class LinearBeliefDynamics:
    def __init__(self, dm_belief, dm_obs, dm_act, nb_steps):
        self.dm_belief = dm_belief
        self.dm_obs = dm_obs
        self.dm_act = dm_act

        self.nb_steps = nb_steps

        # Linearization of dynamics
        self.A = np.zeros((self.dm_belief, self.dm_belief, self.nb_steps))
        self.H = np.zeros((self.dm_obs, self.dm_obs, self.nb_steps))

        # EKF matrices
        self.K = np.zeros((self.dm_belief, self.dm_obs, self.nb_steps))
        self.D = np.zeros((self.dm_belief, self.dm_obs, self.nb_steps))

        # Linearization of belief dynamics
        self.F = np.zeros((self.dm_belief, self.dm_belief, self.nb_steps))
        self.G = np.zeros((self.dm_belief, self.dm_act, self.nb_steps))

        self.T = np.zeros((self.dm_belief * self.dm_belief, self.dm_belief, self.nb_steps))
        self.U = np.zeros((self.dm_belief * self.dm_belief, self.dm_belief * self.dm_belief, self.nb_steps))
        self.V = np.zeros((self.dm_belief * self.dm_belief, self.dm_act, self.nb_steps))

        self.X = np.zeros((self.dm_belief * self.dm_belief, self.dm_belief, self.nb_steps))
        self.Y = np.zeros((self.dm_belief * self.dm_belief, self.dm_belief * self.dm_belief, self.nb_steps))
        self.Z = np.zeros((self.dm_belief * self.dm_belief, self.dm_act, self.nb_steps))

        self.sigma_x = np.zeros((self.dm_belief, self.dm_belief, self.nb_steps))
        self.sigma_z = np.zeros((self.dm_obs, self.dm_obs, self.nb_steps))

    @property
    def params(self):
        return self.A, self.H, self.K, self.D,\
               self.F, self.G, self.T, self.U,\
               self.V, self.X, self.Y, self.Z, self.y

    @params.setter
    def params(self, values):
        self.A, self.H, self.K, self.D,\
        self.F, self.G, self.T, self.U,\
        self.V, self.X, self.Y, self.Z, self.y = values


class AnalyticalLinearBeliefDynamics(LinearBeliefDynamics):
    def __init__(self, f_init, f_dyn, f_obs,
                 noise_dyn, noise_obs,
                 dm_belief, dm_obs, dm_act, nb_steps):
        super(AnalyticalLinearBeliefDynamics, self).__init__(dm_belief, dm_obs, dm_act, nb_steps)

        self.i = f_init
        self.f = f_dyn
        self.h = f_obs

        self.noise_dyn = noise_dyn
        self.noise_obs = noise_obs

        self.dfdx = jacobian(self.f, 0)
        self.dhdx = jacobian(self.h, 0)

        # # legacy
        # self.fm = lambda mu_b, sigma_b, u: self.ekf(mu_b, sigma_b, u)[0]
        # self.W = lambda mu_b, sigma_b, u: self.ekf(mu_b, sigma_b, u)[1]
        # self.phi = lambda mu_b, sigma_b, u: self.ekf(mu_b, sigma_b, u)[2]
        #
        # self.fF = jacobian(self.fm, 0)
        # self.fG = jacobian(self.fm, 2)
        #
        # self.fX = jacobian(self.W, 0)
        # self.fY = jacobian(self.W, 1)
        # self.fZ = jacobian(self.W, 2)
        #
        # self.fT = jacobian(self.phi, 0)
        # self.fU = jacobian(self.phi, 1)
        # self.fV = jacobian(self.phi, 2)

    def evali(self):
        return self.i()

    def evalf(self, mu_b, u):
        return self.f(mu_b, u)

    def evalh(self, mu_b):
        return self.h(mu_b)

    def ekf(self, mu_b, sigma_b, u):
        # extended kalman filtering
        _A = self.dfdx(mu_b, u)
        _H = self.dhdx(self.evalf(mu_b, u))

        _sigma_dyn = self.noise_dyn(mu_b, u)
        _sigma_obs = self.noise_obs(self.evalf(mu_b, u))

        _D = _A @ sigma_b @ _A.T + _sigma_dyn
        _D = 0.5 * (_D + _D.T)

        _K = _D @ _H.T @ np.linalg.inv(_H @ _D @ _H.T + _sigma_obs)

        # deterministic and stochastic mean dynamics
        _f = self.evalf(mu_b, u)
        _W = _K @ _H @ _D

        # covariance dynamics
        _phi = _D - _K @ _H @ _D
        _phi = 0.5 * (_phi + _phi.T)

        return _f, _W, _phi

    def taylor_expansion(self, b, u):
        for t in range(self.nb_steps):
            _in = tuple([b.mu[..., t], b.sigma[..., t], u[..., t]])

            _in_flat, _unflatten = flatten(_in)

            def _ekf_flat(_in_flat):
                return flatten(self.ekf(*_unflatten(_in_flat)))[0]

            _ekf_jac = jacobian(_ekf_flat)

            _grads = _ekf_jac(_in_flat)
            self.F[..., t] = _grads[:self.dm_belief, :self.dm_belief]
            self.G[..., t] = _grads[:self.dm_belief, -self.dm_act:]

            self.X[..., t] = _grads[self.dm_belief:self.dm_belief + self.dm_belief * self.dm_belief, :self.dm_belief]
            self.Y[..., t] = _grads[self.dm_belief:self.dm_belief + self.dm_belief * self.dm_belief, self.dm_belief:self.dm_belief + self.dm_belief * self.dm_belief]
            self.Z[..., t] = _grads[self.dm_belief:self.dm_belief + self.dm_belief * self.dm_belief, -self.dm_act:]

            self.T[..., t] = _grads[self.dm_belief + self.dm_belief * self.dm_belief:, :self.dm_belief]
            self.U[..., t] = _grads[self.dm_belief + self.dm_belief * self.dm_belief:, self.dm_belief:self.dm_belief + self.dm_belief * self.dm_belief]
            self.V[..., t] = _grads[self.dm_belief + self.dm_belief * self.dm_belief:, -self.dm_act:]

            # # legacy
            # self.F[..., t] = self.fF(_mu_b, _sigma_b, _u)
            # self.G[..., t] = self.fG(_mu_b, _sigma_b, _u)

            # self.X[..., t] = np.reshape(self.fX(_mu_b, _sigma_b, _u), (self.dm_belief * self.dm_belief, self.dm_belief), order='F')
            # self.Y[..., t] = np.reshape(self.fY(_mu_b, _sigma_b, _u), (self.dm_belief * self.dm_belief, self.dm_belief * self.dm_belief), order='F')
            # self.Z[..., t] = np.reshape(self.fZ(_mu_b, _sigma_b, _u), (self.dm_belief * self.dm_belief, self.dm_act), order='F')

            # self.T[..., t] = np.reshape(self.fT(_mu_b, _sigma_b, _u), (self.dm_belief * self.dm_belief, self.dm_belief), order='F')
            # self.U[..., t] = np.reshape(self.fU(_mu_b, _sigma_b, _u), (self.dm_belief * self.dm_belief, self.dm_belief * self.dm_belief), order='F')
            # self.V[..., t] = np.reshape(self.fV(_mu_b, _sigma_b, _u), (self.dm_belief * self.dm_belief, self.dm_act), order='F')

    def forward(self, b, u, t):
        _u = u[..., t]

        _mu_b, _sigma_b = b.mu[..., t], b.sigma[..., t]
        _mu_bn, _, _sigma_bn = self.ekf(_mu_b, _sigma_b, _u)

        return _mu_bn, _sigma_bn


class LinearControl:
    def __init__(self, dm_belief, dm_act, nb_steps):
        self.dm_belief = dm_belief
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.K = np.zeros((self.dm_act, self.dm_belief, self.nb_steps))
        self.kff = np.zeros((self.dm_act, self.nb_steps))

    @property
    def params(self):
        return self.K, self.kff

    @params.setter
    def params(self, values):
        self.K, self.kff = values

    def action(self, b, alpha, bref, uref, t):
        dx = b.mu[..., t] - bref[:, t]
        return uref[:, t] + alpha * self.kff[..., t] + self.K[..., t] @ dx
