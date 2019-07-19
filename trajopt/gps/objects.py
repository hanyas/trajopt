#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: objects.py
# @Date: 2019-06-22-15-01
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
from autograd import jacobian, hessian
from copy import deepcopy


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

    def sample(self, x):
        pass


class QuadraticStateValue:
    def __init__(self, dm_state, nb_steps):
        self.dm_state = dm_state
        self.nb_steps = nb_steps

        self.V = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.v = np.zeros((self.dm_state, self.nb_steps, ))
        self.v0 = np.zeros((self.nb_steps, ))
        self.v0_softmax = np.zeros((self.nb_steps, ))


class QuadraticStateActionValue:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.Qxx = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.Quu = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        self.Qux = np.zeros((self.dm_act, self.dm_state, self.nb_steps))

        self.qx = np.zeros((self.dm_state, self.nb_steps, ))
        self.qu = np.zeros((self.dm_act, self.nb_steps, ))

        self.q0 = np.zeros((self.nb_steps, ))
        self.q0_common = np.zeros((self.nb_steps, ))
        self.q0_softmax = np.zeros((self.nb_steps, ))


class QuadraticCost:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act

        self.nb_steps = nb_steps

        self.Cxx = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.cx = np.zeros((self.dm_state, self.nb_steps))

        self.Cuu = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        self.cu = np.zeros((self.dm_act, self.nb_steps))

        self.Cxu = np.zeros((self.dm_state, self.dm_act, self.nb_steps))
        self.c0 = np.zeros((self.nb_steps, ))

    @property
    def params(self):
        return self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu, self.c0

    @params.setter
    def params(self, values):
        self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu, self.c0 = values

    def evaluate(self, x, u):
        _ret = 0.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))
        for t in range(self.nb_steps):
            _ret += x[..., t].T @ self.Cxx[..., t] @ x[..., t] +\
                    _u[..., t].T @ self.Cuu[..., t] @ _u[..., t] +\
                    x[..., t].T @ self.Cxu[..., t] @ _u[..., t] +\
                    self.cx[..., t].T @ x[..., t] +\
                    self.cu[..., t].T @ _u[..., t] + self.c0[..., t]
        return _ret

class AnalyticalQuadraticCost(QuadraticCost):
    def __init__(self, f, dm_state, dm_act, nb_steps):
        super(AnalyticalQuadraticCost, self).__init__(dm_state, dm_act, nb_steps)

        self.f = f

        self.dcdxx = hessian(self.f, 0)
        self.dcduu = hessian(self.f, 1)
        self.dcdxu = jacobian(jacobian(self.f, 0), 1)

        self.dcdx = jacobian(self.f, 0)
        self.dcdu = jacobian(self.f, 1)

    def evalf(self, x, u, a):
        _xref = deepcopy(x)
        return self.f(x, u, a, _xref)

    def taylor_expansion(self, x, u, a):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))
        _xref = deepcopy(x)
        for t in range(self.nb_steps):
            _in = tuple([x[..., t], _u[..., t], a[t], _xref[..., t]])
            self.Cxx[..., t] = 0.5 * self.dcdxx(*_in)
            self.Cuu[..., t] = 0.5 * self.dcduu(*_in)
            self.Cxu[..., t] = self.dcdxu(*_in)

            self.cx[..., t] = self.dcdx(*_in) - self.dcdxx(*_in) @ x[..., t] - self.dcdxu(*_in) @ _u[..., t]
            self.cu[..., t] = self.dcdu(*_in) - self.dcduu(*_in) @ _u[..., t] - x[..., t].T @ self.dcdxu(*_in)

            # residual of taylor expansion
            self.c0[..., t] = self.f(*_in) -\
                              x[..., t].T @ self.Cxx[..., t] @ x[..., t] -\
                              _u[..., t].T @ self.Cuu[..., t] @ _u[..., t] -\
                              x[..., t].T @ self.Cxu[..., t] @ _u[..., t] -\
                              self.cx[..., t].T @ x[..., t] -\
                              self.cu[..., t].T @ _u[..., t]


class LinearGaussianDynamics:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.A = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.B = np.zeros((self.dm_state, self.dm_act, self.nb_steps))
        self.c = np.zeros((self.dm_state, self.nb_steps))
        self.sigma = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = 1e-8 * np.eye(self.dm_state)

    @property
    def params(self):
        return self.A, self.B, self.c, self.sigma

    @params.setter
    def params(self, values):
        self.A, self.B, self.c, self.sigma = values

    def sample(self, x, u):
        pass


class AnalyticalLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, f_init, f_dyn, noise, dm_state, dm_act, nb_steps):
        super(AnalyticalLinearGaussianDynamics, self).__init__(dm_state, dm_act, nb_steps)

        self.i = f_init
        self.f = f_dyn
        self.noise = noise

        self.dfdx = jacobian(self.f, 0)
        self.dfdu = jacobian(self.f, 1)

    def evali(self):
        return self.i()

    def evalf(self, x, u):
        return self.f(x, u)

    def taylor_expansion(self, x, u):
        _A = self.dfdx(x, u)
        _B = self.dfdu(x, u)
        # residual of taylor expansion
        _c = self.evalf(x, u) - _A @ x - _B @ u
        _sigma = self.noise(x, u)
        return _A, _B, _c, _sigma

    def forward(self, x, u, lgc, t):
        _mu_x, _sigma_x = x.mu[..., t], x.sigma[..., t]
        _mu_u, _sigma_u = u.mu[..., t], u.sigma[..., t]

        _K, _kff, _ctl_sigma = lgc.K[..., t], lgc.kff[..., t], lgc.sigma[..., t]
        _A, _B, _c, _dyn_sigma = self.taylor_expansion(_mu_x, _mu_u)

        _mu_xn = self.evalf(_mu_x, _mu_u)

        _AB = np.hstack((_A, _B))
        _sigma_xu = np.vstack((np.hstack((_sigma_x, _sigma_x @ _K.T)),
                               np.hstack((_K @ _sigma_x, _sigma_u))))

        _sigma_xn = _dyn_sigma + _AB @ _sigma_xu @ _AB.T
        _sigma_xn = 0.5 * (_sigma_xn + _sigma_xn.T)

        self.A[..., t], self.B[..., t], self.c[..., t] = _A, _B, _c
        self.sigma[..., t] = _dyn_sigma

        return _mu_xn, _sigma_xn


class LearnedLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, dm_state, dm_act, nb_steps):
        super(LearnedLinearGaussianDynamics, self).__init__(dm_state, dm_act, nb_steps)

    def learn(self, data, pointwise=False):
        if pointwise:
            from mimo import distributions
            _hypparams = dict(M=np.zeros((self.dm_state, self.dm_state + self.dm_act + 1)),
                              V=1.e6 * np.eye(self.dm_state + self.dm_act + 1),
                              affine=True,
                              psi=np.eye(self.dm_state), nu=self.dm_state + 2)
            _prior = distributions.MatrixNormalInverseWishart(**_hypparams)

            for t in range(self.nb_steps):
                _data = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T, data['xn'][:, t, :].T))

                _model = distributions.BayesianLinearGaussian(_prior)
                _model = _model.MAP(_data)

                self.A[..., t] = _model.A[:, :self.dm_state]
                self.B[..., t] = _model.A[:, self.dm_state:self.dm_state + self.dm_act]
                self.c[..., t] = _model.A[:, -1]
                self.sigma[..., t] = _model.sigma
        else:
            _obs = [data['x'][..., n].T for n in range(data['x'].shape[-1])]
            _input = [data['u'][..., n].T for n in range(data['u'].shape[-1])]

            from sds.rarhmm_ls import rARHMM
            rarhmm = rARHMM(nb_states=5, dim_obs=self.dm_state, dim_act=self.dm_act)
            rarhmm.initialize(_obs, _input)
            rarhmm.em(_obs, _input, nb_iter=50, prec=1e-12, verbose=False)

            _mean_obs = np.mean(data['x'], axis=-1).T
            _mean_input = np.mean(data['u'], axis=-1).T
            _, _mean_z = rarhmm.viterbi([_mean_obs], [_mean_input])

            for t in range(self.nb_steps):
                self.A[..., t] = rarhmm.observations.A[_mean_z[0][t], ...]
                self.B[..., t] = rarhmm.observations.B[_mean_z[0][t], ...]
                self.c[..., t] = rarhmm.observations.c[_mean_z[0][t], ...]


class LinearGaussianControl:
    def __init__(self, dm_state, dm_act, nb_steps, init_ctl_sigma=1.):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.K = np.zeros((self.dm_act, self.dm_state, self.nb_steps))
        self.kff = np.zeros((self.dm_act, self.nb_steps))

        self.sigma = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = init_ctl_sigma * np.eye(self.dm_act)

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

    def forward(self, xdist, t):
        _x_mu, _x_sigma = xdist.mu[..., t], xdist.sigma[..., t]
        _K, _kff, _ctl_sigma = self.K[..., t], self.kff[..., t], self.sigma[..., t]

        _u_mu = _K @ _x_mu + _kff
        _u_sigma = _ctl_sigma + _K @ _x_sigma @ _K.T
        _u_sigma = 0.5 * (_u_sigma + _u_sigma.T)

        return _u_mu, _u_sigma
