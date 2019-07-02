#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: ilqr
# @Date: 2019-06-23-14-00
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

from trajopt.elqr.objects import AnalyticalLinearDynamics, AnalyticalQuadraticCost
from trajopt.elqr.objects import QuadraticStateValue
from trajopt.elqr.objects import LinearControl


class eLQR:

    def __init__(self, env, nb_steps,
                 activation='last'):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_inv_dyn = self.env.unwrapped.inverse_dynamics
        self.env_cost = self.env.unwrapped.cost
        self.env_init = self.env.unwrapped.init
        self.env_goal = self.env.unwrapped.goal

        self.ulim = self.env.action_space.high

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # reference trajectory
        self.xref = np.zeros((self.nb_xdim, self.nb_steps + 1))
        self.xref[..., 0] = self.env_init()[0]

        self.uref = np.zeros((self.nb_udim, self.nb_steps))

        self.gocost = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.comecost = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)

        self.gocost.V[..., 0] += np.eye(self.nb_xdim) * 1e-16
        self.comecost.V[..., 0] += np.eye(self.nb_xdim) * 1e-16

        self.dyn = AnalyticalLinearDynamics(self.env_init, self.env_dyn, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.idyn = AnalyticalLinearDynamics(self.env_init, self.env_inv_dyn, self.nb_xdim, self.nb_udim, self.nb_steps)

        self.ctl = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl.kff = np.random.randn(self.nb_udim, self.nb_steps)

        self.ictl = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ictl.kff = 1e-2 * np.random.randn(self.nb_udim, self.nb_steps)

        # activation of cost function
        if activation == 'all':
            self.activation = np.ones((self.nb_steps + 1,), dtype=np.int64)
        else:
            self.activation = np.zeros((self.nb_steps + 1, ), dtype=np.int64)
            self.activation[-1] = 1

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.nb_xdim, self.nb_udim, self.nb_steps + 1)

        self.last_objective = - np.inf

    def forward_pass(self, ctl):
        state = np.zeros((self.nb_xdim, self.nb_steps + 1))
        action = np.zeros((self.nb_udim, self.nb_steps))

        state[..., 0], _ = self.dyn.evali()
        for t in range(self.nb_steps):
            action[..., t] = ctl.action(state[..., t], t)
            state[..., t + 1] = self.dyn.evalf(state[..., t], action[..., t])

        return state, action

    def forward_lqr(self, state):
        for t in range(self.nb_steps):
            _action = self.ctl.action(state, t)

            _state_n = self.dyn.evalf(state, _action)

            # linearize inverse discrete dynamics
            _A, _B, _c = self.idyn.taylor_expansion(_state_n, _action)

            # quadratize cost
            _Cxx, _Cuu, _Cxu, _cx, _cu, _c0 = self.cost.taylor_expansion(state, _action, self.activation[..., t])

            # forward value
            _Qxx = _A.T @ (_Cxx + self.comecost.V[..., t]) @ _A
            _Quu = _B.T @ (_Cxx + self.comecost.V[..., t]) @ _B + _B.T @ _Cxu + _Cxu.T  @ _B + _Cuu
            _Qux = _B.T @ (_Cxx + self.comecost.V[..., t]) @ _A + _Cxu.T @ _A

            _qx = _A.T @ (_Cxx + self.comecost.V[..., t]) @ _c + _A.T @ (_cx + self.comecost.v[..., t])
            _qu = _B.T @ (_Cxx + self.comecost.V[..., t]) @ _c + _Cxu.T @ _c + _B.T @ (_cx + self.comecost.v[..., t]) + _cu

            _q0 = 0.5 * _c.T @ (_Cxx + self.comecost.V[..., t]) @ _c +\
                  _c.T @ (_cx + self.comecost.v[..., t]) + _c0 + self.comecost.v0[..., t]

            # backward value
            _Qxx = _A.T @ (_Cxx + self.comecost.V[..., t]) @ _A
            _Quu = _B.T @ (_Cxx + self.comecost.V[..., t]) @ _B + _B.T @ _Cxu + _Cxu.T  @ _B + _Cuu
            _Qux = _B.T @ (_Cxx + self.comecost.V[..., t]) @ _A + _Cxu.T @ _A

            self.ictl.K[..., t] = - np.linalg.inv(_Quu) @ _Qux
            self.ictl.kff[..., t] = - np.linalg.inv(_Quu) @ _qu

            self.comecost.V[..., t + 1] = _Qxx - _Qux.T @ np.linalg.inv(_Quu) @ _Qux
            self.comecost.v[..., t + 1] = _qx - _Qux.T @ np.linalg.inv(_Quu) @ _qu
            self.comecost.v0[..., t + 1] = _q0 - 0.5 * _qu.T @ np.linalg.inv(_Quu) @ _qu

            # store matrices
            self.idyn.A[..., t] = _A
            self.idyn.B[..., t] = _B
            self.idyn.c[..., t] = _c

            state = - np.linalg.inv(self.gocost.V[..., t + 1] + self.comecost.V[..., t + 1]) @\
                    (self.gocost.v[..., t + 1] + self.comecost.v[..., t + 1])

        return state

    def backward_lqr(self, state):
        # quadratize last cost
        _Cxx, _Cuu, _Cxu, _cx, _cu, _c0 =\
            self.cost.taylor_expansion(state, np.zeros((self.nb_udim, )), self.activation[..., -1])

        self.gocost.V[..., -1] = _Cxx
        self.gocost.v[..., -1] = _cx
        self.gocost.v0[..., -1] = _c0

        state = - np.linalg.inv(self.gocost.V[..., -1] + self.comecost.V[..., -1]) @\
                (self.gocost.v[..., -1] + self.comecost.v[..., -1])

        for t in range(self.nb_steps - 1, -1, -1):
            _action = self.ictl.action(state, t)

            _state_n = self.idyn.evalf(state, _action)
            # linearize discrete dynamics
            _A, _B, _c = self.dyn.taylor_expansion(_state_n, _action)

            # quadratize cost
            _Cxx, _Cuu, _Cxu, _cx, _cu, _c0 = self.cost.taylor_expansion(_state_n, _action, self.activation[..., t])

            # backward value
            _Qxx = _Cxx + _A.T @ self.gocost.V[..., t + 1] @ _A
            _Quu = _Cuu + _B.T @ self.gocost.V[..., t + 1] @ _B
            _Qux = _Cxu.T + _B.T @ self.gocost.V[..., t + 1] @ _A

            _qx = _cx + _A.T @ self.gocost.V[..., t + 1] @ _c + _A.T @ self.gocost.v[..., t + 1]
            _qu = _cu + _B.T @ self.gocost.V[..., t + 1] @ _c + _B.T @ self.gocost.v[..., t + 1]

            _q0 = _c0 + self.gocost.v0[..., t + 1] + 0.5 * _c.T @ self.gocost.V[..., t + 1] @ _c +\
                  _c.T @ self.gocost.v[..., t + 1]

            self.ctl.K[..., t] = - np.linalg.inv(_Quu) @ _Qux
            self.ctl.kff[..., t] = - np.linalg.inv(_Quu) @ _qu

            self.gocost.V[..., t] = _Qxx - _Qux.T @ np.linalg.inv(_Quu) @ _Qux
            self.gocost.v[..., t] = _qx - _Qux.T @ np.linalg.inv(_Quu) @ _qu
            self.gocost.v0[..., t] = _q0 - 0.5 * _qu.T @ np.linalg.inv(_Quu) @ _qu

            # store matrices
            self.dyn.A[..., t] = _A
            self.dyn.B[..., t] = _B
            self.dyn.c[..., t] = _c

            state = - np.linalg.inv(self.gocost.V[..., t] + self.comecost.V[..., t]) @\
                     (self.gocost.v[..., t] + self.comecost.v[..., t])

        return state

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.nb_xdim):
            plt.subplot(self.nb_xdim + self.nb_udim, 1, k + 1)
            plt.plot(t, self.xref[k, :], '-b')

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.nb_udim):
            plt.subplot(self.nb_xdim + self.nb_udim, 1, self.nb_xdim + k + 1)
            plt.plot(t, self.uref[k, :], '-g')

        plt.show()

    def objective(self, x, u):
        _return = 0.0
        for t in range(self.nb_steps):
            _return += self.cost.evalf(x[..., t], u[..., t], self.activation[..., t])
        _return += self.cost.evalf(x[..., -1], np.zeros((self.nb_udim,)), self.activation[..., -1])

        return _return

    def run(self, nb_iter=10):
        _trace = []
        _state, _ = self.dyn.evali()
        for _ in range(nb_iter):
            # forward pass to get ref traj.
            self.xref, self.uref = self.forward_pass(self.ctl)

            # return around current traj.
            _trace.append(self.objective(self.xref, self.uref))

            # forward lqr
            _state = self.forward_lqr(_state)

            # backward lqr
            _state = self.backward_lqr(_state)

        _trace.append(self.objective(self.xref, self.uref))

        return _trace
