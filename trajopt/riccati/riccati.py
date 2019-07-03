#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: ilqr
# @Date: 2019-06-23-14-00
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

from trajopt.riccati.objects import AnalyticalLinearDynamics, AnalyticalQuadraticCost
from trajopt.riccati.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.riccati.objects import LinearControl


class Riccati:

    def __init__(self, env, nb_steps,
                 activation='all'):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_cost = self.env.unwrapped.cost
        self.env_init = self.env.unwrapped.init

        self.ulim = self.env.action_space.high

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # reference trajectory
        self.xref = np.zeros((self.nb_xdim, self.nb_steps + 1))
        self.xref[..., 0] = self.env_init()[0]

        self.uref = np.zeros((self.nb_udim, self.nb_steps))

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.dyn = AnalyticalLinearDynamics(self.env_init, self.env_dyn, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)

        # activation of cost function
        if activation == 'all':
            self.activation = np.ones((self.nb_steps + 1,), dtype=np.int64)
        else:
            self.activation = np.zeros((self.nb_steps + 1, ), dtype=np.int64)
            self.activation[-1] = 1

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.nb_xdim, self.nb_udim, self.nb_steps + 1)

    def forward_pass(self, ctl):
        state = np.zeros((self.nb_xdim, self.nb_steps + 1))
        action = np.zeros((self.nb_udim, self.nb_steps))
        cost = np.zeros((self.nb_steps + 1, ))

        state[..., 0], _ = self.dyn.evali()
        for t in range(self.nb_steps):
            action[..., t] = ctl.action(state[..., t], t)
            cost[..., t] = self.cost.evalf(state[..., t], action[..., t], self.activation[t])
            state[..., t + 1] = self.dyn.evalf(state[..., t], action[..., t])

        cost[..., -1] = self.cost.evalf(state[..., -1], np.zeros((self.nb_udim, )), self.activation[-1])
        return state, action, cost

    def backward_pass(self):
        lc = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        xvalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)

        xvalue.V[..., -1] = self.cost.Cxx[..., -1]
        xvalue.v[..., -1] = self.cost.cx[..., -1]
        for t in range(self.nb_steps - 2, -1, -1):
            _Qxx = self.cost.Cxx[..., t] + self.dyn.A[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.A[..., t]
            _Quu = self.cost.Cuu[..., t] + self.dyn.B[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.B[..., t]
            _Qux = self.cost.Cxu[..., t].T + self.dyn.B[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.A[..., t]

            _qx = self.cost.cx[..., t] + self.dyn.A[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.c[..., t] +\
                  self.dyn.A[..., t].T @ xvalue.v[..., t + 1]

            _qu = self.cost.cu[..., t] + self.dyn.B[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.c[..., t] +\
                  self.dyn.B[..., t].T @ xvalue.v[..., t + 1]

            _Quu_inv = np.linalg.inv(_Quu)

            lc.K[...,t] = - _Quu_inv @ _Qux
            lc.kff[..., t] = - _Quu_inv @ _qu

            xvalue.V[..., t] = _Qxx + _Qux.T * lc.K[..., t]
            xvalue.v[..., t] = _qx + lc.kff[..., t].T @ _Qux

        return lc, xvalue

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

    def run(self):
        # get linear system dynamics around ref traj.
        self.dyn.taylor_expansion(self.xref, self.uref)

        # get quadratic cost around ref traj.
        self.cost.taylor_expansion(self.xref, self.uref, self.activation)

        # backward pass to get ctrl.
        self.ctl, self.vfunc = self.backward_pass()

        # forward pass to get cost and traj.
        self.xref, self.uref, _cost = self.forward_pass(self.ctl)

        return np.sum(_cost)

