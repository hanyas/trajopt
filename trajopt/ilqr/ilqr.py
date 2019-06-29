#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: ilqr
# @Date: 2019-06-23-14-00
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

from trajopt.ilqr.objects import AnalyticalLinearDynamics, AnalyticalQuadraticCost
from trajopt.ilqr.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.ilqr.objects import LinearControl

from trajopt.ilqr.core import backward_pass


class iLQR:

    def __init__(self, env, nb_steps,
                 alphas=np.power(10., np.linspace(0, -3, 11)),
                 lmbda=1., dlmbda=1.,
                 min_lmbda=1.e-6, max_lmbda=1.e3, mult_lmbda=1.6,
                 tolfun=1.e-7, tolgrad=1.e-4, min_imp=0., reg=1,
                 activation='last'):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_cost = self.env.unwrapped.cost
        self.env_init = self.env.unwrapped.init

        self.ulim = self.env.action_space.high

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # backtracking
        self.alphas = alphas
        self.lmbda = lmbda
        self.dlmbda = dlmbda
        self.min_lmbda = min_lmbda
        self.max_lmbda = max_lmbda
        self.mult_lmbda = mult_lmbda

        # regularization type
        self.reg = reg

        # minimum relative improvement
        self.min_imp = min_imp

        # stopping criterion
        self.tolfun = tolfun
        self.tolgrad = tolgrad

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

        self.last_objective = - np.inf

    def forward_pass(self, ctl, alpha):
        state = np.zeros((self.nb_xdim, self.nb_steps + 1))
        action = np.zeros((self.nb_udim, self.nb_steps))

        state[..., 0], _ = self.dyn.evali()
        for t in range(self.nb_steps):
            action[..., t] = ctl.action(state, alpha, self.xref, self.uref, t)
            state[..., t + 1] = self.dyn.evalf(state[..., t], action[..., t])

        return state, action

    def backward_pass(self):
        lc = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        xvalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu,\
        xvalue.V, xvalue.v, dV,\
        lc.K, lc.kff, diverge = backward_pass(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                              self.cost.cu, self.cost.Cxu,
                                              self.dyn.A, self.dyn.B,
                                              self.lmbda, self.reg,
                                              self.nb_xdim, self.nb_udim, self.nb_steps)
        return lc, xvalue, xuvalue, dV, diverge

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

    def run(self, nb_iter=25):
        _trace = []
        # init trajectory
        for alpha in self.alphas:
            _state, _action = self.forward_pass(self.ctl, alpha)
            if np.all(_state < 1.e8):
                self.xref = _state
                self.uref = _action
                self.last_objective = self.objective(self.xref, self.uref)
                break
            else:
                print("Initial trajectory diverges")

        _trace.append(self.last_objective)

        for _ in range(nb_iter):
            # get linear system dynamics around ref traj.
            self.dyn.finite_diff(self.xref, self.uref)

            # get quadratic cost around ref traj.
            self.cost.finite_diff(self.xref, self.uref, self.activation)

            xvalue, xuvalue = None, None
            lc, dvalue = None, None
            # execute a backward pass
            backpass_done = False
            while not backpass_done:
                lc, xvalue, xuvalue, dvalue, diverge = self.backward_pass()
                if np.any(diverge):
                    # increase lmbda
                    self.dlmbda = np.maximum(self.dlmbda * self.mult_lmbda, self.mult_lmbda)
                    self.lmbda = np.maximum(self.lmbda * self.dlmbda, self.min_lmbda)
                    if self.lmbda > self.max_lmbda:
                        break
                    else:
                        continue
                else:
                    backpass_done = True

            # terminate if gradient too small
            _g_norm = np.mean(np.max(np.abs(lc.kff) / (np.abs(self.uref) + 1.), axis=1))
            if _g_norm < self.tolgrad and self.lmbda < 1.e-5:
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)
                break

            _state, _action = None, None
            _return, _dreturn = None, None
            # execute a forward pass
            fwdpass_done = False
            if backpass_done:
                for alpha in self.alphas:
                    # apply on actual system
                    _state, _action = self.forward_pass(ctl=lc, alpha=alpha)

                    # summed mean return
                    _return = self.objective(_state, _action)

                    # check return improvement
                    _dreturn = self.last_objective - _return
                    _expected = - 1. * alpha * (dvalue[0] + alpha * dvalue[1])
                    _imp = _dreturn / _expected
                    if _imp > self.min_imp:
                        fwdpass_done = True
                        break

            # accept or reject
            if fwdpass_done:
                # decrease lmbda
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)

                self.xref = _state
                self.uref = _action
                self.last_objective = _return

                self.vfunc = xvalue
                self.qfunc = xuvalue

                self.ctl = lc

                _trace.append(self.last_objective)

                # terminate if reached objective tolerance
                if _dreturn < self.tolfun:
                    break
            else:
                # increase lmbda
                self.dlmbda = np.maximum(self.dlmbda * self.mult_lmbda, self.mult_lmbda)
                self.lmbda = np.maximum(self.lmbda * self.dlmbda, self.min_lmbda)
                if self.lmbda > self.max_lmbda:
                    break
                else:
                    continue

        return _trace
