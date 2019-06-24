#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: ilqg
# @Date: 2019-06-23-14-00
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

from trajopt.ilqg.objects import AnalyticalLinearGaussianDynamics, AnalyticalQuadraticReward
from trajopt.ilqg.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.ilqg.objects import LinearControl

from trajopt.ilqg.core import backward_pass


class iLQG:

    def __init__(self, env, nb_steps,
                 alphas=np.power(10., np.linspace(0, -3, 11)),
                 lmbda=1., dlmbda=1.,
                 min_lmbda=1.e-6, max_lmbda=1.e3, mult_lmbda=1.6,
                 tolfun=1.e-7, tolgrad=1.e-4, zmin=0., reg=1,
                 activation='last'):

        self.env = env

        self.env_dyn = self.env.unwrapped.dynamics
        self.env_sigma = self.env.unwrapped.sigma

        self.env_rwrd = self.env.unwrapped.reward
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
        self.zmin = zmin

        # stopping criterion
        self.tolfun = tolfun
        self.tolgrad = tolgrad

        # reference trajectory
        self.xref = np.zeros((self.nb_xdim, self.nb_steps + 1))
        for t in range(self.nb_steps + 1):
            self.xref[..., t], _ = self.env_init()

        self.uref = np.zeros((self.nb_udim, self.nb_steps))

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.dyn = AnalyticalLinearGaussianDynamics(self.env_dyn, self.env_sigma, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)

        # activation of reward function
        if activation == 'all':
            self.activation = np.ones((self.nb_steps + 1,), dtype=np.int64)
        else:
            self.activation = np.zeros((self.nb_steps + 1, ), dtype=np.int64)
            self.activation[-1] = 1
        self.rwrd = AnalyticalQuadraticReward(self.env_rwrd, self.nb_xdim, self.nb_udim,
                                              self.nb_steps + 1, self.activation)

        self.last_return = - np.inf

    def forward_pass(self, ctl, alpha):
        data = {'x': np.zeros((self.nb_xdim, self.nb_steps + 1)),
                'u': np.zeros((self.nb_udim, self.nb_steps))}

        x = self.env.reset()
        data['x'][..., 0] = x

        for t in range(self.nb_steps):
            u = ctl.apply(x, alpha, self.xref, self.uref, t)
            data['u'][..., t] = u

            x, _, _, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))
            data['x'][..., t + 1] = x

        return data

    def backward_pass(self):
        lc = LinearControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        xvalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu,\
        xvalue.V, xvalue.v, dV,\
        lc.K, lc.kff, diverge = backward_pass(self.rwrd.Rxx, self.rwrd.rx, self.rwrd.Ruu,
                                              self.rwrd.ru, self.rwrd.Rxu,
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

    def run(self, nb_iter=250):
        _trace = []
        _uref_padd = np.hstack((self.uref, np.zeros((self.nb_udim, 1))))
        _trace.append(self.rwrd.evalf(self.xref, _uref_padd))

        for _ in range(nb_iter):
            # get linear system dynamics around ref traj.
            self.dyn.diff(self.xref, self.uref)

            # get quadratic reward around ref traj.
            self.rwrd.diff(self.xref, self.uref)

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
            g_norm = np.mean(np.max(np.abs(lc.kff) / (np.abs(self.uref[..., -1]) + 1.), axis=1))
            if g_norm < self.tolgrad and self.lmbda < 1.e-5:
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)
                break

            _data, _return, _dreturn = None, None, None
            # execute a forward pass
            fwdpass_done = False
            if backpass_done:
                for alpha in self.alphas:
                    # apply on actual system
                    _data = self.forward_pass(ctl=lc, alpha=alpha)
                    _u_padd = np.hstack((_data['u'], np.zeros((self.nb_udim, 1))))
                    _return = self.rwrd.evalf(_data['x'], _u_padd)

                    # check return improvement
                    _dreturn = _return - self.last_return
                    expected = alpha * (dvalue[0] + alpha * dvalue[1])
                    z = _dreturn / expected
                    if z > self.zmin:
                        fwdpass_done = True
                        break

            # accept or reject
            if fwdpass_done:
                # decrease lmbda
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)

                self.xref = _data['x']
                self.uref = _data['u']

                self.vfunc = xvalue
                self.qfunc = xuvalue

                self.last_return = _return
                self.ctl = lc

                _trace.append(_return)

                # terminate if reached reward tolerance
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
