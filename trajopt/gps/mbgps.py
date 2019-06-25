#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: gps.py
# @Date: 2019-06-06-08-56
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

import scipy as sc
from scipy import optimize

from trajopt.gps.objects import Gaussian, QuadraticCost
from trajopt.gps.objects import AnalyticalLinearGaussianDynamics, AnalyticalQuadraticCost
from trajopt.gps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.gps.objects import LinearGaussianControl

from trajopt.gps.core import kl_divergence, quad_expectation, augment_cost
from trajopt.gps.core import forward_pass, backward_pass


class MBGPS:

    def __init__(self, env, nb_steps, kl_bound,
                 init_ctl_sigma, activation='last'):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_noise = self.env.unwrapped.noise
        self.env_cost = self.env.unwrapped.cost
        self.env_init = self.env.unwrapped.init

        self.ulim = self.env.action_space.high

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # total kl over traj.
        self.kl_bound = kl_bound
        self.alpha = np.array([-100.])

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.nb_xdim, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init()

        self.udist = Gaussian(self.nb_udim, self.nb_steps)
        self.xudist = Gaussian(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.dyn = AnalyticalLinearGaussianDynamics(self.env_dyn, self.env_noise, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.ctl = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps, init_ctl_sigma)

        # activation of cost function
        if activation == 'all':
            self.activation = np.ones((self.nb_steps + 1,), dtype=np.int64)
        else:
            self.activation = np.zeros((self.nb_steps + 1, ), dtype=np.int64)
            self.activation[-1] = 1

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.nb_xdim, self.nb_udim, self.nb_steps + 1)

    def sample(self, nb_episodes, stoch=True):
        data = {'x': np.zeros((self.nb_xdim, self.nb_steps, nb_episodes)),
                'u': np.zeros((self.nb_udim, self.nb_steps, nb_episodes)),
                'xn': np.zeros((self.nb_xdim, self.nb_steps, nb_episodes)),
                'c': np.zeros((self.nb_steps + 1, nb_episodes))}

        for n in range(nb_episodes):
            x = self.env.reset()

            for t in range(self.nb_steps):
                u = self.ctl.sample(x, t, stoch)
                data['u'][..., t, n] = u

                # expose true reward function
                c = self.env_cost(x, u, self.activation[t])
                data['c'][t] = c

                data['x'][..., t, n] = x
                x, _, _, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))
                data['xn'][..., t, n] = x

            c = self.env_cost(x, np.zeros((self.nb_udim, )), self.activation[-1])
            data['c'][-1, n] = c

        return data

    def forward_pass(self, lgc, kalman=False):
        xdist = Gaussian(self.nb_xdim, self.nb_steps + 1)
        udist = Gaussian(self.nb_udim, self.nb_steps)
        xudist = Gaussian(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        if kalman:
            xdist.mu[..., 0], xdist.sigma[..., 0] = self.env_init()
            for t in range(self.nb_steps):
                udist.mu[..., t] = lgc.K[..., t] @ xdist.mu[..., t] + lgc.kff[..., t]
                udist.sigma[..., t] = lgc.sigma[..., t] + lgc.K[..., t] @ xdist.sigma[..., t] @ lgc.K[..., t].T
                udist.sigma[..., t] = 0.5 * (udist.sigma[..., t] + udist.sigma[..., t].T)

                xudist.mu[..., t] = np.hstack((xdist.mu[..., t], udist.mu[..., t]))
                xudist.sigma[..., t] = np.vstack((np.hstack((xdist.sigma[..., t], xdist.sigma[..., t] @ lgc.K[..., t].T)),
                                                  np.hstack((lgc.K[..., t] @ xdist.sigma[..., t], udist.sigma[..., t]))))
                xudist.sigma[..., t] = 0.5 * (xudist.sigma[..., t] + xudist.sigma[..., t].T)

                # online linearization effectively doing extended Kalman filtering
                self.dyn.A[..., t], self.dyn.B[..., t],\
                self.dyn.c[..., t], self.dyn.sigma[..., t] = self.dyn.finite_diff(xdist.mu[..., t], udist.mu[..., t])

                xdist.mu[..., t + 1] = np.hstack((self.dyn.A[..., t], self.dyn.B[..., t])) @\
                                       xudist.mu[..., t] + self.dyn.c[..., t]
                xdist.sigma[..., t + 1] = self.dyn.sigma[..., t] + \
                                          np.hstack((self.dyn.A[..., t], self.dyn.B[..., t])) @ xudist.sigma[..., t] @\
                                                     np.vstack((self.dyn.A[..., t].T, self.dyn.B[..., t].T))
                xdist.sigma[..., t + 1] = 0.5 * (xdist.sigma[..., t + 1] + xdist.sigma[..., t + 1].T)
        else:
            xdist.mu, xdist.sigma,\
            udist.mu, udist.sigma,\
            xudist.mu, xudist.sigma = forward_pass(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                                   self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                   lgc.K, lgc.kff, lgc.sigma,
                                                   self.nb_xdim, self.nb_udim, self.nb_steps)
        return xdist, udist, xudist

    def backward_pass(self, alpha, agcost):
        lgc = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        xvalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu, xuvalue.q0, xuvalue.q0_softmax,\
        xvalue.V, xvalue.v, xvalue.v0, xvalue.v0_softmax,\
        lgc.K, lgc.kff, lgc.sigma = backward_pass(agcost.Cxx, agcost.cx, agcost.Cuu,
                                                  agcost.cu, agcost.Cxu, agcost.c0,
                                                  self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                  alpha, self.nb_xdim, self.nb_udim, self.nb_steps)
        return lgc, xvalue, xuvalue

    def augment_cost(self, alpha):
        agcost = QuadraticCost(self.nb_xdim, self.nb_udim, self.nb_steps + 1)
        agcost.Cxx, agcost.cx, agcost.Cuu,\
        agcost.cu, agcost.Cxu, agcost.c0 = augment_cost(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                                        self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                        self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                        alpha, self.nb_xdim, self.nb_udim, self.nb_steps)
        return agcost

    def dual(self, alpha):
        # augmented cost
        agcost = self.augment_cost(alpha)

        # backward pass
        lgc, xvalue, xuvalue = self.backward_pass(alpha, agcost)

        # forward pass
        xdist, udist, xudist = self.forward_pass(lgc)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0_softmax[..., 0])
        dual += alpha * self.kl_bound

        # gradient
        grad = self.kl_bound - self.kldiv(lgc, xdist)

        return -1. * np.array([dual]), -1. * np.array([grad])

    def kldiv(self, lgc, xdist):
        return kl_divergence(lgc.K, lgc.kff, lgc.sigma,
                             self.ctl.K, self.ctl.kff, self.ctl.sigma,
                             xdist.mu, xdist.sigma,
                             self.nb_xdim, self.nb_udim, self.nb_steps)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.nb_xdim):
            plt.subplot(self.nb_xdim + self.nb_udim, 1, k + 1)
            plt.plot(t, self.xdist.mu[k, :], '-b')
            lb = self.xdist.mu[k, :] - 2. * np.sqrt(self.xdist.sigma[k, k, :])
            ub = self.xdist.mu[k, :] + 2. * np.sqrt(self.xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='blue', alpha='0.1')

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.nb_udim):
            plt.subplot(self.nb_xdim + self.nb_udim, 1, self.nb_xdim + k + 1)
            plt.plot(t, self.udist.mu[k, :], '-g')
            lb = self.udist.mu[k, :] - 2. * np.sqrt(self.udist.sigma[k, k, :])
            ub = self.udist.mu[k, :] + 2. * np.sqrt(self.udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='green', alpha='0.1')

        plt.show()

    def objective(self, x, u):
        # summed mean return
        _return = 0.0
        for t in range(self.nb_steps):
            _return += self.env_cost(x[..., t], u[..., t], self.activation[..., t])
        _return += self.env_cost(x[..., -1], np.zeros((self.nb_udim, )), self.activation[..., -1])

        return _return

    def run(self):
        # get linear system dynamics around mean traj.
        self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl, kalman=True)

        # get quadratic cost around mean traj.
        self.cost.finite_diff(self.xdist.mu, self.udist.mu, self.activation)

        # mean objective under current dists.
        ret = self.objective(self.xdist.mu, self.udist.mu)

        # use scipy optimizer
        res = sc.optimize.minimize(self.dual, np.array([-1.e2]),
                                   method='L-BFGS-B',
                                   jac=True,
                                   bounds=((-1e8, -1e-8), ),
                                   options={'disp': False, 'maxiter': 1000,
                                            'ftol': 1e-10})
        self.alpha = res.x

        # re-compute after opt.
        agcost = self.augment_cost(self.alpha)
        lgc, xvalue, xuvalue = self.backward_pass(self.alpha, agcost)
        xdist, udist, xudist = self.forward_pass(lgc)

        # check kl constraint
        kl = self.kldiv(lgc, xdist)

        if (kl - self.nb_steps * self.kl_bound) < 0.1 * self.nb_steps * self.kl_bound:
            # update controller
            self.ctl = lgc
            # update state-action dists.
            self.xdist, self.udist, self.xudist = xdist, udist, xudist
            # update value functions
            self.vfunc, self.qfunc = xvalue, xuvalue

        return ret
