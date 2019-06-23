#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: gps.py
# @Date: 2019-06-06-08-56
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

import scipy as sc
from scipy import optimize

from trajopt.gps.objects import GaussianInTime, QuadraticReward
from trajopt.gps.objects import AnalyticalLinearGaussianDynamics, AnalyticalQuadraticReward
from trajopt.gps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.gps.objects import LinearGaussianControl

from trajopt.gps.core import kl_divergence, quad_expectation, augment_reward
from trajopt.gps.core import forward_pass, backward_pass


class MBGPS:

    def __init__(self, env, nb_steps, kl_bound, init_ctl_sigma):

        self.env = env

        self.env_dyn = self.env.unwrapped.dynamics
        self.env_sigma = self.env.unwrapped.sigma

        self.env_rwrd = self.env.unwrapped.reward
        self.env_init = self.env.unwrapped.initialize

        self.alim = self.env.action_space.high

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # total kl over traj.
        self.kl_bound = kl_bound
        self.alpha = np.array([100.])

        # create state distribution and initialize first time step
        self.xdist = GaussianInTime(self.nb_xdim, self.nb_steps + 1)
        for t in range(self.nb_steps + 1):
            self.xdist.mu[..., t], self.xdist.sigma[..., t] = self.env_init()

        self.udist = GaussianInTime(self.nb_udim, self.nb_steps)
        self.xudist = GaussianInTime(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.dyn = AnalyticalLinearGaussianDynamics(self.env_dyn, self.env_sigma, self.nb_xdim, self.nb_udim, self.nb_steps)
        self.rwrd = AnalyticalQuadraticReward(self.env_rwrd, self.nb_xdim, self.nb_udim, self.nb_steps + 1)
        self.ctl = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps, init_ctl_sigma)

    def sample(self, nb_episodes, stoch=True):
        data = {'x': np.zeros((self.nb_xdim, self.nb_steps, nb_episodes)),
                'u': np.zeros((self.nb_udim, self.nb_steps, nb_episodes)),
                'xn': np.zeros((self.nb_xdim, self.nb_steps, nb_episodes))}

        for n in range(nb_episodes):
            x = self.env.reset()

            for t in range(self.nb_steps):
                u = self.ctl.sample(x, t, stoch)

                data['u'][..., t, n] = u
                data['x'][..., t, n] = x

                x, _, _, _ = self.env.step(np.clip(u, - self.alim, self.alim))

                data['xn'][..., t, n] = x

        return data

    def forward_pass(self, lgc):
        xdist = GaussianInTime(self.nb_xdim, self.nb_steps + 1)
        udist = GaussianInTime(self.nb_udim, self.nb_steps)
        xudist = GaussianInTime(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        xdist.mu, xdist.sigma,\
        udist.mu, udist.sigma,\
        xudist.mu, xudist.sigma = forward_pass(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                               self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                               lgc.K, lgc.kff, lgc.sigma,
                                               self.nb_xdim, self.nb_udim, self.nb_steps)
        return xdist, udist, xudist

    def backward_pass(self, alpha, agrwrd):
        lgc = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        xvalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu, xuvalue.q0, xuvalue.q0_softmax,\
        xvalue.V, xvalue.v, xvalue.v0, xvalue.v0_softmax,\
        lgc.K, lgc.kff, lgc.sigma = backward_pass(agrwrd.Rxx, agrwrd.rx, agrwrd.Ruu,
                                                  agrwrd.ru, agrwrd.Rxu, agrwrd.r0,
                                                  self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                  alpha, self.nb_xdim, self.nb_udim, self.nb_steps)
        return lgc, xvalue, xuvalue

    def augment_reward(self, alpha):
        agrwrd = QuadraticReward(self.nb_xdim, self.nb_udim, self.nb_steps + 1)
        agrwrd.Rxx, agrwrd.rx, agrwrd.Ruu,\
        agrwrd.ru, agrwrd.Rxu, agrwrd.r0 = augment_reward(self.rwrd.Rxx, self.rwrd.rx, self.rwrd.Ruu,
                                                          self.rwrd.ru, self.rwrd.Rxu, self.rwrd.r0,
                                                          self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                          alpha, self.nb_xdim, self.nb_udim, self.nb_steps)
        return agrwrd

    def dual(self, alpha):
        # augmented reward
        agrwrd = self.augment_reward(alpha)

        # backward pass
        lgc, xvalue, xuvalue = self.backward_pass(alpha, agrwrd)

        # forward pass
        xdist, udist, xudist = self.forward_pass(lgc)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0_softmax[..., 0])
        dual += alpha * self.kl_bound

        # gradient
        grad = self.kl_bound - self.kldiv(lgc, xdist)

        return np.array([dual]), np.array([grad])

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

    def run(self):
        # get linear system dynamics around mean traj.
        self.dyn.diff(self.xdist.mu, self.udist.mu)

        # get quadratic reward around mean traj.
        self.rwrd.diff(self.xdist.mu, self.udist.mu)

        # current state distribution
        self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl)

        # use scipy optimizer
        res = sc.optimize.minimize(self.dual, np.array([1.e2]),
                                   method='L-BFGS-B',
                                   jac=True,
                                   bounds=((1e-8, 1e8), ),
                                   options={'disp': False, 'maxiter': 1000,
                                            'ftol': 1e-10})
        self.alpha = res.x

        # re-compute after opt.
        agrwrd = self.augment_reward(self.alpha)
        lgc, xvalue, xuvalue = self.backward_pass(self.alpha, agrwrd)
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
        else:
            return
