#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: gps.py
# @Date: 2019-06-06-08-56
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import numpy as np

import scipy as sc
from scipy import optimize

from trajopt.gps import core


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

class LinearGaussianControl:
    def __init__(self, nb_xdim, nb_udim, nb_steps, init_ctl_sigma=1.):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        self.K = np.zeros((self.nb_udim, self.nb_xdim, self.nb_steps))
        self.kff = np.zeros((self.nb_udim, self.nb_steps))

        self.sigma = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = init_ctl_sigma * np.eye(self.nb_udim)

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


class GaussianInTime:
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


class QuadraticReward:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim

        self.nb_steps = nb_steps

        self.Rxx = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.rx = np.zeros((self.nb_xdim, self.nb_steps))

        self.Ruu = np.zeros((self.nb_udim, self.nb_udim, self.nb_steps))
        self.ru = np.zeros((self.nb_udim, self.nb_steps))

        self.Rxu = np.zeros((self.nb_xdim, self.nb_udim, self.nb_steps))
        self.r0 = np.zeros((self.nb_steps, ))

    @property
    def params(self):
        return self.Rxx, self.rx, self.Ruu, self.ru, self.Rxu, self.r0

    @params.setter
    def params(self, values):
        self.Rxx, self.rx, self.Ruu, self.ru, self.Rxu, self.r0 = values


class QuadraticStateValue:
    def __init__(self, nb_xdim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_steps = nb_steps

        self.V = np.zeros((self.nb_xdim, self.nb_xdim, self.nb_steps))
        self.v = np.zeros((self.nb_xdim, self.nb_steps, ))
        self.v0 = np.zeros((self.nb_steps, ))
        self.v0_softmax = np.zeros((self.nb_steps, ))


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

        self.q0 = np.zeros((self.nb_steps, ))
        self.q0_common = np.zeros((self.nb_steps, ))
        self.q0_softmax = np.zeros((self.nb_steps, ))


class MBGPS:

    def __init__(self, env, nb_steps, kl_bound, init_ctl_sigma):

        self.env = env

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # total kl over traj.
        self.kl_bound = kl_bound
        self.alpha = np.array([100.])

        # create state distribution and initialize first time step
        self.sdist = GaussianInTime(self.nb_xdim, self.nb_steps + 1)
        for t in range(self.nb_steps + 1):
            self.sdist.mu[..., t], self.sdist.sigma[..., t] = self.env.unwrapped._model.init()

        self.adist = GaussianInTime(self.nb_udim, self.nb_steps)
        self.sadist = GaussianInTime(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        # create and set linear Gaussian dynamics from env
        self.dyn = LinearGaussianDynamics(self.nb_xdim, self.nb_udim, self.nb_steps)

        # create and set quadratic rewards from env
        self.rwrd = QuadraticReward(self.nb_xdim, self.nb_udim, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        self.ctl = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps, init_ctl_sigma)

    def forward_pass(self, lgc):
        sdist = GaussianInTime(self.nb_xdim, self.nb_steps + 1)
        adist = GaussianInTime(self.nb_udim, self.nb_steps)
        sadist = GaussianInTime(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        sdist.mu, sdist.sigma,\
        adist.mu, adist.sigma,\
        sadist.mu, sadist.sigma = core.forward_pass(self.sdist.mu[..., 0], self.sdist.sigma[..., 0],
                                                    self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                    lgc.K, lgc.kff, lgc.sigma,
                                                    self.nb_xdim, self.nb_udim, self.nb_steps)
        return sdist, adist, sadist

    def backward_pass(self, alpha, agrwrd):
        lgc = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        svalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        savalue = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        savalue.Qxx, savalue.Qux, savalue.Quu,\
        savalue.qx, savalue.qu, savalue.q0, savalue.q0_softmax,\
        svalue.V, svalue.v, svalue.v0, svalue.v0_softmax,\
        lgc.K, lgc.kff, lgc.sigma = core.backward_pass(agrwrd.Rxx, agrwrd.rx, agrwrd.Ruu,
                                                       agrwrd.ru, agrwrd.Rxu, agrwrd.r0,
                                                       self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                       alpha, self.nb_xdim, self.nb_udim, self.nb_steps)
        return lgc, svalue, savalue

    def augment_reward(self, alpha):
        agrwrd = QuadraticReward(self.nb_xdim, self.nb_udim, self.nb_steps + 1)
        agrwrd.Rxx, agrwrd.rx, agrwrd.Ruu,\
        agrwrd.ru, agrwrd.Rxu, agrwrd.r0 = core.augment_reward(self.rwrd.Rxx, self.rwrd.rx, self.rwrd.Ruu,
                                                               self.rwrd.ru, self.rwrd.Rxu, self.rwrd.r0,
                                                               self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                               alpha, self.nb_xdim, self.nb_udim, self.nb_steps)
        return agrwrd

    def dual(self, alpha):
        # augmented reward
        agrwrd = self.augment_reward(alpha)

        # backward pass
        lgc, svalue, savalue = self.backward_pass(alpha, agrwrd)

        # forward pass
        sdist, adist, sadist = self.forward_pass(lgc)

        # dual expectation
        dual = core.quad_expectation(sdist.mu[..., 0], sdist.sigma[..., 0],
                                     svalue.V[..., 0], svalue.v[..., 0],
                                     svalue.v0_softmax[..., 0])
        dual += alpha * self.kl_bound

        # gradient
        grad = self.kl_bound - self.kldiv(lgc, sdist)

        return np.array([dual]), np.array([grad])

    def kldiv(self, lgc, sdist):
        return core.kl_divergence(lgc.K, lgc.kff, lgc.sigma,
                                  self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                  sdist.mu, sdist.sigma,
                                  self.nb_xdim, self.nb_udim, self.nb_steps)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()
        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)

        plt.subplot(2, 1, 1)
        plt.plot(t, self.sdist.mu[0, :], '-b')
        lb = self.sdist.mu[0, :] - 2. * np.sqrt(self.sdist.sigma[0, 0, :])
        ub = self.sdist.mu[0, :] + 2. * np.sqrt(self.sdist.sigma[0, 0, :])
        plt.fill_between(t, lb, ub, color='blue', alpha='0.1')

        plt.subplot(2, 1, 2)
        plt.plot(t, self.sdist.mu[1, :], '-r')
        lb = self.sdist.mu[1, :] - 2. * np.sqrt(self.sdist.sigma[1, 1, :])
        ub = self.sdist.mu[1, :] + 2. * np.sqrt(self.sdist.sigma[1, 1, :])
        plt.fill_between(t, lb, ub, color='red', alpha='0.1')

        plt.show()

    def run(self):
        # get linear system dynamics around mean traj.
        self.dyn.params = self.env.unwrapped.model.dyn(self.sdist.mu, self.adist.mu)

        # get quadratic reward around mean traj.
        self.rwrd.params = self.env.unwrapped.model.rwrd(self.sdist.mu, self.adist.mu)

        # current state distribution
        self.sdist, self.adist, self.sadist = self.forward_pass(self.ctl)

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
        lgc, svalue, savalue = self.backward_pass(self.alpha, agrwrd)
        sdist, adist, sadist = self.forward_pass(lgc)

        # check kl constraint
        kl = self.kldiv(lgc, sdist)

        if (kl - self.nb_steps * self.kl_bound) < 0.1 * self.nb_steps * self.kl_bound:
            # update controller
            self.ctl = lgc
            # update state-action dists.
            self.sdist, self.adist, self.sadist = sdist, adist, sadist
            # update value functions
            self.vfunc, self.qfunc = svalue, savalue
        else:
            return
