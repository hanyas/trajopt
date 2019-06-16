#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: gps.py
# @Date: 2019-06-06-08-56
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import numpy as np

import scipy as sc
from scipy import optimize

from mimo import distributions, models
from mimo.util.text import progprint_xrange

from gps import core


class LinearGaussianDynamics:
    def __init__(self, nb_xdim, nb_udim, nb_steps):
        self.nb_xdim = nb_xdim
        self.nb_udim = nb_udim
        self.nb_steps = nb_steps

        # GMM model
        nb_models = 3

        gating_hypparams = dict(K=nb_models, alphas=np.ones((nb_models,)))
        gating_prior = distributions.Dirichlet(**gating_hypparams)

        components_hypparams = dict(M=np.zeros((self.nb_xdim, self.nb_xdim + self.nb_udim + 1)),
                                    V=1. * np.eye(self.nb_xdim + self.nb_udim + 1),
                                    affine=True,
                                    psi=np.eye(self.nb_xdim),
                                    nu=2 * self.nb_xdim + 1)
        components_prior = distributions.MatrixNormalInverseWishart(**components_hypparams)

        self._gmm = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                   components=[distributions.BayesianLinearGaussian(components_prior) for _ in range(nb_models)])

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

    def update_model(self, data):
        # reshape for inference
        _x = data['x'].reshape(-1, self.nb_xdim)
        _u = data['u'].reshape(-1, self.nb_udim)
        _xn = data['xn'].reshape(-1, self.nb_xdim)
        _data = np.hstack((_x, _u, _xn))

        self.update_prior(_data)

        for t in range(self.nb_steps):
            _query = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T, data['xn'][:, t, :].T))
            _prior = self.mean_prior(_query)
            _model = distributions.BayesianLinearGaussian(prior=_prior)
            _model.MAP(_query)
            self.A[..., t] = _model.A[:, :self.nb_xdim]
            self.B[..., t] = _model.A[:, self.nb_xdim:self.nb_xdim + self.nb_udim]
            self.c[..., t] = _model.A[:, -1]
            self.sigma[..., t] = _model.sigma

    def update_prior(self, data, nb_iter=500):
        self._gmm.add_data(data)
        for _ in progprint_xrange(nb_iter):
            self._gmm.resample_model()

    def query_prior(self, data):
        _data = np.atleast_2d(data)
        return self._gmm.labels_list[-1].get_responsibility(_data)

    def mean_prior(self, data):
        resp = self.query_prior(data)
        r = np.mean(resp, axis=0)
        hypparams = self._gmm.mean_posterior(weights=r)
        return distributions.MatrixNormalInverseWishart(M=hypparams[0], V=hypparams[1], affine=True,
                                                        psi=hypparams[2], nu=hypparams[3])


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

    def set(self, values, stationary=True):
        if stationary:
            for t in range(self.nb_steps):
                self.Rxx[..., t], self.rx[..., t], self.Ruu[..., t],\
                self.ru[..., t], self.Rxu[..., t], self.r0[..., t] = values
        else:
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


class MFGPS:

    def __init__(self, env, nb_episodes, nb_steps,
                 kl_bound, init_ctl_sigma):

        self.env = env
        self.alim = self.env.action_space.high

        self.nb_xdim = self.env.observation_space.shape[0]
        self.nb_udim = self.env.action_space.shape[0]

        self.nb_episodes = nb_episodes
        self.nb_steps = nb_steps

        self.kl_bound = kl_bound
        self.alphas = 100. * np.ones((self.nb_steps, ))

        self.sdist = GaussianInTime(self.nb_xdim, self.nb_steps + 1)
        self.sdist.mu[..., 0], self.sdist.sigma[..., 0] = self.env.unwrapped._model.init()

        self.adist = GaussianInTime(self.nb_udim, self.nb_steps)
        self.sadist = GaussianInTime(self.nb_xdim + self.nb_udim, self.nb_steps + 1)

        self.dyn = LinearGaussianDynamics(self.nb_xdim, self.nb_udim, self.nb_steps)
        self.vfunc = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)
        self.rwrd = QuadraticReward(self.nb_xdim, self.nb_udim, self.nb_steps + 1)

        self.ctl = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps, init_ctl_sigma)

        self.data = {}

    def sample(self, nb_episodes, nb_steps, stoch=True):
        data = {'x': np.zeros((self.nb_xdim, nb_steps, nb_episodes)),
                'u': np.zeros((self.nb_udim, nb_steps, nb_episodes)),
                'xn': np.zeros((self.nb_xdim, nb_steps, nb_episodes))}

        for n in range(nb_episodes):
            x = self.env.reset()

            for t in range(nb_steps):
                u = self.ctl.sample(x, t - 1, stoch)

                data['u'][..., t, n] = u
                data['x'][..., t, n] = x

                x, _, _, _ = self.env.step(np.clip(u, - self.alim, self.alim))

                data['xn'][..., t, n] = x

        return data

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

    def backward_pass(self, alphas, agrwrd):
        lgc = LinearGaussianControl(self.nb_xdim, self.nb_udim, self.nb_steps)
        svalue = QuadraticStateValue(self.nb_xdim, self.nb_steps + 1)
        savalue = QuadraticStateActionValue(self.nb_xdim, self.nb_udim, self.nb_steps)

        savalue.Qxx, savalue.Qux, savalue.Quu,\
        savalue.qx, savalue.qu, savalue.q0, savalue.q0_softmax,\
        svalue.V, svalue.v, svalue.v0, svalue.v0_softmax,\
        lgc.K, lgc.kff, lgc.sigma = core.backward_pass(agrwrd.Rxx, agrwrd.rx, agrwrd.Ruu,
                                                       agrwrd.ru, agrwrd.Rxu, agrwrd.r0,
                                                       self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                       alphas, self.nb_xdim, self.nb_udim, self.nb_steps)
        return lgc, svalue, savalue

    def augment_reward(self, alphas):
        agrwrd = QuadraticReward(self.nb_xdim, self.nb_udim, self.nb_steps + 1)
        agrwrd.Rxx, agrwrd.rx, agrwrd.Ruu,\
        agrwrd.ru, agrwrd.Rxu, agrwrd.r0 = core.augment_reward(self.rwrd.Rxx, self.rwrd.rx, self.rwrd.Ruu,
                                                               self.rwrd.ru, self.rwrd.Rxu, self.rwrd.r0,
                                                               self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                               alphas, self.nb_xdim, self.nb_udim, self.nb_steps)
        return agrwrd

    def dual(self, alphas):
        # augmented reward
        agrwrd = self.augment_reward(alphas)

        # backward pass
        lgc, svalue, savalue = self.backward_pass(alphas, agrwrd)

        # forward pass
        sdist, adist, sadist = self.forward_pass(lgc)

        # dual expectation
        dual = core.quad_expectation(sdist.mu[..., 0], sdist.sigma[..., 0],
                                     svalue.V[..., 0], svalue.v[..., 0],
                                     svalue.v0_softmax[..., 0])
        dual += np.sum(alphas * self.kl_bound)

        return np.array([dual])

    def grad(self, alphas):
        # augmented reward
        agrwrd = self.augment_reward(alphas)

        # backward pass
        lgc, svalue, savalue = self.backward_pass(alphas, agrwrd)

        # forward pass
        sdist, adist, sadist = self.forward_pass(lgc)

        # gradient
        grad = self.kl_bound - self.kldiv(lgc, sdist)

        return grad

    def kldiv(self, lgc, sdist):
        return core.kl_divergence(lgc.K, lgc.kff, lgc.sigma,
                                  self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                  sdist.mu, sdist.sigma,
                                  self.nb_xdim, self.nb_udim, self.nb_steps)

    def run(self, nb_episodes, nb_steps):
        # run current controller
        self.data = self.sample(nb_episodes, nb_steps)

        # plot state dist.
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.data['x'][0, ...])
        plt.show()

        # fit time-variant linear dynamics
        self.dyn.update_model(self.data)

        # use scipy optimizer
        res = sc.optimize.minimize(self.dual, self.alphas,
                                   method='L-BFGS-B',
                                   jac=self.grad,
                                   bounds=((1e-8, 1e8), ) * self.nb_steps,
                                   options={'disp': True, 'maxiter': 500, 'ftol': 1e-12})
        self.alphas = res.x[0:self.nb_steps]

        # re-compute after opt.
        agrwrd = self.augment_reward(self.alphas)
        lgc, svalue, savalue = self.backward_pass(self.alphas, agrwrd)
        sdist, adist, sadist = self.forward_pass(lgc)

        # check kl constraint
        kl = self.kldiv(lgc, sdist).sum()
        print("kl: ", kl)

        if (kl - self.nb_steps * self.kl_bound) < 0.1 * self.nb_steps * self.kl_bound:
            # update controller
            self.ctl = lgc
            # update state-action dists.
            self.sdist, self.adist, self.sadist = sdist, adist, sadist
            # update value functions
            self.vfunc, self.qfunc = svalue, savalue
        else:
            self.alphas = 1. * np.ones((self.nb_steps, ))


if __name__ == "__main__":

    import gym

    env = gym.make('LQR-v0')
    env._max_episode_steps = 100

    gps = MFGPS(env, nb_episodes=10, nb_steps=100,
                kl_bound=0.1, init_ctl_sigma=1.e-1)

    # run gps
    for _ in range(1):
        gps.run(nb_episodes=10, nb_steps=1000)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(gps.sdist.mu.T)
    plt.subplot(2, 1, 2)
    plt.plot(gps.adist.mu.T)
    plt.show()

    # # kl sanity check
    # sdist = GaussianInTime(nb_dim=2, nb_steps=100)
    # kl = gps.kldiv(gps.ctl, sdist)
    # print(np.sum(kl))
