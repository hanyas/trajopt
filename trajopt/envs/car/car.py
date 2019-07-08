#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: ndrobot
# @Date: 2019-06-25-09-09
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class Car(gym.Env):

    def __init__(self):
        self.nb_xdim = 4
        self.nb_bdim = 4
        self.nb_udim = 2
        self.nb_zdim = 2

        self._dt = 0.5
        self._g = np.array([0., 0., 0., 0.])

        # car length
        self._l = 0.1

        # belief cost weights
        self._bw = np.array([1., 1., 1., 1.])
        self._vw = np.array([1., 1., 1., 1.])
        # action cost weights
        self._uw = np.array([1., 1.])

        self._xmax = np.array([np.inf, np.inf, np.inf, np.inf])
        self._zmax = np.array([np.inf, np.inf, np.inf, np.inf])
        self._umax = np.array([np.inf, np.inf])

        self._state_space = spaces.Box(low=-self._xmax,
                                      high=self._xmax)

        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax)

        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self.seed()
        self.reset()

    @property
    def state_space(self):
        return self._state_space

    @property
    def xlim(self):
        return self._xmax

    @property
    def ulim(self):
        return self._umax

    @property
    def dt(self):
        return self._dt

    @property
    def goal(self):
        return self._g

    def init(self):
        # initial belief
        _b0 = np.array([2., 2., 0., 0.])
        _sigma_b0 = 1. * np.eye(self.nb_bdim)
        return _b0, _sigma_b0

    def dynamics(self, x, u):
        _u = np.clip(u, -self._umax, self._umax)
        # x, y, th, v
        xn = x + self._dt * np.array([x[3] * np.cos(x[2]),
                                      x[3] * np.sin(x[2]),
                                      x[3] * np.tan(_u[1]) / self._l,
                                      _u[0]])
        xn = np.clip(xn, -self._xmax, self._xmax)
        return xn

    def dyn_noise(self, x=None, u=None):
        _u = np.clip(u, -self._umax, self._umax)
        _x = np.clip(x, -self._xmax, self._xmax)
        return 1.e-4 * np.eye(self.nb_xdim)

    def observe(self, x):
        return np.array([x[0], x[1]])

    def obs_noise(self, x=None):
        _sigma = 1.e-4 * np.eye(self.nb_zdim)
        _sigma += np.array([[0.5 * (5. - x[0])**2, 0.],
                            [0., 0.]])
        return _sigma

    # cost defined over belief
    def cost(self, mu_b, sigma_b, u, a):
        if a:
            return (mu_b - self._g).T @ np.diag(100. * self._bw) @ (mu_b - self._g) +\
                   np.trace(np.diag(100. * self._vw) @ sigma_b)
        else:
            return u.T @ np.diag(self._uw) @ u + np.trace(np.diag(self._vw) @ sigma_b)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # state-action dependent dynamics noise
        _sigma_dyn = self.dyn_noise(self.state, u)
        # evolve deterministic dynamics
        self.state = self.dynamics(self.state, u)
        # add dynamics noise
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=_sigma_dyn)

        # state-action dependent dynamics noise
        _sigma_obs = self.obs_noise(self.state)
        # observe state
        _z = self.observe(self.state)
        # add observation noise
        _z = self.np_random.multivariate_normal(mean=_z, cov=_sigma_obs)
        return _z, [], False, {}

    def reset(self):
        self.state = np.array([0., 4., 0., 0.])
        return self.observe(self.state)
