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


class LightDark(gym.Env):

    def __init__(self):
        self.nb_xdim = 2
        self.nb_udim = 2
        self.nb_zdim = 2

        self._dt = 0.05
        self._g = np.array([0., 0.])

        self._xw = - np.array([0.5, 0.5])
        self._uw = - np.array([0.5, 0.5])
        self._vw = - np.array([200., 200.])

        self._xmax = np.array([7., 4.])
        self._zmax = np.array([7., 4.])
        self._umax = np.array([np.inf, np.inf])

        self._state_space = spaces.Box(low=-self._xmax,
                                      high=self._xmax)

        self.observation_space = spaces.Box(low=-self._zmax,
                                            high=self._zmax)

        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax)

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
        self.state = np.array([2.5, 0.])
        _z0 = np.array([2., 2.])
        _sigma_z0 = 5. * np.eye(self.nb_zdim)
        return _z0, _sigma_z0

    def dynamics(self, x, u):
        return x + self._dt * u

    def dyn_noise(self, x=None, u=None):
        return 1.e-8 * np.eye(self.nb_xdim)

    def observe(self, x):
        return x

    def obs_noise(self, x=None, u=None):
        _sigma = 1e-2 * np.eye(self.nb_zdim)
        _sigma[0, 0] += 0.5 * (5. - x[0])**2
        _sigma[1, 1] += 0.5 * (2. - x[1])**2
        return _sigma

    def cost(self, z, sigma, u, a):
        if a:
            return (z - self._g).T @ np.diag(self._xw) @ (z - self._g) +\
                    np.trace(np.diag(self._vw) @ sigma) +\
                    u.T @ np.diag(self._uw) @ u
        else:
            return u.T @ np.diag(self._uw) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # state-action dependent dynamics noise
        _sigma_x = self.dyn_noise(self.state, u)
        # evolve deterministic dynamics
        self.state = self.dynamics(self.state, u)
        # add noise
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=_sigma_x)

        # state-action dependent dynamics noise
        _sigma_z = self.obs_noise(self.state, u)
        # observe state
        _z = self.observe(self.state)
        # add noise
        _z = self.np_random.multivariate_normal(mean=_z, cov=_sigma_z)
        return _z, [], False, {}

    def reset(self):
        _z0, _sigma_z0 = self.init()
        return _z0, _sigma_z0
