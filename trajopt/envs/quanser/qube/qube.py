import autograd.numpy as np
from autograd import jacobian
from autograd.tracer import getval

import gym
from gym import spaces
from gym.utils import seeding

from trajopt.envs.quanser.qube.base import QubeBase, QubeDynamics, ActionLimiter


class Qube(QubeBase):
    def __init__(self, fs, fs_ctrl):
        super(Qube, self).__init__(fs, fs_ctrl)
        self.dyn = QubeDynamics()
        self._sim_state = None
        self._vis = {'vp': None, 'arm': None, 'pole': None, 'curve': None}

    def _set_gui(self):
        scene_range = 0.2
        arm_radius = 0.003
        arm_length = 0.085
        pole_radius = 0.0045
        pole_length = 0.129
        # http://www.glowscript.org/docs/VPythonDocs/canvas.html
        self._vis['vp'].scene.width = 400
        self._vis['vp'].scene.height = 300
        self._vis['vp'].scene.background = self._vis['vp'].color.gray(0.95)
        self._vis['vp'].scene.lights = []
        self._vis['vp'].distant_light(
            direction=self._vis['vp'].vector(0.2, 0.2, 0.5),
            color=self._vis['vp'].color.white)
        self._vis['vp'].scene.up = self._vis['vp'].vector(0, 0, 1)
        self._vis['vp'].scene.range = scene_range
        self._vis['vp'].scene.center = self._vis['vp'].vector(0.04, 0, 0)
        self._vis['vp'].scene.forward = self._vis['vp'].vector(-2, 1.2, -1)
        self._vis['vp'].box(pos=self._vis['vp'].vector(0, 0, -0.07),
                            length=0.09, width=0.1, height=0.09,
                            color=self._vis['vp'].color.gray(0.5))
        self._vis['vp'].cylinder(
            axis=self._vis['vp'].vector(0, 0, -1), radius=0.005,
            length=0.03, color=self._vis['vp'].color.gray(0.5))
        # Arm
        arm = self._vis['vp'].cylinder()
        arm.radius = arm_radius
        arm.length = arm_length
        arm.color = self._vis['vp'].color.blue
        # Pole
        pole = self._vis['vp'].cylinder()
        pole.radius = pole_radius
        pole.length = pole_length
        pole.color = self._vis['vp'].color.red
        # Curve
        curve = self._vis['vp'].curve(color=self._vis['vp'].color.white,
                                      radius=0.0005, retain=2000)
        return arm, pole, curve

    def _calibrate(self):
        _low, _high = np.array([-0.1, - np.pi / 36., -0.1, -0.1]),\
                      np.array([0.1, np.pi / 36., 0.1, 0.1])
        self._sim_state = self._np_random.uniform(low=_low, high=_high)
        self._state = self._zero_sim_step()

    def _sim_step(self, u):
        # add action noise
        u = u + np.random.randn(1) * 1e-2

        u_cmd = self._lim_act(self._sim_state, u)
        # u_cmd = np.clip(u, self.action_space.low, self.action_space.high)

        thdd, aldd = self.dyn(self._sim_state, u_cmd)

        # Update internal simulation state
        self._sim_state[3] += self.timing.dt * aldd
        self._sim_state[2] += self.timing.dt * thdd
        self._sim_state[1] += self.timing.dt * self._sim_state[3]
        self._sim_state[0] += self.timing.dt * self._sim_state[2]

        # apply state constraints
        self._sim_state = np.clip(self._sim_state, self.observation_space.low, self.observation_space.high)

        # add observation noise
        self._sim_state = self._sim_state + np.random.randn(4) * 1e-4

        return self._sim_state, u

    def reset(self):
        self._calibrate()
        if self._vis['curve'] is not None:
            self._vis['curve'].clear()
        return self.step(np.array([0.0]))[0]

    def render(self, mode='human'):
        if self._vis['vp'] is None:
            import importlib
            self._vis['vp'] = importlib.import_module('vpython')
            self._vis['arm'],\
            self._vis['pole'],\
            self._vis['curve'] = self._set_gui()
        th, al, _, _ = self._state
        arm_pos = (self.dyn.Lr * np.cos(th), self.dyn.Lr * np.sin(th), 0.0)
        pole_ax = (-self.dyn.Lp * np.sin(al) * np.sin(th),
                   self.dyn.Lp * np.sin(al) * np.cos(th),
                   self.dyn.Lp * np.cos(al))
        self._vis['arm'].axis = self._vis['vp'].vector(*arm_pos)
        self._vis['pole'].pos = self._vis['vp'].vector(*arm_pos)
        self._vis['pole'].axis = self._vis['vp'].vector(*pole_ax)
        self._vis['curve'].append(
            self._vis['pole'].pos + self._vis['pole'].axis)
        self._vis['vp'].rate(self.timing.render_rate)


class QubeTO(gym.Env):

    def __init__(self):
        self.dm_state = 4
        self.dm_act = 1

        self._dt = 0.01

        self._sigma = 1e-8 * np.eye(self.dm_state)

        self.dyn = QubeDynamics()

        # g = [th, al, thd, ald]
        self._g = np.array([0., np.pi, 0., 0.])
        self._gw = np.array([1e-1, 1e0, 1e-2, 1e-3])

        # x = [x, th, dx, dth]
        self._xmax = np.array([2.3, np.inf, 30., 40.])
        self.observation_space = spaces.Box(low=-self._xmax,
                                            high=self._xmax)

        self._uw = np.array([1e-3])
        self._umax = 10.0
        self.action_space = spaces.Box(low=-self._umax,
                                       high=self._umax, shape=(1,))

        safety_th_lim = 1.5
        self._lim_act = ActionLimiter(self.observation_space,
                                      self.action_space,
                                      safety_th_lim)

        self.state = None
        self.np_random = None

        self.seed()

        _low, _high = np.array([-0.1, - np.pi / 18., -0.1, -0.1]),\
                      np.array([0.1, np.pi / 18., 0.1, 0.1])
        self._x0 = self.np_random.uniform(low=_low, high=_high)
        self._sigma_0 = 1e-4 * np.eye(self.dm_state)

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
        return self._x0, self._sigma_0

    def dynamics(self, x, u):
        _u = self._lim_act(x, u)
        # _u = np.clip(u, -self._umax, self._umax)

        def f(x, u):
            thdd, aldd = self.dyn(x, u)
            return np.hstack((x[2], x[3], thdd, aldd))

        k1 = f(x, _u)
        k2 = f(x + 0.5 * self.dt * k1, _u)
        k3 = f(x + 0.5 * self.dt * k2, _u)
        k4 = f(x + self.dt * k3, _u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        xn = np.clip(xn, -self._xmax, self._xmax)
        return xn

    def features(self, x):
        return x

    def features_jacobian(self, x):
        _J = jacobian(self.features, 0)
        _j = self.features(x) - _J(x) @ x
        return _J, _j

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self._umax, self._umax)
        _x = np.clip(x, -self._xmax, self._xmax)
        return self._sigma

    def cost(self, x, u, a):
        _J, _j = self.features_jacobian(getval(x))
        _x = _J(getval(x)) @ x + _j
        return a * (_x - self._g).T @ np.diag(self._gw) @ (_x - self._g) + u.T @ np.diag(self._uw) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # state-action dependent noise
        _sigma = self.noise(self.state, u)
        # evolve deterministic dynamics
        self.state = self.dynamics(self.state, u)
        # add noise
        self.state = self.np_random.multivariate_normal(mean=self.state, cov=_sigma)
        return self.state, [], False, {}

    def reset(self):
        _mu_0, _sigma_0 = self.init()
        self.state = self.np_random.multivariate_normal(mean=_mu_0, cov=_sigma_0)
        return self.state


class QubeTOWithCartesianCost(QubeTO):

    def __init__(self):
        super(QubeTOWithCartesianCost, self).__init__()

        # g = [th, cs_al, sn_al, dth, dal]
        self._g = np.array([0., -1., 0., 0., 0.])
        self._gw = np.array([1e-1, 1e0, 0., 1e-2, 1e-3])

    def features(self, x):
        return np.array([x[0],
                        np.cos(x[1]), np.sin(x[1]),
                        x[2], x[3]])
