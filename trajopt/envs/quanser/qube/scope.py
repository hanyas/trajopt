import autograd.numpy as np
import gym

import trajopt
from trajopt.envs.quanser.qube.ctrl import SwingUpCtrl

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Scope(object):
    def __init__(self, env, ctl, ax, maxt=25, dt=0.01):
        self.env = env
        self.ctl = ctl

        self.ax = ax
        self.nb_plots = len(self.ax)
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [[0] for _ in range(self.nb_plots)]
        self.line = []
        for i in range(self.nb_plots):
            self.line.append(Line2D(self.tdata, self.ydata[i]))

        for i in range(self.nb_plots):
            self.ax[i].add_line(self.line[i])
            self.ax[i].set_xlim(0, self.maxt)

        self.ax[0].set_ylim(-5., 5.)
        self.ax[1].set_ylim(- np.pi, 2. * np.pi)
        self.ax[2].set_ylim(- 30., 30.)
        self.ax[3].set_ylim(- 40., 40.)
        self.ax[4].set_ylim(-10., 10.)

        self.state = self.env.reset()
        self.done = False

    def simulate(self):
        if not self.done:
            act = self.ctl(self.state)
            self.state, _, self.done, _ = env.step(act)
            yield np.hstack((self.state, act))
        else:
            self.done = False
            self.state = self.env.reset()
            yield np.hstack((self.state, np.array([0.])))

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            for i in range(self.nb_plots):
                self.ydata[i] = [self.ydata[i][-1]]
                self.ax[i].set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
                self.ax[i].figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        for i in range(self.nb_plots):
            self.ydata[i].append(y[i])
            self.line[i].set_data(self.tdata, self.ydata[i])
        return self.line


env = gym.make('QQube-v0')
env._max_episode_steps = 2500

ctl = SwingUpCtrl(ref_energy=0.04, energy_gain=35.0, acc_max=5.0)

fig, ax = plt.subplots(5, figsize=(14, 16))
scope = Scope(env, ctl, ax)

# pass a generator in "emitter" to produce data for the update func
ani = animation.FuncAnimation(fig=fig, func=scope.update, frames=scope.simulate, interval=10, blit=True)

plt.show()
