import autograd.numpy as np

import gym
import matplotlib.pyplot as plt

from trajopt.envs.quanser.cartpole.ctrl import SwingUpCtrl
from trajopt.envs.quanser.common import GentlyTerminating, Logger


def get_angles(sin_theta, cos_theta):
    theta = np.arctan2(sin_theta, cos_theta)
    if theta > 0:
        alpha = (-np.pi + theta)
    else:
        alpha = (np.pi + theta)
    return alpha, theta


class PlotSignal:
    def __init__(self, window=10000):
        self.window = window
        self.values = {}

    def update(self, **argv):
        for k in argv:
            if k not in self.values:
                self.values[k] = [argv[k]]
            else:
                self.values[k].append(argv[k])
            self.values[k] = self.values[k][-self.window:]

    def plot_signal(self):
        N = len(self.values)
        plt.clf()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.pause(0.0000001)

    def last_plot(self):
        N = len(self.values)
        plt.clf()
        plt.ioff()
        for i, k in enumerate(self.values):
            plt.subplot(N, 1, i + 1)
            plt.title(k)

            plt.plot(self.values[k])

        plt.show()


def do_trajectory(env, ctrl, plot, time_steps=10000, use_plot=True,
                  collect_fr=10, plot_fr=10, render=True, render_fr=50):

    obs = env.reset()
    for n in range(time_steps):
        act = ctrl(obs)
        obs, _, done, _ = env.step(np.array(act[0]))

        if done:
            print("Physical Limits or End of Time reached")
            break

        if render:
            if n % render_fr == 0:
                env.render()

        if use_plot:

            if n % collect_fr == 0:
                alpha, theta = get_angles(obs[1], obs[2])
                plot.update(theta=theta, alpha=alpha, theta_dt=obs[4], volt=act[0], u=act[1], x=obs[0])
                env.render()

            if n % plot_fr == 0:
                plot.plot_signal()


def get_env_and_controller():
    env_name = "Quanser-CartpoleRR-v0"
    return Logger(GentlyTerminating(gym.make(env_name))), SwingUpCtrl(long=False)


def main():
    # Initialize Controller & Environment:
    env, ctrl = get_env_and_controller()

    n_episodes = 5
    time_steps = int(15. * 500.)

    state_array = np.zeros((n_episodes, time_steps + 1, 4))
    action_array = np.zeros((n_episodes, time_steps, 1))

    for i in range(n_episodes):
        print("\n\n###############################")
        print("Episode {0}".format(i))

        # Reset the environment:
        env.reset()
        obs, _, _, add_data = env.step(np.zeros(1))
        state_array[i, 0] = add_data['x']
        action_array[i, 0] = add_data['u']

        # Start the Control Loop:
        print("\nStart Controller:\t\t\t", end="")
        for n in range(time_steps):
            act = ctrl(obs)
            obs, _, done, add_data = env.step(act[0] * np.ones(1))
            action_array[i, n] = add_data['u']
            state_array[i, n+1] = add_data['x']

            if done:
                # print("Physical Limits or End of Time reached")
                break

        # Stop the cart:
        env.step(np.zeros(1))
        print("Episode Finished!")
        # time.sleep(3.)

    env.close()


if __name__ == "__main__":
    main()
