import gym
from trajopt.ilqr import iLQR

import warnings
warnings.filterwarnings("ignore")


# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 500

alg = iLQR(env, nb_steps=500,
           activation=range(450, 500))

# run iLQR
trace = alg.run(nb_iter=10)

# plot reference trajectory
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()

state, action, _ = alg.forward_pass(ctl=alg.ctl, alpha=1.)

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(state[0, :], '-b')
plt.subplot(3, 1, 2)
plt.plot(state[1, :], '-b')

plt.subplot(3, 1, 3)
plt.plot(action[0, :], '-g')

plt.show()