import gym
from trajopt.ilqr import iLQR

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

alg = iLQR(env, nb_steps=100)

# run iLQR
trace = alg.run()

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
