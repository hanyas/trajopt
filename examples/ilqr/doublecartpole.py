import gym
from trajopt.ilqr import iLQR


# double cartpole env
env = gym.make('DoubleCartpole-TO-v0')
env._max_episode_steps = 500

alg = iLQR(env, nb_steps=500,
           activation={'shift': 450, 'mult': 2.})

# run iLQR
trace = alg.run()

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()

