import gym
from trajopt.gps import MBGPS


# double cartpole env
env = gym.make('DoubleCartpole-TO-v0')
env._max_episode_steps = 500

alg = MBGPS(env, nb_steps=500,
            kl_bound=10.,
            init_ctl_sigma=5.0,
            activation={'shift': 450, 'mult': 2.})

# run gps
trace = alg.run()

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
