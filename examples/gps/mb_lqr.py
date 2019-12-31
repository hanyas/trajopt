import gym
from trajopt.gps import MBGPS


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 60

alg = MBGPS(env, nb_steps=60,
            kl_bound=100.,
            init_ctl_sigma=100.)

# run gps
trace = alg.run()

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
