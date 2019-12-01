import gym
from trajopt.gps import MBGPS


# pendulum env
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 500

alg = MBGPS(env, nb_steps=500,
            kl_bound=0.1,
            init_ctl_sigma=1.0,
            activation=range(450, 500))

# run gps
trace = alg.run(nb_iter=25)

# plot dists
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
