import gym
from trajopt.rgps import MFRGPS
from trajopt.gps import MFGPS
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100000

env_sigma = env.unwrapped._sigma
state = env.reset()

init_state = tuple([state, env_sigma])

alg = MFRGPS(env, nb_steps=40,
             kl_bound=5.,
             param_kl_bound=20,
             init_state=init_state,
             init_action_sigma=100.)

alg2 = MFGPS(env, nb_steps=40,
             kl_bound=5.,
             init_state=init_state,
             init_action_sigma=100.)


s = 1837
env.seed(s)
np.random.seed(s)
trace = alg.run(nb_episodes=25, nb_iter=9, verbose=True)


# run gps
s = 1837
env.seed(s)
np.random.seed(s)
trace2 = alg2.run(nb_episodes=25, nb_iter=9, verbose=True)

import copy
rob_ctl = copy.copy(alg.ctl)
nom_ctl = copy.copy(alg2.ctl)


# # plot dists
# alg.plot()
# alg2.plot()

# execute and plot

import matplotlib.pyplot as plt


alg.ctl = rob_ctl
alg2.ctl = rob_ctl
rob_ctl_worst_dyn = np.mean(alg.sample(100, stoch=False, worst=True, use_model=True)['c'],0)
rob_ctl_nominal_dyn = np.mean(alg2.sample(100, stoch=False)['c'],0)

alg.ctl = nom_ctl
alg2.ctl = nom_ctl
nom_ctl_worst_dyn = np.mean(alg.sample(100, stoch=False, worst=True, use_model=True)['c'],0)
nom_ctl_nominal_dyn = np.mean(alg2.sample(100, stoch=False)['c'],0)


plt.figure()
plt.hist(rob_ctl_worst_dyn, density=True, label= "rob_ctl_worst")
plt.hist(nom_ctl_worst_dyn, density=True, label= "nom_ctl_worst")
plt.legend()

plt.figure()
plt.hist(rob_ctl_nominal_dyn, density=True, label= "rob_ctl_nominal")
plt.hist(nom_ctl_nominal_dyn, density=True, label= "nom_ctl_nominal")
plt.legend()


# for k in range(alg.dm_state):
#     plt.subplot(alg.dm_state + alg.dm_act, 1, k + 1)
#     plt.plot(data['x'][k, ...])

# for k in range(alg.dm_act):
#     plt.subplot(alg.dm_state + alg.dm_act, 1, alg.dm_state + k + 1)
#     plt.plot(data['u'][k, ...])

plt.show()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()

