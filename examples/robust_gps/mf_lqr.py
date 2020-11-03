import gym
from trajopt.robust_gps import MFROBGPS
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100000

env_sigma = env.unwrapped._sigma
state = env.reset()

s = 1
env.seed(s)
np.random.seed(s)

init_state = tuple([state, env_sigma])

alg = MFROBGPS(env, nb_steps=60,
            kl_bound=5.,
            param_kl_bound=10.,
            init_state=init_state,
            init_action_sigma=100.)



# run gps
trace1 = alg.run(nb_episodes=25, nb_iter=10, verbose=True)

alg.plot()

trace2 = alg.run(nb_episodes=25, nb_iter=10, verbose=True)

# plot dists
alg.plot()

# execute and plot
data = alg.sample(25, stoch=False)

import matplotlib.pyplot as plt

plt.figure()
for k in range(alg.dm_state):
    plt.subplot(alg.dm_state + alg.dm_act, 1, k + 1)
    plt.plot(data['x'][k, ...])

for k in range(alg.dm_act):
    plt.subplot(alg.dm_state + alg.dm_act, 1, alg.dm_state + k + 1)
    plt.plot(data['u'][k, ...])

plt.show()

# plot objective
plt.figure()
plt.plot(trace)
plt.show()

import matplotlib.pyplot as plt
alphas = np.logspace(0,8,100)
dl = [self.parameter_dual(alpha)[0] for alpha in alphas]
plt.plot(alphas, dl)
plt.xscale('log')
plt.show()


mu_param, sigma_param, xvalue, diverge = self.parameter_backward_pass(-1e10)
kl = self.parameter_kldiv(mu_param, sigma_param)
plt.plot(kl)
plt.show()

for a in [10, 100, 1000, 10000]:
    mu_param, sigma_param, xvalue, diverge = self.parameter_backward_pass(-a)
    kl = self.parameter_kldiv(mu_param, sigma_param)
    plt.plot(kl, label=a)
plt.legend()
plt.show()
