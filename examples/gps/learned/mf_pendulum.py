import gym
from trajopt.gps import MFGPS

import warnings
warnings.filterwarnings("ignore")


# pendulum task
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100000

alg = MFGPS(env, nb_steps=500,
            kl_bound=250.,
            init_state=env.init(),
            init_action_sigma=25.)

# run gps
trace = alg.run(nb_episodes=25, nb_iter=25, verbose=True)

# plot dists
alg.plot()

# execute and plot
data = alg.sample(25, stoch=True)

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
