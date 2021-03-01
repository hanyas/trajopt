import autograd.numpy as np

import gym
from trajopt.rgps import MFRGPS
from trajopt.gps import MFGPS

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# pendulum task
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100
env.unwrapped.dt = 0.05
env.unwrapped.umax = 5. * np.ones((1, ))
env.unwrapped.perturb = True

dm_state = env.unwrapped.dm_state
dm_act = env.unwrapped.dm_act

prior = {'K': 1e-6}

kls = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

np.random.seed(1337)
env.seed(1337)

rgps = MFRGPS(env, nb_steps=100,
              policy_kl_bound=1e1,
              param_kl_bound=1e-4,  # 1e1,
              init_state=env.init(),
              init_action_sigma=25.,
              action_penalty=1e-1,
              activation={'mult': 1., 'shift': 80},
              prior=prior)

rgps_trace = rgps.run(nb_learning_episodes=10,
                      nb_evaluation_episodes=100,
                      nb_iter=50, verbose=True)

np.random.seed(1337)
env.seed(1337)

gps = MFGPS(env, nb_steps=100,
            kl_bound=1e1,
            init_state=env.init(),
            init_action_sigma=25.,
            action_penalty=1e-1,
            activation={'mult': 1., 'shift': 80},
            dyn_prior=prior)

# run gps
gps_trace = gps.run(nb_learning_episodes=10,
                    nb_evaluation_episodes=100,
                    nb_iter=50, verbose=True)

np.random.seed(1337)
env.seed(1337)
rgps_data = rgps.rollout(100, stoch=False)
print("Cost of Rbst. Ctl.:", np.mean(np.sum(rgps_data['c'], axis=0)))

np.random.seed(1337)
env.seed(1337)
gps_data = gps.rollout(100, stoch=False)
print("Cost of Nom. Ctl.:", np.mean(np.sum(gps_data['c'], axis=0)))

plt.plot(rgps_trace)
plt.yscale('log')

plt.plot(gps_trace)
plt.yscale('log')

plt.figure(figsize=(8, 12))
plt.suptitle("Robust v Nominal Controller")

for k in range(dm_state):
    plt.subplot(dm_state + dm_act, 2, 2 * k + 1)
    plt.plot(rgps_data['x'][k, ...])

for k in range(dm_act):
    plt.subplot(dm_state + dm_act, 2, 2 * dm_state + 1 + k)
    plt.plot(np.clip(rgps_data['u'], - env.ulim[:, None, None], env.ulim[:, None, None])[k, ...])

for k in range(dm_state):
    plt.subplot(dm_state + dm_act, 2, 2 * (k + 1))
    plt.plot(gps_data['x'][k, ...])

for k in range(dm_act):
    plt.subplot(dm_state + dm_act, 2, 2 * (dm_state + 1) + k)
    plt.plot(np.clip(gps_data['u'], - env.ulim[:, None, None], env.ulim[:, None, None])[k, ...])

plt.show()

plt.figure(figsize=(6, 12))
plt.suptitle("Standard vs Robust Ctl: Feedback Controller")

for i in range(dm_state):
    plt.subplot(dm_state + dm_act, 1, i + 1)
    plt.plot(rgps.ctl.K[0, i, ...], color='r', marker='o', markersize=3)
    plt.plot(gps.ctl.K[0, i, ...], color='k')

for i in range(dm_act):
    plt.subplot(dm_state + dm_act, 1, dm_state + i + 1)
    plt.plot(rgps.ctl.kff[i, ...], color='r', marker='o', markersize=3)
    plt.plot(gps.ctl.kff[i, ...], color='k')

plt.show()
