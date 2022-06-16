import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    import warnings
    warnings.filterwarnings("ignore")

    # pendulum env
    env = gym.make('Pendulum-TO-v0')
    env._max_episode_steps = 10000
    env.unwrapped.dt = 0.02
    env.unwrapped.umax = np.array([2.5])
    env.unwrapped.periodic = False

    dm_state = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    state = env.reset()
    init_state = tuple([state, 1e-4 * np.eye(dm_state)])
    solver = MBGPS(env, init_state=init_state,
                   init_action_sigma=25., nb_steps=300,
                   kl_bound=.1, action_penalty=1e-3,
                   activation={'shift': 250, 'mult': 0.5})

    solver.run(nb_iter=100, verbose=False)

    solver.ctl.sigma = np.dstack([1e-1 * np.eye(dm_act)] * 300)
    data = solver.rollout(nb_episodes=1, stoch=True, init=state)

    obs, act = np.squeeze(data['x'], axis=-1).T, np.squeeze(data['u'], axis=-1).T
    return obs, act


def parallel_gps(nb_jobs=50):
    kwargs_list = [{} for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, 20),
                       verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    obs, act = list(map(list, zip(*results)))
    return obs, act


obs, act = parallel_gps(nb_jobs=50)

plt.figure()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 4))
for _obs, _act in zip(obs, act):
    ax[0].plot(_obs[:, :-1])
    ax[1].plot(_obs[:, -1])
    ax[2].plot(_act)
plt.show()

import pickle
data = {'obs': obs, 'act': act}
pickle.dump(data, open("gps_pendulum_other.pkl", "wb"))
