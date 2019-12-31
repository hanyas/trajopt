import gym
from trajopt.gps import MBGPS

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()

import numpy as np

import warnings
warnings.filterwarnings("ignore")


def create_job(kwargs):
    env = gym.make('Pendulum-TO-v0')
    env._max_episode_steps = 500
    env.unwrapped._dt = 0.01

    alg = MBGPS(env, nb_steps=500,
                kl_bound=0.01,
                init_ctl_sigma=25.0,
                activation={'shift': 450, 'mult': 0.5})

    alg.run(nb_iter=200, verbose=True)

    alg.ctl.sigma = 1e-2 * np.ones_like(alg.ctl.sigma)
    data = alg.sample(nb_episodes=1, stoch=True)

    obs = np.atleast_2d(np.squeeze(data['x'])).T
    act = np.atleast_2d(np.squeeze(data['u'])).T

    return obs, act


def parallel_gps(nb_jobs=50):
    kwargs_list = [{} for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    obs, act = list(map(list, zip(*results)))
    return obs, act


obs, act = parallel_gps(nb_jobs=50)

# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 4))
# for _obs, _act in zip(obs, act):
#     ax[0].plot(_obs[:, :-1])
#     ax[1].plot(_obs[:, -1])
#     ax[2].plot(_act)
# plt.show()

import pickle
data = {'obs': obs, 'act': act}
pickle.dump(data, open("gps_pendulum_polar.pkl", "wb"))
