import gym
from trajopt.ilqr import iLQR

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    env = gym.make('Pendulum-TO-v0')
    env._max_episode_steps = 500

    alg = iLQR(env, nb_steps=500,
               activation=range(450, 500))

    alg.run(nb_iter=100)

    state, action, _ = alg.forward_pass(ctl=alg.ctl, alpha=1.)
    return state[:, :-1].T, action.T


def parallel_ilqr(nb_jobs=50):
    kwargs_list = [{} for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores), verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    obs, act = list(map(list, zip(*results)))
    return obs, act


obs, act = parallel_ilqr(nb_jobs=50)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for _obs, _act in zip(obs, act):
    ax[0].plot(_obs[:, 0])
    ax[1].plot(_obs[:, 1])
    ax[2].plot(_act[:, 0])
plt.show()

import pickle
data = {'obs': obs, 'act': act}
pickle.dump(data, open("ilqr_pendulum_polar.pkl", "wb"))
