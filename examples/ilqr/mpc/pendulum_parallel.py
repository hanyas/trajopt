import autograd.numpy as np

import gym
from trajopt.ilqr import iLQR

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    import warnings
    warnings.filterwarnings("ignore")

    # pendulum env
    env = gym.make('Pendulum-TO-v1')
    env._max_episode_steps = 100000
    env.unwrapped._dt = 0.05

    dm_state = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    horizon, nb_steps = 15, 150
    state = np.zeros((dm_state, nb_steps + 1))
    action = np.zeros((dm_act, nb_steps))

    state[:, 0] = env.reset()
    for t in range(nb_steps):
        solver = iLQR(env, init_state=state[:, t],
                      init_action=None, nb_steps=horizon)
        solver.run(nb_iter=10, verbose=False)

        _nominal_action = solver.uref
        action[:, t] = _nominal_action[:, 0]
        state[:, t + 1], _, _, _ = env.step(action[:, t])

    return state[:, :-1].T, action.T


def parallel_ilqr(nb_jobs=50):
    kwargs_list = [{} for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores),
                       verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
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
