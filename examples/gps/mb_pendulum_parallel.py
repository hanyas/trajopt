import autograd.numpy as np

import gym
from trajopt.gps import MBGPS

from joblib import Parallel, delayed

import multiprocessing
nb_cores = multiprocessing.cpu_count()


def create_job(kwargs):
    import warnings
    warnings.filterwarnings("ignore")

    # pendulum env
    env = gym.make('Pendulum-TO-v0')
    env._max_episode_steps = 10000
    env.unwrapped._dt = 0.01

    dm_state = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    horizon, nb_steps = 50, 500

    state = np.zeros((dm_state, nb_steps + 1))
    action = np.zeros((dm_act, nb_steps))

    state[:, 0] = env.reset()
    for t in range(nb_steps):
        solver = MBGPS(env, init_state=tuple([state[:, t], 1e-16 * np.eye(dm_state)]),
                       init_action_sigma=5., nb_steps=horizon, kl_bound=1.)
        trace = solver.run(nb_iter=10, verbose=False)

        _nominal_action = solver.udist.mu

        action[:, t] = _nominal_action[:, 0]
        state[:, t + 1], _, _, _ = env.step(action[:, t])

        print('Time Step:', t, 'Cost:', trace[-1])

    return state[:, :-1].T, action.T


def parallel_gps(nb_jobs=50):
    kwargs_list = [{} for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores),
                       verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
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
