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

    # cartpole env
    env = gym.make('Cartpole-TO-v0')
    env._max_episode_steps = 10000
    env.unwrapped.dt = 0.05
    env.unwrapped.umax = np.array([5.])
    env.unwrapped.periodic = True

    dm_state = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]

    horizon, nb_steps = 20, 100

    env_sigma = env.unwrapped.sigma

    state = np.zeros((dm_state, nb_steps + 1))
    action = np.zeros((dm_act, nb_steps))

    state[:, 0] = env.reset()
    for t in range(nb_steps):
        solver = MBGPS(env, init_state=tuple([state[:, t], env_sigma]),
                       init_action_sigma=1., nb_steps=horizon,
                       kl_bound=2., action_penalty=1e-5)
        trace = solver.run(nb_iter=10, verbose=False)

        _act = solver.ctl.sample(state[:, t], 0, stoch=False)
        action[:, t] = np.clip(_act, -env.ulim, env.ulim)
        state[:, t + 1], _, _, _ = env.step(action[:, t])

        print('Time Step:', t, 'Cost:', trace[-1])

    return state[:, :-1].T, action.T


def parallel_gps(nb_jobs=50):
    kwargs_list = [{} for _ in range(nb_jobs)]
    results = Parallel(n_jobs=min(nb_jobs, nb_cores),
                       verbose=10, backend='loky')(map(delayed(create_job), kwargs_list))
    obs, act = list(map(list, zip(*results)))
    return obs, act


obs, act = parallel_gps(nb_jobs=12)

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(12, 4))
for _obs, _act in zip(obs, act):
    ax[0].plot_distributions(_obs[:, 0])
    ax[1].plot_distributions(_obs[:, 1])
    ax[2].plot_distributions(_obs[:, 2])
    ax[3].plot_distributions(_obs[:, 3])
    ax[4].plot_distributions(_act)
plt.show()

# import pickle
# data = {'obs': obs, 'act': act}
# pickle.dump(data, open("gps_cartpole_cart.pkl", "wb"))
