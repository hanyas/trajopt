import autograd.numpy as np

import gym
from trajopt.rgps import MFRGPS

import warnings
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed


def create_job(kwargs):
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(kwargs['seed'])

    # pendulum task
    env = gym.make('Pendulum-TO-v0')
    env._max_episode_steps = 100
    env.unwrapped.dt = 0.05

    env.seed(kwargs['seed'])

    prior = {'K': 1e-3}

    alg = MFRGPS(env, nb_steps=100,
                 policy_kl_bound=5.,
                 param_kl_bound=kwargs['param_kl_bound'],
                 init_state=env.init(),
                 init_action_sigma=5.,
                 action_penalty=np.array([1e-5]),
                 slew_rate=False,
                 prior=prior)

    trace = alg.run(nb_episodes=10, nb_iter=5, verbose=False)
    return trace


def parallel_rgos(params):
    kwargs_list = [{'param_kl_bound': param, 'seed': 1337} for param in params]
    traces = Parallel(n_jobs=len(params), verbose=10, backend='loky')\
                        (map(delayed(create_job), kwargs_list))
    return traces


kls = [1e3, 5e2, 1e1, 1e-1]
traces = parallel_rgos(params=kls)

import matplotlib.pyplot as plt

# plot objective
plt.figure()
for trace, kl in zip(traces, kls):
    plt.plot(trace, label=str(kl))

plt.legend()
plt.show()
