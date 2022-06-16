import numpy as np

import gym
from trajopt.bspilqr import BSPiLQR

from trajopt.bspilqr.objects import EKF


# car task
env = gym.make('Car-TO-v0')
env._max_episode_steps = 10000

nb_steps = 25

state_dim = env.state_dim
belief_dim = env.belief_dim
obs_dim = env.obs_dim
act_dim = env.act_dim

mu_b = np.zeros((belief_dim, nb_steps + 1))
sigma_b = np.zeros((belief_dim, belief_dim, nb_steps + 1))

state = np.zeros((state_dim, nb_steps + 1))
obs = np.zeros((obs_dim, nb_steps + 1))
act = np.zeros((act_dim, nb_steps))

obs[:, 0] = env.reset()
state[:, 0] = env.state

_mu_b, _sigma_b = env.init()
filter = EKF(env)
mu_b[:, 0], sigma_b[..., 0] = filter.innovate(_mu_b, _sigma_b, obs[:, 0])

for t in range(nb_steps):
    solver = BSPiLQR(env=env, init_belief=(mu_b[:, t], sigma_b[..., t]), nb_steps=10)
    trace = solver.run(nb_iter=25, verbose=False)

    act[:, t] = solver.uref[:, 0]
    obs[:, t + 1], _, _, _ = env.step(act[:, t])
    state[:, t + 1] = env.state

    mu_b[:, t + 1], sigma_b[..., t + 1] = filter.inference(mu_b[:, t], sigma_b[..., t],
                                                           act[:, t], obs[:, t + 1])

    print('Time Step:', t, 'Cost:', trace[-1])
