import autograd.numpy as np

import gym
from trajopt.riccati import Riccati

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

dm_state = env.observation_space.shape[0]
dm_act = env.action_space.shape[0]

nb_steps = 60

env_sigma = env.unwrapped._sigma

state = np.zeros((dm_state, nb_steps + 1))
action = np.zeros((dm_act, nb_steps))

state[:, 0] = env.reset()
alg = Riccati(env, nb_steps=nb_steps,
              init_state=tuple([state[:, 0], env_sigma]))

# run iLQR
trace = alg.run()

# plot forward pass
alg.plot()
