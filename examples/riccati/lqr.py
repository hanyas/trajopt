import autograd.numpy as np

import gym
from trajopt.riccati import Riccati

np.random.seed(1337)

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

env.seed(1337)

alg = Riccati(env, nb_steps=60,
              init_state=env.init())

# run Riccati
trace = alg.run()

alg.plot()
