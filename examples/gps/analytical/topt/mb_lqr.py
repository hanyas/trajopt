import gym
from trajopt.gps import MBGPS

import warnings
warnings.filterwarnings("ignore")


# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 60

solver = MBGPS(env,  nb_steps=60,
               init_state=env.init(),
               init_action_sigma=100.,
               kl_bound=5.,
               activation={'mult': 1.5, 'shift': 50})

trace = solver.run(nb_iter=10, verbose=True)

# plot dists
solver.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
