import gym
from trajopt.bspilqr import BSPiLQR

# light dark task
env = gym.make('LightDark-TO-v0')
env._max_episode_steps = 25

alg = BSPiLQR(env, nb_steps=25,
              activation=range(25))

# run belief-space ilqr
trace = alg.run()

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
