import gym
from trajopt.bspilqr import BSPiLQR

# car task
env = gym.make('Car-TO-v0')
env._max_episode_steps = 25

alg = BSPiLQR(env, nb_steps=25,
              activation=range(-1, 0))

# run belief-space ilqr
trace = alg.run(nb_iter=5)

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
