import gym
from trajopt.ilqr import iLQR


# pendulum env
env = gym.make('Pendulum-TO-v1')
env._max_episode_steps = 500

alg = iLQR(env, nb_steps=500,
           activation=range(350, 500))

# run iLQR
trace = alg.run(nb_iter=100)

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
