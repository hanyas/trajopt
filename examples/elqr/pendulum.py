import gym
from trajopt.elqr import eLQR

# pendulum task
env = gym.make('Pendulum-TO-v0')
env._max_episode_steps = 100

state = env.reset()
alg = eLQR(env, nb_steps=100,
           init_state=state)

# run eLQR
trace = alg.run()

# plot forward pass
alg.plot()

# plot objective
import matplotlib.pyplot as plt

plt.figure()
plt.plot(trace)
plt.show()
