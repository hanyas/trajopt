import gym
from trajopt.riccati import Riccati

# lqr task
env = gym.make('LQR-TO-v0')
env._max_episode_steps = 100

alg = Riccati(env, nb_steps=60)

# run iLQR
trace = alg.run()

# plot forward pass
alg.plot()
