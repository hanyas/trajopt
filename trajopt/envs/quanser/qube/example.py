#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: example.py
# @Date: 2019-06-30-20-53
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import numpy as np
import gym

import trajopt
from trajopt.envs.quanser.qube.ctrl import SwingUpCtrl

# quanser cartpole env
env = gym.make('Quanser-Qube-TO-v0')
env._max_episode_steps = 1000000

ctrl = SwingUpCtrl(ref_energy=0.04, energy_gain=30.0, acc_max=5.0)

obs = env.reset()
done = False
while not done:
    env.render()
    act = ctrl(obs)
    obs, _, _, _ = env.step(act)
