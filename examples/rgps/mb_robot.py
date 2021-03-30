import autograd.numpy as np

import gym
from trajopt.rgps import MBRGPS

from matplotlib import rc
import matplotlib.pyplot as plt

import tikzplotlib

import warnings
warnings.filterwarnings("ignore")

rc('lines', **{'linewidth': 1})
rc('text', usetex=True)


def beautify(ax):
    ax.set_frame_on(True)
    ax.minorticks_on()

    ax.grid(True)
    ax.grid(linestyle=':')

    ax.tick_params(which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False,
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    # ax.autoscale(tight=True)
    # ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc='best')

    return ax


# pendulum task
env = gym.make('Robot-TO-v0')
env._max_episode_steps = 100

np.random.seed(1337)
env.seed(1337)

rgps = MBRGPS(env, nb_steps=100,
              init_state=env.init(),
              policy_kl_bound=0.25,
              param_nominal_kl_bound=5e2,
              param_regularizer_kl_bound=10,
              init_action_sigma=0.1,
              action_penalty=1e-1,
              nominal_variance=1e-6)
rgps.run(nb_iter=900, verbose=True)

np.random.seed(1337)
env.seed(1337)

gps = MBRGPS(env, nb_steps=100,
             init_state=env.init(),
             policy_kl_bound=.25,
             param_nominal_kl_bound=5e2,
             param_regularizer_kl_bound=10,
             init_action_sigma=0.1,
             action_penalty=1e-1,
             nominal_variance=1e-6)
gps.run(nb_iter=50, verbose=True,
        optimize_adversary=False)

# compute attack on final standard controller
gps.param_nominal_kl_bound = np.array([1e3])
gps.param, gps.eta = gps.reguarlized_parameter_optimization(gps.ctl)
print("Disturbance KL:", gps.parameter_nominal_kldiv(gps.param).sum())

fig = plt.figure(figsize=(6, 12))
plt.suptitle("Standard vs Robust Ctl: Feedback Controller")
for i in range(gps.dm_state):
    plt.subplot(gps.dm_state + gps.dm_act, 1, i + 1)
    plt.plot(gps.ctl.K[0, i, ...], color='b', marker='o', markersize=2)
    plt.plot(rgps.ctl.K[0, i, ...], color='r', marker='x', markersize=2)

for i in range(gps.dm_act):
    plt.subplot(gps.dm_state + gps.dm_act, 1, gps.dm_state + i + 1)
    plt.plot(gps.ctl.kff[i, ...], color='b', marker='o', markersize=2)
    plt.plot(rgps.ctl.kff[i, ...], color='r', marker='x', markersize=2)

axs = fig.get_axes()
axs = [beautify(ax) for ax in axs]
plt.show()

# tikzplotlib.save("robot_feedback_gains.tex")

std_xdist, std_udist, _ = gps.cubature_forward_pass(gps.ctl, gps.nominal)
robust_xdist, robust_udist, _ = rgps.cubature_forward_pass(rgps.ctl, rgps.nominal)

cost_nom_env_std_ctl = gps.cost.evaluate(std_xdist, std_udist)
cost_nom_env_rbst_ctl = rgps.cost.evaluate(robust_xdist, robust_udist)

print("Expected Cost of Standard and Robust Control on Nominal Env")
print("Std. Ctl.: ", cost_nom_env_std_ctl, "Rbst. Ctl.", cost_nom_env_rbst_ctl)

std_worst_xdist, std_worst_udist, _ = gps.cubature_forward_pass(gps.ctl, gps.param)
robust_worst_xdist, robust_worst_udist, _ = rgps.cubature_forward_pass(rgps.ctl, rgps.param)

cost_adv_env_std_ctl = gps.cost.evaluate(std_worst_xdist, std_worst_udist)
cost_adv_env_rbst_ctl = rgps.cost.evaluate(robust_worst_xdist, robust_worst_udist)

print("Expected Cost of Standard and Robust Control on Adverserial Env")
print("Std. Ctl.: ", cost_adv_env_std_ctl, "Rbst. Ctl.", cost_adv_env_rbst_ctl)


fig = plt.figure()
plt.suptitle('Standard and Robust Control Without Adversary')
for k in range(rgps.dm_state):
    plt.subplot(rgps.dm_state + rgps.dm_act, 1, k + 1)

    t = np.linspace(0, rgps.nb_steps, rgps.nb_steps + 1)

    plt.plot(t, std_xdist.mu[k, :], '-b')
    lb = std_xdist.mu[k, :] - 2. * np.sqrt(std_xdist.sigma[k, k, :])
    ub = std_xdist.mu[k, :] + 2. * np.sqrt(std_xdist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

for k in range(rgps.dm_act):
    plt.subplot(rgps.dm_state + rgps.dm_act, 1, rgps.dm_state + k + 1)

    t = np.linspace(0, rgps.nb_steps - 1, rgps.nb_steps)

    plt.plot(t, std_udist.mu[k, :], '-b')
    lb = std_udist.mu[k, :] - 2. * np.sqrt(std_udist.sigma[k, k, :])
    ub = std_udist.mu[k, :] + 2. * np.sqrt(std_udist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

for k in range(gps.dm_state):
    plt.subplot(gps.dm_state + gps.dm_act, 1, k + 1)

    t = np.linspace(0, rgps.nb_steps, rgps.nb_steps + 1)

    plt.plot(t, robust_xdist.mu[k, :], '-r')
    lb = robust_xdist.mu[k, :] - 2. * np.sqrt(robust_xdist.sigma[k, k, :])
    ub = robust_xdist.mu[k, :] + 2. * np.sqrt(robust_xdist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='red', alpha=0.1)

for k in range(rgps.dm_act):
    plt.subplot(gps.dm_state + gps.dm_act, 1, gps.dm_state + k + 1)

    t = np.linspace(0, rgps.nb_steps - 1, rgps.nb_steps)

    plt.plot(t, robust_udist.mu[k, :], '-r')
    lb = robust_udist.mu[k, :] - 2. * np.sqrt(robust_udist.sigma[k, k, :])
    ub = robust_udist.mu[k, :] + 2. * np.sqrt(robust_udist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='red', alpha=0.1)

axs = fig.get_axes()
axs = [beautify(ax) for ax in axs]
plt.show()

# tikzplotlib.save("robot_trajectories_nominal.tex")

fig = plt.figure()
plt.suptitle('Standard and Robust Control With Adversary')
for k in range(gps.dm_state):
    plt.subplot(gps.dm_state + gps.dm_act, 1, k + 1)

    t = np.linspace(0, rgps.nb_steps, rgps.nb_steps + 1)

    plt.plot(t, std_worst_xdist.mu[k, :], '-g')
    lb = std_worst_xdist.mu[k, :] - 2. * np.sqrt(std_worst_xdist.sigma[k, k, :])
    ub = std_worst_xdist.mu[k, :] + 2. * np.sqrt(std_worst_xdist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='green', alpha=0.1)

for k in range(rgps.dm_act):
    plt.subplot(gps.dm_state + gps.dm_act, 1, gps.dm_state + k + 1)

    t = np.linspace(0, rgps.nb_steps - 1, rgps.nb_steps)

    plt.plot(t, std_worst_udist.mu[k, :], '-g')
    lb = std_worst_udist.mu[k, :] - 2. * np.sqrt(std_worst_udist.sigma[k, k, :])
    ub = std_worst_udist.mu[k, :] + 2. * np.sqrt(std_worst_udist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='green', alpha=0.1)

for k in range(gps.dm_state):
    plt.subplot(gps.dm_state + gps.dm_act, 1, k + 1)

    t = np.linspace(0, rgps.nb_steps, rgps.nb_steps + 1)

    plt.plot(t, robust_worst_xdist.mu[k, :], '-m')
    lb = robust_worst_xdist.mu[k, :] - 2. * np.sqrt(robust_worst_xdist.sigma[k, k, :])
    ub = robust_worst_xdist.mu[k, :] + 2. * np.sqrt(robust_worst_xdist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='magenta', alpha=0.1)

for k in range(rgps.dm_act):
    plt.subplot(gps.dm_state + gps.dm_act, 1, gps.dm_state + k + 1)

    t = np.linspace(0, rgps.nb_steps - 1, rgps.nb_steps)

    plt.plot(t, robust_worst_udist.mu[k, :], '-m')
    lb = robust_worst_udist.mu[k, :] - 2. * np.sqrt(robust_worst_udist.sigma[k, k, :])
    ub = robust_worst_udist.mu[k, :] + 2. * np.sqrt(robust_worst_udist.sigma[k, k, :])
    plt.fill_between(t, lb, ub, color='magenta', alpha=0.1)

axs = fig.get_axes()
axs = [beautify(ax) for ax in axs]
plt.show()

# tikzplotlib.save("robot_trajectories_adversarial.tex")

from trajopt.rgps.objects import MatrixNormalParameters
interp = MatrixNormalParameters(rgps.dm_state, rgps.dm_act, rgps.nb_steps)

alphas = np.linspace(0., 2., 21)

cost_adv_env_std_ctl = []
cost_adv_env_rbst_ctl = []
kl_distance = []

for alpha in alphas:
    print('Alpha:', alpha)

    interp.mu, interp.sigma = gps.interp_gauss_kl(gps.nominal.mu, gps.nominal.sigma,
                                                  gps.param.mu, gps.param.sigma, alpha)

    kl_distance.append(np.sum(gps.parameter_nominal_kldiv(interp)))

    std_worst_xdist, std_worst_udist, _ = gps.cubature_forward_pass(gps.ctl, interp)
    robust_worst_xdist, robust_worst_udist, _ = gps.cubature_forward_pass(rgps.ctl, interp)

    cost_adv_env_std_ctl.append(gps.cost.evaluate(std_worst_xdist, std_worst_udist))
    cost_adv_env_rbst_ctl.append(gps.cost.evaluate(robust_worst_xdist, robust_worst_udist))

    print("Expected Cost of Standard and Robust Control on Adverserial Env")
    print("Std. Ctl.: ", cost_adv_env_std_ctl[-1], "Rbst. Ctl.", cost_adv_env_rbst_ctl[-1])

fig = plt.figure()
plt.plot(kl_distance, cost_adv_env_std_ctl, 'b', marker='o')
# plt.xscale('log')
plt.yscale('log')
plt.plot(kl_distance, cost_adv_env_rbst_ctl, 'r', marker='*')
# plt.xscale('log')
plt.yscale('log')

axs = fig.gca()
axs = beautify(axs)
plt.show()

# tikzplotlib.save("robot_cost_over_distance.tex")

kl_over_time = gps.parameter_nominal_kldiv(gps.param)

fig = plt.figure()
plt.plot(kl_over_time, 'k', marker='.')
plt.yscale('log')

axs = fig.gca()
axs = beautify(axs)
plt.show()

# tikzplotlib.save("robot_kl_over_time.tex")
