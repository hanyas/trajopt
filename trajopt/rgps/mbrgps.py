import autograd.numpy as np

import scipy as sc
from scipy import optimize

from copy import deepcopy

from trajopt.rgps.objects import Gaussian, QuadraticCost
from trajopt.rgps.objects import AnalyticalQuadraticCost
from trajopt.rgps.objects import QuadraticStateValue
from trajopt.rgps.objects import QuadraticStateActionValue
from trajopt.rgps.objects import LinearGaussianControl
from trajopt.rgps.objects import AnalyticalLinearGaussianDynamics
from trajopt.rgps.objects import MatrixNormalParameters

from trajopt.gps.objects import pass_alpha_as_vector

from trajopt.rgps.core import policy_divergence
from trajopt.rgps.core import gaussian_divergence
from trajopt.rgps.core import gaussian_interp_w2
from trajopt.rgps.core import gaussian_interp_kl
from trajopt.rgps.core import quad_expectation
from trajopt.rgps.core import policy_augment_cost
from trajopt.rgps.core import policy_backward_pass
from trajopt.rgps.core import parameter_augment_cost
from trajopt.rgps.core import parameter_backward_pass
from trajopt.rgps.core import regularized_parameter_augment_cost
from trajopt.rgps.core import cubature_forward_pass

import logging

LOGGER = logging.getLogger(__name__)


# Model-based Robust Guided Policy Search
class MBRGPS:

    def __init__(self, env, nb_steps,
                 init_state, init_action_sigma=1.,
                 policy_kl_bound=0.1, param_nominal_kl_bound=100.,
                 param_regularizer_kl_bound=1.,
                 policy_kl_stepwise=False, activation=None,
                 slew_rate=False, action_penalty=None,
                 nominal_variance=1e-8):

        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_cost = self.env.unwrapped.cost
        self.env_noise = self.env.unwrapped.noise
        self.env_init = init_state

        self.ulim = self.env.action_space.high

        self.dm_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]
        self.dm_param = self.dm_state * (self.dm_act + self.dm_state + 1)

        self.nb_steps = nb_steps

        # use slew rate penalty or not
        self.env.unwrapped.slew_rate = slew_rate
        if action_penalty is not None:
            self.env.unwrapped.uw = action_penalty * np.ones((self.dm_act, ))

        self.policy_kl_stepwise = policy_kl_stepwise
        if self.policy_kl_stepwise:
            self.policy_kl_bound = policy_kl_bound * np.ones((self.nb_steps, ))
            self.alpha = 1e8 * np.ones((self.nb_steps, ))
        else:
            self.policy_kl_bound = policy_kl_bound * np.ones((1, ))
            self.alpha = 1e8 * np.ones((1, ))

        self.param_nominal_kl_bound = param_nominal_kl_bound * np.ones((1, ))
        self.beta = 1e16 * np.ones((1, ))

        self.param_regularizer_kl_bound = param_regularizer_kl_bound * np.ones((1, ))
        self.eta = 1e16 * np.ones((1, ))

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init

        self.udist = Gaussian(self.dm_act, self.nb_steps)
        self.xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        self.dyn = AnalyticalLinearGaussianDynamics(self.env_dyn, self.env_noise,
                                                    self.dm_state, self.dm_act, self.nb_steps)

        # We assume process noise over dynamics is known
        self.noise = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        for t in range(self.nb_steps):
            self.noise[..., t] = self.dyn.sigma[..., t]

        self.param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        self.nominal = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        for t in range(self.nb_steps):
            self.nominal.sigma[..., t] = nominal_variance * np.eye(self.dm_param)

        self.ctl = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps, init_action_sigma)

        # activation of cost function in shape of sigmoid
        if activation is None:
            self.weighting = np.ones((self.nb_steps + 1, ))
        elif "mult" and "shift" in activation:
            t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
            self.weighting = 1. / (1. + np.exp(- activation['mult'] * (t - activation['shift'])))
        elif "discount" in activation:
            self.weighting = np.ones((self.nb_steps + 1,))
            gamma = activation["discount"] * np.ones((self.nb_steps, ))
            self.weighting[1:] = np.cumprod(gamma)
        else:
            raise NotImplementedError

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.dm_state, self.dm_act, self.nb_steps + 1)

        self.data = {}

    def rollout(self, nb_episodes, stoch=True, env=None,
                ctl=None, adversary=None):
        if env is None:
            env = self.env
            env_cost = self.env_cost
        else:
            env = env
            env_cost = env.unwrapped.cost

        if ctl is None:
            ctl = self.ctl
        else:
            ctl = ctl

        data = {'x': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'u': np.zeros((self.dm_act, self.nb_steps, nb_episodes)),
                'xn': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'c': np.zeros((self.nb_steps + 1, nb_episodes))}

        for n in range(nb_episodes):
            x = env.reset()

            for t in range(self.nb_steps):
                u = ctl.sample(x, t, stoch)
                data['u'][..., t, n] = u

                # expose true reward function
                c = env_cost(x, u, data['u'][..., t - 1, n], self.weighting[t])
                data['c'][t] = c

                data['x'][..., t, n] = x
                if adversary is None:
                    x, _, _, _ = env.step(u)
                else:
                    adversary = {'mu': self.param.mu[..., t],
                                 'sigma': self.param.sigma[..., t]}
                    x, _, _, _ = env.evolve(u, adversary)
                data['xn'][..., t, n] = x

            c = env_cost(x, np.zeros((self.dm_act, )),  np.zeros((self.dm_act, )), self.weighting[-1])
            data['c'][-1, n] = c

        return data

    def propagate(self, lgc):
        xdist, udist, lgd = self.dyn.extended_kalman(self.env_init, lgc, self.ulim)

        cost = np.zeros((self.nb_steps + 1, ))
        for t in range(self.nb_steps):
            cost[..., t] = self.env_cost(xdist.mu[..., t], udist.mu[..., t], udist.mu[..., t - 1], self.weighting[t])
        cost[..., -1] = self.env_cost(xdist.mu[..., -1], np.zeros((self.dm_act, )), np.zeros((self.dm_act, )), self.weighting[-1])

        return xdist, udist, lgd, cost

    def cubature_forward_pass(self, lgc, param):
        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)
        xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        xdist.mu, xdist.sigma,\
        udist.mu, udist.sigma,\
        xudist.mu, xudist.sigma = cubature_forward_pass(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                                        param.mu, param.sigma, self.noise,
                                                        lgc.K, lgc.kff, lgc.sigma,
                                                        self.dm_state, self.dm_act, self.nb_steps)

        # dm_augmented = self.dm_state + self.dm_act + 1 + self.dm_state
        # chol_sigma_augmented = 1e-8 * np.eye(dm_augmented)
        #
        # input_cubature_points = np.zeros((dm_augmented, 2 * dm_augmented))
        # output_cubature_points = np.zeros((self.dm_state, 2 * dm_augmented))
        #
        # xdist.mu[..., 0] = self.xdist.mu[..., 0]
        # xdist.sigma[..., 0] = self.xdist.sigma[..., 0]
        # for t in range(self.nb_steps):
        #     At = param.mu[0: self.dm_state * self.dm_state, t]
        #     Bt = param.mu[self.dm_state * self.dm_state: self.dm_state * self.dm_state + self.dm_state * self.dm_act, t]
        #     ct = param.mu[self.dm_state * self.dm_state + self.dm_state * self.dm_act:
        #                   self.dm_state * self.dm_state + self.dm_state * self.dm_act + self.dm_state, t]
        #
        #     At = np.reshape(At, (self.dm_state, self.dm_state), order='F')
        #     Bt = np.reshape(Bt, (self.dm_state, self.dm_act), order='F')
        #     ct = np.reshape(ct, (self.dm_state, 1), order='F')
        #
        #     udist.mu[..., t] = lgc.K[..., t] @ xdist.mu[..., t] + lgc.kff[..., t]
        #     udist.sigma[..., t] = lgc.sigma[..., t] + lgc.K[..., t] @ xdist.sigma[..., t] @ lgc.K[..., t].T
        #     udist.sigma[..., t] = 0.5 * (udist.sigma[..., t] + udist.sigma[..., t].T)
        #     udist.sigma[..., t] += 1e-8 * np.eye(self.dm_act)
        #
        #     xudist.mu[..., t] = np.hstack((xdist.mu[..., t], udist.mu[..., t]))
        #     xudist.sigma[..., t] = np.block([[xdist.sigma[..., t], xdist.sigma[..., t] @ lgc.K[..., t].T],
        #                                      [lgc.K[..., t] @ xdist.sigma[..., t], udist.sigma[..., t]]])
        #     xudist.sigma[..., t] += 1e-8 * np.eye(self.dm_state + self.dm_act)
        #
        #     mu_augmented = np.hstack((xudist.mu[..., t], 1., np.zeros((self.dm_state, ))))
        #     chol_sigma_augmented[0: self.dm_state + self.dm_act,
        #                          0: self.dm_state + self.dm_act] = np.linalg.cholesky(xudist.sigma[..., t])
        #     chol_sigma_augmented[-self.dm_state:, -self.dm_state:] = np.eye(self.dm_state)
        #
        #     input_cubature_points = np.hstack((chol_sigma_augmented, - chol_sigma_augmented))
        #     input_cubature_points *= np.sqrt(dm_augmented)
        #     input_cubature_points += mu_augmented[:, None]
        #
        #     for k in range(2 * dm_augmented):
        #         vec_xu = input_cubature_points[:, k]
        #         mat_xu = np.kron(vec_xu[:self.dm_state + self.dm_act + 1], np.eye(self.dm_state))
        #         covar = self.noise[..., t] + mat_xu @ param.sigma[..., t] @ mat_xu.T
        #         covar = 0.5 * (covar + covar.T)
        #
        #         chol_covar = np.linalg.cholesky(covar)
        #         output_cubature_points[:, k] = np.hstack((At, Bt, ct, chol_covar)) @ vec_xu
        #
        #     xdist.mu[..., t + 1] = np.mean(output_cubature_points, axis=-1)
        #
        #     xdist.sigma[..., t + 1] = np.zeros((self.dm_state, self.dm_state))
        #     for k in range(2 * dm_augmented):
        #         diff = output_cubature_points[:, k] - xdist.mu[..., t + 1]
        #         xdist.sigma[..., t + 1] += diff[:, None] @ diff[:, None].T
        #
        #     xdist.sigma[..., t + 1] /= (2 * dm_augmented)
        #     xdist.sigma[..., t + 1] = 0.5 * (xdist.sigma[..., t + 1] + xdist.sigma[..., t + 1].T)
        #
        #     if t == self.nb_steps - 1:
        #         xudist.mu[..., t + 1] = np.hstack((xdist.mu[..., t + 1], np.zeros((self.dm_act, ))))
        #         xudist.sigma[:self.dm_state, :self.dm_state, t + 1] = xdist.sigma[..., t + 1]

        return xdist, udist, xudist

    @pass_alpha_as_vector
    def policy_augment_cost(self, alpha):
        agcost = QuadraticCost(self.dm_state, self.dm_act, self.nb_steps + 1)
        agcost.Cxx, agcost.cx, agcost.Cuu,\
        agcost.cu, agcost.Cxu, agcost.c0 = policy_augment_cost(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                                               self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                               self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                               alpha, self.dm_state, self.dm_act, self.nb_steps)
        return agcost

    @pass_alpha_as_vector
    def policy_backward_pass(self, alpha, agcost, param):
        lgc = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu, xuvalue.q0, \
        xvalue.V, xvalue.v, xvalue.v0, \
        lgc.K, lgc.kff, lgc.sigma, diverge = policy_backward_pass(agcost.Cxx, agcost.cx, agcost.Cuu,
                                                                  agcost.cu, agcost.Cxu, agcost.c0,
                                                                  param.mu, param.sigma, self.noise,
                                                                  alpha, self.dm_state, self.dm_act, self.nb_steps)
        return lgc, xvalue, xuvalue, diverge

    def policy_kldiv(self, lgc, xdist):
        return policy_divergence(lgc.K, lgc.kff, lgc.sigma,
                                 self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                 xdist.mu, xdist.sigma,
                                 self.dm_state, self.dm_act, self.nb_steps)

    def policy_dual(self, alpha, param):
        # augmented cost
        agcost = self.policy_augment_cost(alpha)

        # backward pass
        lgc, xvalue, xuvalue, diverge = self.policy_backward_pass(alpha, agcost, param)

        # forward pass
        xdist, udist, xudist = self.cubature_forward_pass(lgc, param)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        if self.policy_kl_stepwise:
            dual = np.array([dual]) - np.sum(alpha * self.policy_kl_bound)
            grad = self.policy_kldiv(lgc, xdist) - self.policy_kl_bound
        else:
            dual = np.array([dual]) - alpha * self.policy_kl_bound
            grad = np.sum(self.policy_kldiv(lgc, xdist)) - self.policy_kl_bound

        return -1. * dual, -1. * grad

    def policy_dual_optimization(self, alpha, param, iters=10):
        if self.policy_kl_stepwise:
            min_alpha, max_alpha = 1e-4 * np.ones((self.nb_steps, )), 1e64 * np.ones((self.nb_steps, ))
        else:
            min_alpha, max_alpha = 1e-4 * np.ones((1, )), 1e64 * np.ones((1, ))

        best_alpha = alpha

        best_dual, best_grad = np.inf, np.inf
        for i in range(iters):
            dual, grad = self.policy_dual(alpha, param)

            if not np.isnan(dual) and not np.any(np.isnan(grad)):
                if grad < best_grad:
                    best_alpha = alpha
                    best_dual = dual
                    best_grad = grad

                if self.policy_kl_stepwise:
                    for t in range(self.nb_steps):
                        if np.all(abs(grad) < 0.1 * self.policy_kl_bound):
                            return alpha, dual, grad
                        else:
                            if grad[t] > 0:  # alpha too large
                                max_alpha[t] = alpha[t]
                                alpha[t] = np.sqrt(min_alpha[t] * max_alpha[t])
                            else:  # alpha too small
                                min_alpha[t] = alpha[t]
                                alpha[t] = np.sqrt(min_alpha[t] * max_alpha[t])
                else:
                    if abs(grad) < 0.1 * self.policy_kl_bound:
                        LOGGER.debug("Param KL: %.2e, Grad: %f, Beta: %f" % (self.policy_kl_bound, grad, alpha))
                        return alpha, dual, grad
                    else:
                        if grad > 0:  # alpha too large
                            max_alpha = alpha
                            alpha = np.sqrt(min_alpha * max_alpha)
                            LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Beta too big, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                                         % (self.policy_kl_bound, grad, alpha, min_alpha, max_alpha))
                        else:  # alpha too small
                            min_alpha = alpha
                            alpha = np.sqrt(min_alpha * max_alpha)
                            LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Beta too small, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                                         % (self.policy_kl_bound, grad, alpha, min_alpha, max_alpha))
            else:
                min_alpha = alpha
                alpha = np.sqrt(min_alpha * max_alpha)

        return best_alpha, best_dual, best_grad

    def parameter_augment_cost(self, beta):
        agcost = QuadraticCost(self.dm_param, self.dm_param, self.nb_steps)
        agcost.Cxx, agcost.cx, agcost.c0 = parameter_augment_cost(self.nominal.mu, self.nominal.sigma,
                                                                  beta, self.dm_param, self.nb_steps)
        return agcost

    def parameter_backward_pass(self, beta, agcost, xdist, ctl, eta=0.):
        param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        xvalue.V, xvalue.v, xvalue.v0,\
        param.mu, param.sigma, diverge = parameter_backward_pass(xdist.mu, xdist.sigma,
                                                                 ctl.K, ctl.kff, ctl.sigma, self.noise,
                                                                 self.cost.cx, self.cost.Cxx, self.cost.Cuu,
                                                                 self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                                 agcost.Cxx, agcost.cx, agcost.c0,
                                                                 beta, eta, self.dm_state, self.dm_act, self.dm_param,
                                                                 self.nb_steps)
        return param, xvalue, diverge

    def parameter_dual(self, beta, ctl):

        agcost = self.parameter_augment_cost(beta)

        # initial adversial xdist. with policy xdist.
        q_xdist = deepcopy(self.xdist)

        # first iteration to establish initial conv_kl
        param, xvalue, diverge = self.parameter_backward_pass(beta, agcost, q_xdist, ctl)
        if diverge != -1:
            return np.nan, np.nan
        p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

        # convergence of inner loop
        xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                        q_xdist.mu, q_xdist.sigma,
                                        self.dm_state, self.nb_steps + 1)
        while np.any(xdist_kl > 1e-3):
            param, xvalue, diverge = self.parameter_backward_pass(beta, agcost, q_xdist, ctl)
            if diverge != -1:
                return np.nan, np.nan
            p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

            # check convergence of loop
            xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                            q_xdist.mu, q_xdist.sigma,
                                            self.dm_state, self.nb_steps + 1)

            # interpolate between distributions
            q_xdist.mu, q_xdist.sigma = self.interp_gauss_kl(q_xdist.mu, q_xdist.sigma,
                                                             p_xdist.mu, p_xdist.sigma, 1e-1)
        # dual expectation
        dual = quad_expectation(q_xdist.mu[..., 0], q_xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        dual = np.array([dual]) + beta * (np.sum(self.parameter_nominal_kldiv(param)) - self.param_nominal_kl_bound)
        grad = np.sum(self.parameter_nominal_kldiv(param)) - self.param_nominal_kl_bound

        return -1. * dual, -1. * grad

    def parameter_dual_optimization(self, beta, ctl, iters=50):
        min_beta, max_beta = 1e-4 * np.ones((1, )), 1e64 * np.ones((1, ))

        best_beta = beta
        best_dual, best_grad = np.inf, np.inf
        for i in range(iters):
            dual, grad = self.parameter_dual(beta, ctl)

            if not np.isnan(dual) and not np.any(np.isnan(grad)):
                if grad < best_grad:
                    best_beta = beta
                    best_grad = grad
                    best_dual = dual

                if abs(grad) < 0.1 * self.param_nominal_kl_bound:
                    LOGGER.debug("Param KL: %.2e, Grad: %f, Beta: %f" % (self.param_regularizer_kl_bound, grad, beta))
                    return beta, dual, grad
                else:
                    if grad > 0:  # beta too large
                        max_beta = beta
                        beta = np.sqrt(min_beta * max_beta)
                        LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Beta too big, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                                     % (self.param_nominal_kl_bound, grad, beta, min_beta, max_beta))
                    else:  # beta too small
                        min_beta = beta
                        beta = np.sqrt(min_beta * max_beta)
                        LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Beta too small, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                                     % (self.param_nominal_kl_bound, grad, beta, min_beta, max_beta))
            else:
                min_beta = beta
                beta = np.sqrt(min_beta * max_beta)

        return best_beta, best_dual, best_grad

    def regularized_parameter_augment_cost(self, eta, last):
        agcost = QuadraticCost(self.dm_param, self.dm_param, self.nb_steps)
        agcost.Cxx, agcost.cx, agcost.c0 = regularized_parameter_augment_cost(last.mu, last.sigma, eta,
                                                                              self.dm_param, self.nb_steps)
        return agcost

    def regularized_parameter_dual(self, eta, ctl, last):

        agcost = self.regularized_parameter_augment_cost(eta, last)

        # initial adversial xdist. with policy xdist.
        q_xdist = deepcopy(self.xdist)

        # first iteration to establish initial conv_kl
        param, xvalue, diverge = self.parameter_backward_pass(0., agcost, q_xdist, ctl, eta)
        if diverge != -1:
            return np.nan, np.nan
        p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

        # convergence of inner loop
        xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                        q_xdist.mu, q_xdist.sigma,
                                        self.dm_state, self.nb_steps + 1)
        while np.any(xdist_kl > 1e-3):
            param, xvalue, diverge = self.parameter_backward_pass(0., agcost, q_xdist, ctl, eta)
            if diverge != -1:
                return np.nan, np.nan
            p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

            # check convergence of loop
            xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                            q_xdist.mu, q_xdist.sigma,
                                            self.dm_state, self.nb_steps + 1)

            # interpolate between distributions
            q_xdist.mu, q_xdist.sigma = self.interp_gauss_kl(q_xdist.mu, q_xdist.sigma,
                                                             p_xdist.mu, p_xdist.sigma, 1e-1)
        # dual expectation
        dual = quad_expectation(q_xdist.mu[..., 0], q_xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        dual = np.array([dual]) + eta * (np.sum(self.parameter_regularizer_kldiv(last, param)) - self.param_regularizer_kl_bound)
        grad = np.sum(self.parameter_regularizer_kldiv(last, param)) - self.param_regularizer_kl_bound

        return -1. * dual, -1. * grad

    def regularized_parameter_dual_optimization(self, eta, ctl, last, iters=10):
        min_eta, max_eta = 1e-4 * np.ones((1,)), 1e64 * np.ones((1,))

        best_eta = eta
        best_dual, best_grad = np.inf, np.inf
        for i in range(iters):
            dual, grad = self.regularized_parameter_dual(eta, ctl, last)

            if not np.isnan(dual) and not np.any(np.isnan(grad)):
                if grad < best_grad:
                    best_eta = eta
                    best_dual = dual
                    best_grad = grad

                if abs(grad) < 0.1 * self.param_regularizer_kl_bound:
                    LOGGER.debug("Param KL: %.2e, Grad: %f, Eta: %f" % (self.param_regularizer_kl_bound, grad, eta))
                    return eta, dual, grad
                else:
                    if grad > 0:  # eta too large
                        max_eta = eta
                        eta = np.sqrt(min_eta * max_eta)
                        LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Eta too big, New Eta: %2.3e, Min. Eta: %2.3e, Max. Eta: %2.3e"
                                     % (self.param_regularizer_kl_bound, grad, eta, min_eta, max_eta))
                    else:  # eta too small
                        min_eta = eta
                        eta = np.sqrt(min_eta * max_eta)
                        LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Eta too big, New Eta: %2.3e, Min. Eta: %2.3e, Max. Eta: %2.3e"
                                     % (self.param_regularizer_kl_bound, grad, eta, min_eta, max_eta))
            else:
                min_eta = eta
                eta = np.sqrt(min_eta * max_eta)

        return best_eta, best_dual, best_grad

    def parameter_nominal_kldiv(self, param):
        return self.gaussians_kldiv(param.mu, param.sigma,
                                    self.nominal.mu, self.nominal.sigma,
                                    self.dm_param, self.nb_steps)

    def parameter_regularizer_kldiv(self, last, param):
        return self.gaussians_kldiv(param.mu, param.sigma,
                                    last.mu, last.sigma,
                                    self.dm_param, self.nb_steps)

    @staticmethod
    def interp_gauss_kl(mu_q, sigma_q, mu_p, sigma_p, a):
        dim, nb_steps = mu_q.shape

        mu, sigma = gaussian_interp_kl(mu_q, sigma_q, mu_p, sigma_p,
                                       a, dim, nb_steps)

        # mu = np.zeros((dim, nb_steps))
        # sigma = np.zeros((dim, dim, nb_steps))
        #
        # for t in range(nb_steps):
        #     sigma = np.linalg.inv(a * np.linalg.inv(sigma_p) + (1. - a) * np.linalg.inv(sigma_q))
        #     mu = sigma @ (a * np.linalg.inv(sigma_p) @ mu_p + (1. - a) * np.linalg.inv(sigma_q) @ mu_q)

        return mu, sigma

    @staticmethod
    def interp_gauss_w2(mu_q, sigma_q, mu_p, sigma_p, a):
        dim, nb_steps = mu_q.shape

        # mu, sigma = gaussian_interp_w2(mu_q, sigma_q, mu_p, sigma_p,
        #                                a, dim, nb_steps)

        mu = np.zeros((dim, nb_steps))
        sigma = np.zeros((dim, dim, nb_steps))

        for t in range(nb_steps):
            mu[:, t] = (1. - a) * mu_q[:, t] + a * mu_p[:, t]
            sigma_0_chol = sc.linalg.sqrtm(sigma_q[..., t])
            sigma_0_chol_inv = np.linalg.inv(sigma_0_chol)
            _sigma = (1. - a) * sigma_q[..., t] + a * sc.linalg.sqrtm(sigma_0_chol @ sigma_p[..., t] @ sigma_0_chol)
            sigma[..., t] = sigma_0_chol_inv @ _sigma @ _sigma @ sigma_0_chol_inv

        return mu, sigma

    @staticmethod
    def gaussians_kldiv(mu_p, sigma_p, mu_q, sigma_q, dim, length):

        assert len(sigma_p) == len(sigma_q)
        assert len(mu_p) == len(mu_q)

        kl = gaussian_divergence(mu_p, sigma_p, mu_q, sigma_q, dim, length)[0]

        # kl = np.zeros((length, ))
        # for t in range(length):
        #     lmbda_q = np.linalg.inv(sigma_q[..., t])
        #
        #     diff_mu = (mu_q[..., t] - mu_p[..., t])[:, None]
        #     quad_term = diff_mu.T @ lmbda_q @ diff_mu
        #     trace_term = np.trace(lmbda_q @ sigma_p[..., t])
        #     log_det_term = np.log(np.linalg.det(sigma_q[..., t]) / np.linalg.det(sigma_p[..., t]))
        #
        #     kl[t] = 0.5 * (trace_term + quad_term + log_det_term - dim)

        return kl

    def parameter_optimization(self, ctl, iters=100):
        # worst-case parameter optimization
        beta = 1e16 * np.ones((1,))
        beta, _, _ = self.parameter_dual_optimization(beta, ctl, iters=iters)

        agcost = self.parameter_augment_cost(beta)

        # initial adversial xdist. with policy xdist.
        q_xdist = deepcopy(self.xdist)

        # first iteration to establish initial conv_kl
        param, _, _ = self.parameter_backward_pass(beta, agcost, q_xdist, ctl)
        p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

        # convergence of inner loop
        xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                        q_xdist.mu, q_xdist.sigma,
                                        self.dm_state, self.nb_steps + 1)
        while np.any(xdist_kl > 1e-3):
            param, _, _ = self.parameter_backward_pass(beta, agcost, q_xdist, ctl)
            p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

            # check convergence of loop
            xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                            q_xdist.mu, q_xdist.sigma,
                                            self.dm_state, self.nb_steps + 1)

            # interpolate between distributions
            q_xdist.mu, q_xdist.sigma = self.interp_gauss_kl(q_xdist.mu, q_xdist.sigma,
                                                             p_xdist.mu, p_xdist.sigma, 1e-1)

        return param, beta

    def reguarlized_parameter_optimization(self, ctl, iters=100,
                                           verbose=True):
        # worst-case parameter optimization
        last = deepcopy(self.nominal)

        param_nom_kl = 0.
        eta = 1e16 * np.ones((1,))
        while param_nom_kl < self.param_nominal_kl_bound:
            eta, _, _ = self.regularized_parameter_dual_optimization(eta, ctl, last, iters=iters)

            agcost = self.regularized_parameter_augment_cost(eta, last)

            # initial adversial xdist. with policy xdist.
            q_xdist = deepcopy(self.xdist)

            # first iteration to establish initial conv_kl
            param, _, _ = self.parameter_backward_pass(0., agcost, q_xdist, ctl, eta)
            p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

            # convergence of inner loop
            xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                            q_xdist.mu, q_xdist.sigma,
                                            self.dm_state, self.nb_steps + 1)
            while np.any(xdist_kl > 1e-3):
                param, _, _ = self.parameter_backward_pass(0., agcost, q_xdist, ctl, eta)
                p_xdist, _, _ = self.cubature_forward_pass(ctl, param)

                # check convergence of loop
                xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                                q_xdist.mu, q_xdist.sigma,
                                                self.dm_state, self.nb_steps + 1)

                # interpolate between distributions
                q_xdist.mu, q_xdist.sigma = self.interp_gauss_kl(q_xdist.mu, q_xdist.sigma,
                                                                 p_xdist.mu, p_xdist.sigma, 1e-1)

            param_reg_kl = np.sum(self.parameter_regularizer_kldiv(last, param))
            if np.abs(param_reg_kl - self.param_regularizer_kl_bound) < 0.1 * self.param_regularizer_kl_bound:
                # update param dist.
                last = param
                # compute target kl to nominal
                param_nom_kl = np.sum(self.parameter_nominal_kldiv(param))

        return last, eta

    def policy_optimization(self, param):
        # policy optimization
        if self.policy_kl_stepwise:
            alpha_init = 1e4 * np.ones((self.nb_steps,))
            alpha_bounds = ((1e-16, 1e16), ) * self.nb_steps
        else:
            alpha_init = 1e4 * np.ones((1,))
            alpha_bounds = ((1e-16, 1e16), ) * 1

        # policy optimization
        res = sc.optimize.minimize(self.policy_dual, alpha_init,
                                   method='L-BFGS-B', jac=True,
                                   bounds=alpha_bounds, args=param,
                                   options={'disp': False, 'maxiter': 100000,
                                            'ftol': 1e-12, 'iprint': 0})
        alpha = res.x

        # alpha, _, _ = self.policy_dual_optimization(alpha_init, param, iters=500)

        # re-compute after opt.
        policy_agcost = self.policy_augment_cost(alpha)
        lgc, xvalue, xuvalue, diverge = self.policy_backward_pass(alpha, policy_agcost, param)
        worst_xdist, worst_udist, worst_xudist = self.cubature_forward_pass(lgc, param)

        return lgc, worst_xdist, xvalue, xuvalue, alpha

    def plot_distributions(self, xdist=None, udist=None, show=True,
                           axs=None, xcolor='blue', ucolor='green'):
        if xdist is None and udist is None:
            xdist = self.xdist
            udist = self.udist

        import matplotlib.pyplot as plt

        nb_plots = 0
        if axs is None:
            if xdist is not None:
                nb_plots += self.dm_state
            if udist is not None:
                nb_plots += self.dm_act
            _, axs = plt.subplots(nb_plots, figsize=(8, 12))

        if xdist is not None:
            for k in range(self.dm_state):
                t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
                axs[k].plot(t, xdist.mu[k, :], xcolor)
                lb = xdist.mu[k, :] - 2. * np.sqrt(xdist.sigma[k, k, :])
                ub = xdist.mu[k, :] + 2. * np.sqrt(xdist.sigma[k, k, :])
                axs[k].fill_between(t, lb, ub, color=xcolor, alpha=0.1)

        if udist is not None:
            for k in range(self.dm_act):
                t = np.linspace(0, self.nb_steps - 1, self.nb_steps)
                axs[self.dm_state + k].plot(t, udist.mu[k, :], ucolor)
                lb = udist.mu[k, :] - 2. * np.sqrt(udist.sigma[k, k, :])
                ub = udist.mu[k, :] + 2. * np.sqrt(udist.sigma[k, k, :])
                axs[self.dm_state + k].fill_between(t, lb, ub, color=ucolor, alpha=0.1)

        if show:
            plt.show()

        return axs

    def run(self, nb_iter=10, verbose=False,
            optimize_adversary=True, iterative_adversary=False):

        trace = []

        # get mean traj. and linear system dynamics
        _, _, lgd, cost = self.propagate(self.ctl)

        # update linearization of dynamics
        self.dyn.params = lgd.A, lgd.B, lgd.c, lgd.sigma
        for t in range(self.nb_steps):
            A, B, c = self.dyn.A[..., t], self.dyn.B[..., t], self.dyn.c[..., t]
            tmp = np.hstack((A, B, c[:, None]))
            self.nominal.mu[..., t] = np.reshape(tmp, self.dm_param, order='F')

        # current state distribution
        self.xdist, self.udist, self.xudist = self.cubature_forward_pass(self.ctl, self.nominal)

        # get quadratic cost around mean traj.
        self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)
        trace.append(np.sum(cost))

        for iter in range(nb_iter):
            if optimize_adversary:
                if iterative_adversary:
                    self.param, self.eta = self.reguarlized_parameter_optimization(self.ctl)
                else:
                    self.param, self.beta = self.parameter_optimization(self.ctl)
            else:
                self.param = self.nominal

            param_nom_kl = np.sum(self.parameter_nominal_kldiv(self.param))

            lgc, worst_xdist, xvalue, xuvalue, alpha = self.policy_optimization(self.param)

            # check kl constraint
            policy_kl = self.policy_kldiv(lgc, worst_xdist)
            if not self.policy_kl_stepwise:
                policy_kl = np.sum(policy_kl)

            if np.all((policy_kl - self.policy_kl_bound) < 0.25 * self.policy_kl_bound) \
                    or np.all(policy_kl < self.policy_kl_bound):

                # update alpha
                self.alpha = alpha

                # update controller
                self.ctl = lgc

                # update value functions
                self.vfunc, self.qfunc = xvalue, xuvalue

                # extended-Kalman forward simulation
                _, _, lgd, cost = self.propagate(lgc)

                # update linearization of dynamics
                self.dyn.params = lgd.A, lgd.B, lgd.c, lgd.sigma
                for t in range(self.nb_steps):
                    A, B, c = self.dyn.A[..., t], self.dyn.B[..., t], self.dyn.c[..., t]
                    tmp = np.hstack((A, B, c[:, None]))
                    self.nominal.mu[..., t] = np.reshape(tmp, self.dm_param, order='F')

                # current state distribution
                self.xdist, self.udist, self.xudist = self.cubature_forward_pass(self.ctl, self.param)

                # get quadratic cost around mean traj.
                self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)
                trace.append(np.sum(cost))

                if verbose:
                    if iter == 0:
                        print("%9s %8s %7s %8s %8s" % ("", "param_kl", "", "policy_kl", ""))
                        print("%6s %6s %6s %2s %6s %6s %12s" % ("iter", "req.", "act.", "", "req.", "act.", "return"))
                    print("%6i %.2e %.2e %6.2f %6.2f %12.2f" % (iter, self.param_nominal_kl_bound, param_nom_kl,
                                                                np.sum(self.policy_kl_bound), np.sum(policy_kl),
                                                                np.sum(cost)))
            else:
                print("Something is wrong, KL not satisfied: ", "req", np.sum(self.policy_kl_bound),
                                                                "act.", np.sum(policy_kl))
                if self.policy_kl_stepwise:
                    self.alpha = 1e8 * np.ones((self.nb_steps,))
                else:
                    self.alpha = 1e8 * np.ones((1,))

        return trace
