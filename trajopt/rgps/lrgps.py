import autograd.numpy as np

import scipy as sc
from scipy import optimize

from copy import deepcopy

from trajopt.rgps.objects import Gaussian, QuadraticCost
from trajopt.rgps.objects import AnalyticalQuadraticCost
from trajopt.rgps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.rgps.objects import LinearGaussianControl
from trajopt.rgps.objects import MatrixNormalParameters

from trajopt.gps.objects import pass_alpha_as_vector

from trajopt.rgps.core import policy_divergence
from trajopt.rgps.core import gaussian_divergence
from trajopt.rgps.core import gaussian_interp_w2
from trajopt.rgps.core import gaussian_interp_kl
from trajopt.rgps.core import quad_expectation
from trajopt.rgps.core import policy_augment_cost, policy_backward_pass
from trajopt.rgps.core import parameter_backward_pass
from trajopt.rgps.core import parameter_augment_cost
from trajopt.rgps.core import cubature_forward_pass

import logging

LOGGER = logging.getLogger(__name__)


# Robust GPS on a linear system
class LRGPS:

    def __init__(self, env, nb_steps,
                 init_state, init_action_sigma=1.,
                 policy_kl_bound=0.1, param_kl_bound=100,
                 kl_stepwise=False, activation=None,
                 slew_rate=False, action_penalty=None):

        # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_noise = self.env.unwrapped.sigma
        self.env_cost = self.env.unwrapped.cost
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

        self.kl_stepwise = kl_stepwise
        if self.kl_stepwise:
            self.policy_kl_bound = policy_kl_bound * np.ones((self.nb_steps, ))
            self.alpha = 1e8 * np.ones((self.nb_steps, ))
        else:
            self.policy_kl_bound = policy_kl_bound * np.ones((1, ))
            self.alpha = 1e8 * np.ones((1, ))

        self.param_kl_bound = param_kl_bound
        self.beta = np.array([1e16])

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init

        self.udist = Gaussian(self.dm_act, self.nb_steps)
        self.xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        self.nominal = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)

        # LQG dynamics
        # _A = np.array([[1.11627793, 0.00000000],
        #                [0.11107100, 1.10517083]])
        # _B = np.array([[0.10570721],
        #                [0.00536379]])
        # _c = np.array([-1.16277934, -2.16241837])

        _A = np.array([[ 9.99999500e-01,  9.99000500e-03],
                       [-9.99000500e-05,  9.98001499e-01]])
        _B = np.array([[4.99666792e-05],
                       [9.99000500e-03]])
        _c = np.array([0.,  0.])

        _params = np.hstack((_A, _B, _c[:, None]))
        for t in range(self.nb_steps):
            self.nominal.mu[..., t] = np.reshape(_params, self.dm_param, order='F')
            self.nominal.sigma[..., t] = 1e-8 * np.eye(self.dm_param)

        self.param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)

        # We assume process noise over dynamics is known
        self.noise = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        for t in range(self.nb_steps):
            self.noise[..., t] = self.env_noise

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

        self.last_return = - np.inf

        self.data = {}

    def rollout(self, nb_episodes, stoch=True, env=None):
        if env is None:
            env = self.env
            env_cost = self.env_cost
        else:
            env = env
            env_cost = env.unwrapped.cost

        data = {'x': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'u': np.zeros((self.dm_act, self.nb_steps, nb_episodes)),
                'xn': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'c': np.zeros((self.nb_steps + 1, nb_episodes))}

        for n in range(nb_episodes):
            x = env.reset()

            for t in range(self.nb_steps):
                u = self.ctl.sample(x, t, stoch)
                data['u'][..., t, n] = u

                # expose true reward function
                c = env_cost(x, u, data['u'][..., t - 1, n], self.weighting[t])
                data['c'][t] = c

                data['x'][..., t, n] = x
                x, _, _, _ = env.step(u)
                data['xn'][..., t, n] = x

            c = env_cost(x, np.zeros((self.dm_act, )),  np.zeros((self.dm_act, )), self.weighting[-1])
            data['c'][-1, n] = c

        return data

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
    def policy_backward_pass(self, alpha, agcost):
        lgc = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu, xuvalue.q0, \
        xvalue.V, xvalue.v, xvalue.v0, \
        lgc.K, lgc.kff, lgc.sigma, diverge = policy_backward_pass(agcost.Cxx, agcost.cx, agcost.Cuu,
                                                                  agcost.cu, agcost.Cxu, agcost.c0,
                                                                  self.param.mu, self.param.sigma, self.noise,
                                                                  alpha, self.dm_state, self.dm_act, self.nb_steps)
        return lgc, xvalue, xuvalue, diverge

    def policy_dual(self, alpha):
        # augmented cost
        agcost = self.policy_augment_cost(alpha)

        # backward pass
        lgc, xvalue, xuvalue, diverge = self.policy_backward_pass(alpha, agcost)

        # forward pass
        xdist, udist, xudist = self.cubature_forward_pass(lgc, self.param)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        if self.kl_stepwise:
            dual = np.array([dual]) - np.sum(alpha * self.policy_kl_bound)
            grad = self.policy_kldiv(lgc, xdist) - self.policy_kl_bound
        else:
            dual = np.array([dual]) - alpha * self.policy_kl_bound
            grad = np.sum(self.policy_kldiv(lgc, xdist)) - self.policy_kl_bound

        return -1. * dual, -1. * grad

    def policy_kldiv(self, lgc, xdist):
        return policy_divergence(lgc.K, lgc.kff, lgc.sigma,
                                 self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                 xdist.mu, xdist.sigma,
                                 self.dm_state, self.dm_act, self.nb_steps)

    def parameter_augment_cost(self, beta):
        agcost = QuadraticCost(self.dm_param, self.dm_param, self.nb_steps)
        agcost.Cxx, agcost.cx, agcost.c0 = parameter_augment_cost(self.nominal.mu, self.nominal.sigma,
                                                                  beta, self.dm_param, self.nb_steps)
        return agcost

    def parameter_backward_pass(self, beta, agcost, xdist):
        param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        xvalue.V, xvalue.v, xvalue.v0,\
        param.mu, param.sigma, diverge = parameter_backward_pass(xdist.mu, xdist.sigma,
                                                                 self.ctl.K, self.ctl.kff, self.ctl.sigma, self.noise,
                                                                 self.cost.cx, self.cost.Cxx, self.cost.Cuu,
                                                                 self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                                 agcost.Cxx, agcost.cx, agcost.c0,
                                                                 beta, self.dm_state, self.dm_act, self.dm_param,
                                                                 self.nb_steps)
        return param, xvalue, diverge

    def parameter_dual(self, beta):

        agcost = self.parameter_augment_cost(beta)

        # # initialize adversial xdist with wide param dist.
        # param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        # q_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

        # initial adversial xdist. with policy xdist.
        q_xdist = deepcopy(self.xdist)

        # first iteration to establish initial conv_kl
        param, xvalue, diverge = self.parameter_backward_pass(beta, agcost, q_xdist)
        if diverge != -1:
            return np.nan, np.nan
        p_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

        # conergence of inner loop
        xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                        q_xdist.mu, q_xdist.sigma,
                                        self.dm_state, self.nb_steps + 1)
        while np.any(xdist_kl > 1e-3):
            param, xvalue, diverge = self.parameter_backward_pass(beta, agcost, q_xdist)
            if diverge != -1:
                return np.nan, np.nan
            p_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

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

        dual += beta * (np.sum(self.parameter_kldiv(param)) - self.param_kl_bound)

        # dual gradient
        grad = np.sum(self.parameter_kldiv(param)) - self.param_kl_bound

        return -1. * np.array([dual]), -1. * np.array([grad])

    def parameter_dual_optimization(self, beta, iters=10):
        min_beta, max_beta = 1e-4, 1e64
        best_beta = beta

        best_dual, best_grad = np.inf, None
        for i in range(iters):
            dual, grad = self.parameter_dual(beta)

            if not np.isnan(dual) and not np.isnan(grad):
                if dual < best_dual:
                    best_beta = beta
                    best_dual = dual
                    best_grad = grad

                if abs(grad) < 0.1 * self.param_kl_bound:
                    LOGGER.debug("Param KL: %.2e, Grad: %f, Beta: %f" % (self.param_kl_bound, grad, beta))
                    return beta, dual, grad
                else:
                    if grad > 0:  # beta too large
                        max_beta = beta
                        beta = np.sqrt(min_beta * max_beta)
                        LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Beta too big, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                                     % (self.param_kl_bound, grad, beta, min_beta, max_beta))
                    else:  # beta too small
                        min_beta = beta
                        beta = np.sqrt(min_beta * max_beta)
                        LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Beta too small, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                                     % (self.param_kl_bound, grad, beta, min_beta, max_beta))

            else:
                min_beta = beta
                beta = np.sqrt(min_beta * max_beta)
                LOGGER.debug("Param KL: %.1e, Grad: %2.3e, Backward pass diverged, New Beta: %2.3e, Min. Beta: %2.3e, Max. Beta: %2.3e"
                             % (self.param_kl_bound, grad, beta, min_beta, max_beta))

        return best_beta, best_dual, best_grad

    def parameter_kldiv(self, param):
        return self.gaussians_kldiv(param.mu, param.sigma,
                                    self.nominal.mu, self.nominal.sigma,
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

    def plot(self, xdist=None, udist=None):
        xdist = self.xdist if xdist is None else xdist
        udist = self.udist if udist is None else udist

        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, k + 1)
            plt.plot(t, xdist.mu[k, :], '-b')
            lb = xdist.mu[k, :] - 2. * np.sqrt(xdist.sigma[k, k, :])
            ub = xdist.mu[k, :] + 2. * np.sqrt(xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + k + 1)
            plt.plot(t, udist.mu[k, :], '-g')
            lb = udist.mu[k, :] - 2. * np.sqrt(udist.sigma[k, k, :])
            ub = udist.mu[k, :] + 2. * np.sqrt(udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='green', alpha=0.1)

        plt.show()

    def compare(self, std_ctl):
        assert std_ctl is not None

        std_xdist, std_udist, _ = self.cubature_forward_pass(std_ctl, self.nominal)
        std_worst_xdist, std_worst_udist, _ = self.cubature_forward_pass(std_ctl, self.param)

        robust_xdist, robust_udist, _ = self.cubature_forward_pass(self.ctl, self.nominal)
        robust_worst_xdist, robust_worst_udist, _ = self.cubature_forward_pass(self.ctl, self.param)

        cost_nom_env_std_ctl = self.cost.evaluate(std_xdist, std_udist)
        cost_nom_env_rbst_ctl = self.cost.evaluate(robust_xdist, robust_udist)

        print("Cost of Standard and Robust Control on Nominal Env")
        print("Std. Ctl.: ", cost_nom_env_std_ctl, "Rbst. Ctl.", cost_nom_env_rbst_ctl)

        cost_adv_env_std_ctl = self.cost.evaluate(std_worst_xdist, std_worst_udist)
        cost_adv_env_rbst_ctl = self.cost.evaluate(robust_worst_xdist, robust_worst_udist)

        print("Cost of Standard and Robust Control on Adverserial Env")
        print("Std. Ctl.: ", cost_adv_env_std_ctl, "Rbst. Ctl.", cost_adv_env_rbst_ctl)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 12))
        plt.suptitle("Standard vs Robust Ctl: Nominal Env")

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, k + 1)

            plt.plot(t, std_xdist.mu[k, :], '-b')
            lb = std_xdist.mu[k, :] - 2. * np.sqrt(std_xdist.sigma[k, k, :])
            ub = std_xdist.mu[k, :] + 2. * np.sqrt(std_xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

            plt.plot(t, robust_xdist.mu[k, :], '-r')
            lb = robust_xdist.mu[k, :] - 2. * np.sqrt(robust_xdist.sigma[k, k, :])
            ub = robust_xdist.mu[k, :] + 2. * np.sqrt(robust_xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='red', alpha=0.1)

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + k + 1)

            plt.plot(t, std_udist.mu[k, :], '-g')
            lb = std_udist.mu[k, :] - 2. * np.sqrt(std_udist.sigma[k, k, :])
            ub = std_udist.mu[k, :] + 2. * np.sqrt(std_udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='green', alpha=0.1)

            plt.plot(t, robust_udist.mu[k, :], '-m')
            lb = robust_udist.mu[k, :] - 2. * np.sqrt(robust_udist.sigma[k, k, :])
            ub = robust_udist.mu[k, :] + 2. * np.sqrt(robust_udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='magenta', alpha=0.1)

        plt.show()

        plt.figure(figsize=(6, 12))
        plt.suptitle("Standard vs Robust Ctl: Perturbed Env")

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, k + 1)

            plt.plot(t, std_worst_xdist.mu[k, :], '-b')
            lb = std_worst_xdist.mu[k, :] - 2. * np.sqrt(std_worst_xdist.sigma[k, k, :])
            ub = std_worst_xdist.mu[k, :] + 2. * np.sqrt(std_worst_xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

            plt.plot(t, robust_worst_xdist.mu[k, :], '-r')
            lb = robust_worst_xdist.mu[k, :] - 2. * np.sqrt(robust_worst_xdist.sigma[k, k, :])
            ub = robust_worst_xdist.mu[k, :] + 2. * np.sqrt(robust_worst_xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='red', alpha=0.1)

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + k + 1)

            plt.plot(t, std_worst_udist.mu[k, :], '-g')
            lb = std_worst_udist.mu[k, :] - 2. * np.sqrt(std_worst_udist.sigma[k, k, :])
            ub = std_worst_udist.mu[k, :] + 2. * np.sqrt(std_worst_udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='green', alpha=0.1)

            plt.plot(t, robust_worst_udist.mu[k, :], '-m')
            lb = robust_worst_udist.mu[k, :] - 2. * np.sqrt(robust_worst_udist.sigma[k, k, :])
            ub = robust_worst_udist.mu[k, :] + 2. * np.sqrt(robust_worst_udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='magenta', alpha=0.1)

        plt.show()

        plt.figure(figsize=(6, 12))
        plt.suptitle("Standard vs Robust Ctl: Feedback Controller")

        for i in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, i + 1)
            plt.plot(self.ctl.K[0, i, ...], color='r', linestyle='None', marker='o', markersize=3)
            plt.plot(std_ctl.K[0, i, ...], color='k')

        for i in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + i + 1)
            plt.plot(self.ctl.kff[i, ...], color='r', linestyle='None', marker='o', markersize=3)
            plt.plot(std_ctl.kff[i, ...], color='k')

        plt.show()

    def run(self, nb_iter=10, verbose=False):

        _trace = []

        # current state distribution
        self.xdist, self.udist, self.xudist = self.cubature_forward_pass(self.ctl, self.nominal)

        # get quadratic cost around mean traj.
        self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)

        # mean objective under current ctrl.
        _return = self.cost.evaluate(self.xdist, self.udist)
        self.last_return = np.sum(_return)

        _trace.append(self.last_return)

        for iter in range(nb_iter):
            # worst-case parameter optimization
            self.beta, _, _ = self.parameter_dual_optimization(self.beta, iters=50)

            agcost = self.parameter_augment_cost(self.beta)

            # # initialize adversial xdist with wide param dist.
            # param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
            # q_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

            # initial adversial xdist. with policy xdist.
            q_xdist = deepcopy(self.xdist)

            # first iteration to establish initial conv_kl
            self.param, _, _ = self.parameter_backward_pass(self.beta, agcost, q_xdist)
            p_xdist, _, _ = self.cubature_forward_pass(self.ctl, self.param)

            # conergence of inner loop
            xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                            q_xdist.mu, q_xdist.sigma,
                                            self.dm_state, self.nb_steps + 1)
            while np.any(xdist_kl > 1e-3):
                self.param, _, _ = self.parameter_backward_pass(self.beta, agcost, q_xdist)
                p_xdist, _, _ = self.cubature_forward_pass(self.ctl, self.param)

                # check convergence of loop
                xdist_kl = self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma,
                                                q_xdist.mu, q_xdist.sigma,
                                                self.dm_state, self.nb_steps + 1)

                # interpolate between distributions
                q_xdist.mu, q_xdist.sigma = self.interp_gauss_kl(q_xdist.mu, q_xdist.sigma,
                                                                 p_xdist.mu, p_xdist.sigma, 1e-1)

            param_kl = np.sum(self.parameter_kldiv(self.param))

            if self.kl_stepwise:
                alpha_init = 1e8 * np.ones((self.nb_steps,))
                alpha_bounds = ((1e-16, 1e16), ) * self.nb_steps
            else:
                alpha_init = 1e8 * np.ones((1,))
                alpha_bounds = ((1e-16, 1e16), ) * 1

            # policy optimization
            res = sc.optimize.minimize(self.policy_dual, alpha_init,
                                       method='L-BFGS-B', jac=True,
                                       bounds=alpha_bounds,
                                       options={'disp': False, 'maxiter': 100000,
                                                'ftol': 1e-12, 'iprint': 99})
            self.alpha = res.x

            # re-compute after opt.
            policy_agcost = self.policy_augment_cost(self.alpha)
            lgc, xvalue, xuvalue, diverge = self.policy_backward_pass(self.alpha, policy_agcost)
            worst_xdist, worst_udist, worst_xudist = self.cubature_forward_pass(lgc, self.param)

            # check kl constraint
            policy_kl = self.policy_kldiv(lgc, worst_xdist)
            if not self.kl_stepwise:
                policy_kl = np.sum(policy_kl)

            if np.all((policy_kl - self.policy_kl_bound) < 0.25 * self.policy_kl_bound) \
                    or np.all(policy_kl < self.policy_kl_bound):
                # update controller
                self.ctl = lgc

                # update value functions
                self.vfunc, self.qfunc = xvalue, xuvalue

                # current state distribution
                self.xdist, self.udist, self.xudist = self.cubature_forward_pass(self.ctl, self.param)

                # trajectory return
                _return = self.cost.evaluate(self.xdist, self.udist)

                # get quadratic cost around mean traj.
                self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)

                # mean objective under last dists.
                _trace.append(_return)

                # update last return to current
                self.last_return = _return

                if verbose:
                    if iter == 0:
                        print("%9s %8s %7s %8s %8s" % ("", "param_kl", "", "policy_kl", ""))
                        print("%6s %6s %6s %2s %6s %6s %12s" % ("iter", "req.", "act.", "", "req.", "act.", "return"))
                    print("%6i %.1e %.1e %6.2f %6.2f %12.2f" % (iter, self.param_kl_bound, param_kl,
                                                                np.sum(self.policy_kl_bound), np.sum(policy_kl),
                                                                _return))
            else:
                print("Something is wrong, KL not satisfied: ", "req", np.sum(self.policy_kl_bound),
                                                                "act.", np.sum(policy_kl))
                if self.kl_stepwise:
                    self.alpha = 1e8 * np.ones((self.nb_steps,))
                else:
                    self.alpha = 1e8 * np.ones((1,))

        return _trace
