import autograd.numpy as np

import scipy as sc
from scipy import optimize

from copy import deepcopy

from trajopt.rgps.objects import Gaussian, QuadraticCost
from trajopt.rgps.objects import AnalyticalQuadraticCost
from trajopt.rgps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.rgps.objects import LinearGaussianControl
from trajopt.rgps.objects import LearnedProbabilisticLinearDynamicsWithKnownNoise
from trajopt.rgps.objects import MatrixNormalParameters

from trajopt.rgps.core import kl_divergence, quad_expectation
from trajopt.rgps.core import policy_augment_cost, policy_backward_pass
from trajopt.rgps.core import parameter_augment_cost, parameter_backward_pass
from trajopt.rgps.core import parameter_dual_regularization, regularized_parameter_backward_pass
from trajopt.rgps.core import parameter_augment_cost
from trajopt.rgps.core import cubature_forward_pass


# Model-free Robust Guided Policy Search
class MFRGPS:

    def __init__(self, env, nb_steps,
                 init_state, init_action_sigma=1.,
                 policy_kl_bound=0.1, param_kl_bound=100,
                 activation=None, slew_rate=False,
                 action_penalty=None, prior=None):

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

        self.policy_kl_bound = policy_kl_bound
        self.param_kl_bound = param_kl_bound

        self.alpha = np.array([1e4])
        self.beta = np.array([1e2])

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init

        self.udist = Gaussian(self.dm_act, self.nb_steps)
        self.xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)
        self.pfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        self.nominal = LearnedProbabilisticLinearDynamicsWithKnownNoise(self.dm_state, self.dm_act, self.nb_steps,
                                                                        self.env_noise, prior)
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

    def sample(self, nb_episodes, stoch=True):
        data = {'x': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'u': np.zeros((self.dm_act, self.nb_steps, nb_episodes)),
                'xn': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'c': np.zeros((self.nb_steps + 1, nb_episodes))}

        for n in range(nb_episodes):
            x = self.env.reset()

            for t in range(self.nb_steps):
                u = self.ctl.sample(x, t, stoch)
                # u = np.clip(u, -self.ulim, self.ulim)
                data['u'][..., t, n] = u

                # expose true reward function
                c = self.env_cost(x, u, data['u'][..., t - 1, n], self.weighting[t])
                data['c'][t] = c

                data['x'][..., t, n] = x
                x, _, _, _ = self.env.step(u)
                data['xn'][..., t, n] = x

            c = self.env_cost(x, np.zeros((self.dm_act, )),  np.zeros((self.dm_act, )), self.weighting[-1])
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

        return xdist, udist, xudist

    def policy_augment_cost(self, alpha):
        agcost = QuadraticCost(self.dm_state, self.dm_act, self.nb_steps + 1)
        agcost.Cxx, agcost.cx, agcost.Cuu,\
        agcost.cu, agcost.Cxu, agcost.c0 = policy_augment_cost(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                                               self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                               self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                               alpha, self.dm_state, self.dm_act, self.nb_steps)
        return agcost

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
        dual -= alpha * self.policy_kl_bound

        # dual gradient
        grad = self.policy_kldiv(lgc, xdist) - self.policy_kl_bound

        return -1. * np.array([dual]), -1. * np.array([grad])

    def policy_kldiv(self, lgc, xdist):
        return kl_divergence(lgc.K, lgc.kff, lgc.sigma,
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
        if diverge:
            return np.nan, np.nan
        p_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

        # conergence of inner loop
        xdist_kl = np.sum(self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma, q_xdist.mu, q_xdist.sigma))
        while xdist_kl > 1e-3:
            param, xvalue, diverge = self.parameter_backward_pass(beta, agcost, q_xdist)
            if diverge:
                return np.nan, np.nan
            p_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

            # check convergence of loop
            xdist_kl = np.sum(self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma, q_xdist.mu, q_xdist.sigma))

            # interpolate between distributions
            for t in range(1, self.nb_steps + 1):
                q_xdist.mu[..., t], q_xdist.sigma[..., t] = self.interp_gauss_w2(q_xdist.mu[..., t], q_xdist.sigma[..., t],
                                                                                 p_xdist.mu[..., t], p_xdist.sigma[..., t],
                                                                                 np.array([1e-1]))
        # dual expectation
        dual = quad_expectation(q_xdist.mu[..., 0], q_xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        dual += beta * (np.sum(self.parameter_kldiv(param)) - self.param_kl_bound)

        # dual gradient
        grad = np.sum(self.parameter_kldiv(param)) - self.param_kl_bound

        return -1. * np.array([dual]), -1. * np.array([grad])

    def parameter_dual_regularization(self, pdist, qdist, kappa):
        regcost = QuadraticCost(self.dm_state, self.dm_state, self.nb_steps + 1)

        regcost.Cxx, regcost.cx, regcost.c0 = parameter_dual_regularization(pdist.mu, pdist.sigma,
                                                                            qdist.mu, qdist.sigma,
                                                                            kappa, self.dm_state, self.nb_steps)
        return regcost

    def regularized_parameter_backward_pass(self, beta, agcost, xdist, regcost):
        param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        xvalue.V, xvalue.v, xvalue.v0,\
        param.mu, param.sigma, diverge = regularized_parameter_backward_pass(xdist.mu, xdist.sigma,
                                                                             self.ctl.K, self.ctl.kff, self.ctl.sigma, self.noise,
                                                                             self.cost.cx, self.cost.Cxx, self.cost.Cuu,
                                                                             self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                                             agcost.Cxx, agcost.cx, agcost.c0,
                                                                             regcost.Cxx, regcost.cx, regcost.c0,
                                                                             beta, self.dm_state, self.dm_act, self.dm_param,
                                                                             self.nb_steps)
        return param, xvalue, diverge

    def regularized_parameter_dual(self, beta, kappa):
        agcost = self.parameter_augment_cost(beta)

        q_xdist = deepcopy(self.xdist)
        # p_xdist = deepcopy(self.xdist)

        param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)
        p_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

        regcost = self.parameter_dual_regularization(p_xdist, q_xdist, kappa)
        param, xvalue, diverge = self.regularized_parameter_backward_pass(beta, agcost, p_xdist, regcost)
        if diverge:
            return np.nan, np.nan

        # conergence of inner loop
        xdist_kl = np.sum(self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma, q_xdist.mu, q_xdist.sigma))
        while xdist_kl > 1e-3:
            regcost = self.parameter_dual_regularization(p_xdist, q_xdist, kappa)
            param, xvalue, diverge = self.regularized_parameter_backward_pass(beta, agcost, p_xdist, regcost)
            if diverge:
                return np.nan, np.nan

            q_xdist = deepcopy(p_xdist)
            p_xdist, _, _ = self.cubature_forward_pass(self.ctl, param)

            # check convergence of loop
            xdist_kl = np.sum(self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma, q_xdist.mu, q_xdist.sigma))

        # dual expectation
        dual = quad_expectation(q_xdist.mu[..., 0], q_xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        dual += beta * (np.sum(self.parameter_kldiv(param)) - self.param_kl_bound)
        dual -= kappa * xdist_kl

        # dual gradient
        grad = np.sum(self.parameter_kldiv(param)) - self.param_kl_bound

        return -1. * np.array([dual]), -1. * np.array([grad])

    def parameter_dual_optimization(self, beta, iters=10, regularized=False, kappa=None):
        if regularized:
            assert kappa is not None

        min_beta, max_beta = 1e-4, 1e4

        # adapted from:
        # https://github.com/cbfinn/gps/.../traj_opt_lqr_python.py

        dual, grad = np.nan, np.nan
        for i in range(iters):
            if regularized:
                dual, grad = self.regularized_parameter_dual(beta, kappa)
            else:
                dual, grad = self.parameter_dual(beta)

            if not np.isnan(dual) and not np.isnan(grad):
                if abs(grad) < 0.1 * self.param_kl_bound:
                    return beta, dual, grad
                else:
                    if grad > 0:  # beta too large
                        max_beta = beta
                        geom = np.sqrt(min_beta * max_beta)
                        beta = max(geom, 0.1 * max_beta)
                    else:  # beta too small
                        min_beta = beta
                        geom = np.sqrt(min_beta * max_beta)
                        beta = min(geom, 5.0 * min_beta)
            else:
                min_beta = beta
                geom = np.sqrt(min_beta * max_beta)
                beta = min(geom, 10.0 * min_beta)

        return beta, dual, grad

    def parameter_kldiv(self, param):
        return self.gaussians_kldiv(param.mu, param.sigma, self.nominal.mu, self.nominal.sigma)

    @staticmethod
    def interp_gauss_kl(mu_0, sigma_0, mu_1, sigma_1, a):
        sigma_p = np.linalg.inv(a * np.linalg.inv(sigma_1) + (1. - a) * np.linalg.inv(sigma_0))
        mu_p = sigma_p @ (a * np.linalg.inv(sigma_1) @ mu_1 + (1. - a) * np.linalg.inv(sigma_0) @ mu_0)
        return mu_p, sigma_p

    @staticmethod
    def interp_gauss_w2(mu_0, sigma_0, mu_1, sigma_1, a):
        mu_p = (1. - a) * mu_0 + a * mu_1
        sigma_0_chol = sc.linalg.sqrtm(sigma_0)
        sigma_0_chol_inv = np.linalg.inv(sigma_0_chol)
        _sigma = (1. - a) * sigma_0 + a * sc.linalg.sqrtm(sigma_0_chol @ sigma_1 @ sigma_0_chol)
        sigma_p = sigma_0_chol_inv @ _sigma @ _sigma @ sigma_0_chol_inv
        return mu_p, sigma_p

    @staticmethod
    def gaussians_kldiv(mu_p, sigma_p, mu_q, sigma_q):
        assert len(sigma_p) == len(sigma_q)
        assert len(mu_p) == len(mu_q)

        length = sigma_p.shape[-1]
        kl = np.zeros((length, ))
        for t in range(length):
            lmbda_q = np.linalg.inv(sigma_q[..., t])

            diff_mu = (mu_q[..., t] - mu_p[..., t])[:, None]
            quad_term = diff_mu.T @ lmbda_q @ diff_mu
            trace_term = np.trace(lmbda_q @ sigma_p[..., t])
            log_det_term = np.log(np.linalg.det(sigma_q[..., t]) / np.linalg.det(sigma_p[..., t]))

            kl[t] = 0.5 * (trace_term + quad_term + log_det_term - mu_p.shape[0])
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

    def run(self, nb_episodes, nb_iter=10,
            verbose=False, debug_dual=False):

        _trace = []

        # run init controller
        self.data = self.sample(nb_episodes)

        # leanr posterior over dynamics
        self.nominal.learn(self.data)

        # current state distribution
        self.xdist, self.udist, self.xudist = self.cubature_forward_pass(self.ctl, self.nominal)

        # get quadratic cost around mean traj.
        self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)

        # mean objective under current ctrl.
        self.last_return = np.mean(np.sum(self.data['c'], axis=0))
        _trace.append(self.last_return)

        for iter in range(nb_iter):
            # worst-case parameter optimization
            self.beta, _, _ = self.parameter_dual_optimization(np.array([1e4]), iters=200)

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
            xdist_kl = np.sum(self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma, q_xdist.mu, q_xdist.sigma))
            while xdist_kl > 1e-3:
                self.param, _, _ = self.parameter_backward_pass(self.beta, agcost, q_xdist)
                p_xdist, _, _ = self.cubature_forward_pass(self.ctl, self.param)

                # check convergence of loop
                xdist_kl = np.sum(self.gaussians_kldiv(p_xdist.mu, p_xdist.sigma, q_xdist.mu, q_xdist.sigma))

                # interpolate between distributions
                for t in range(1, self.nb_steps + 1):
                    q_xdist.mu[..., t], q_xdist.sigma[..., t] = self.interp_gauss_w2(q_xdist.mu[..., t], q_xdist.sigma[..., t],
                                                                                     p_xdist.mu[..., t], p_xdist.sigma[..., t],
                                                                                     np.array([1e-1]))

            param_kl = np.sum(self.parameter_kldiv(self.param))

            # policy optimization
            res = sc.optimize.minimize(self.policy_dual, np.array([1e4]),
                                       method='L-BFGS-B',
                                       jac=True,
                                       bounds=((1e-8, 1e32), ),
                                       options={'disp': False, 'maxiter': 10000,
                                                'ftol': 1e-6, 'iprint': 0})
            self.alpha = res.x

            if debug_dual:
                try:
                    self.plot_dual(self.policy_dual, np.log10(0.01 * res.x), np.log10(100. * res.x), numdiff=False)
                except ValueError:
                    self.plot_dual(self.policy_dual, np.log10(0.5 * res.x), np.log10(2. * res.x))

            # re-compute after opt.
            policy_agcost = self.policy_augment_cost(self.alpha)
            lgc, xvalue, xuvalue, diverge = self.policy_backward_pass(self.alpha, policy_agcost)

            # get expected improvment:
            worst_xdist, worst_udist, worst_xudist = self.cubature_forward_pass(lgc, self.param)
            nominal_xdist, nominal_udist, nominal_xudist = self.cubature_forward_pass(lgc, self.nominal)

            _expected_worst_return = self.cost.evaluate(worst_xdist.mu, worst_udist.mu)
            _expected_nominal_return = self.cost.evaluate(nominal_xdist.mu, nominal_udist.mu)

            # check kl constraint
            policy_kl = self.policy_kldiv(lgc, worst_xdist)
            if np.abs(policy_kl - self.policy_kl_bound) < 0.25 * self.policy_kl_bound:
                # update controller
                self.ctl = lgc

                # update value functions
                self.vfunc, self.qfunc = xvalue, xuvalue

                # run current controller
                self.data = self.sample(nb_episodes)

                # current return
                _actual_return = np.mean(np.sum(self.data['c'], axis=0))

                # expected vs actual improvement
                _actual_imp = self.last_return - _actual_return
                _expected_worst_imp = self.last_return - _expected_worst_return

                # leanr posterior over dynamics
                self.nominal.learn(self.data)

                # current state distribution
                self.xdist, self.udist, self.xudist = self.cubature_forward_pass(self.ctl, self.param)

                # get quadratic cost around mean traj.
                self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)

                # mean objective under last dists.
                _trace.append(_actual_return)

                # update last return to current
                self.last_return = _actual_return

                if verbose:
                    if iter == 0:
                        print("%9s %8s %4s %8s %8s" % ("", "param_kl", "", "policy_kl", ""))
                        print("%6s %6s %6s %6s %6s %12s" % ("iter", "req.", "act.", "req.", "act.", "return"))
                    print("%6i %6.2f %6.2f %6.2f %6.2f %12.2f" % (iter, self.param_kl_bound, param_kl,
                                                                  self.policy_kl_bound, policy_kl, _actual_return))
            else:
                print("Something is wrong, KL not satisfied")
                self.alpha = np.array([1e4])

        return _trace

    @staticmethod
    def plot_dual(dual_fun, opt, elow=0, ehigh=8, logax=True, numdiff=False):
        import matplotlib.pyplot as plt
        import scipy as sc

        fig, ax1 = plt.subplots()
        if logax:
            values = np.logspace(elow, ehigh).flatten()
            ax1.set_xscale('log')
        else:
            values = np.linspace(10**elow, 10**ehigh).flatten()

        grad_fdiff = None
        if numdiff:
            dual_value = lambda alpha: dual_fun(alpha)[0].item()
            eps = 1e-8 * np.sqrt(np.finfo(float).eps)
            grad_fdiff = np.hstack([sc.optimize.approx_fprime(np.array([val]), dual_value, [eps]) for val in values])

        obj, grad = zip(*[dual_fun(val) for val in values])
        obj, grad = np.hstack(obj), np.hstack(grad)

        ax1.plot(values, obj, 'b')
        ax1.set_ylabel("objective", color='b')
        ax1.set_xlabel("alpha")
        ax1.axvline(opt, color='k', ls="--")

        ax2 = ax1.twinx()
        ax2.set_ylabel("gradient", color='r')
        ax2.plot(values, grad, 'r')
        if numdiff:
            ax2.plot(values, grad_fdiff, 'r--')
        ax2.axhline(0, color='k')

        plt.show()
