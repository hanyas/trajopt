import autograd.numpy as np

import scipy as sc
from scipy import optimize

from trajopt.rgps.objects import Gaussian, QuadraticCost
from trajopt.rgps.objects import LearnedLinearGaussianDynamics, AnalyticalQuadraticCost
from trajopt.rgps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.rgps.objects import LinearGaussianControl
from trajopt.rgps.objects import MatrixNormalParameters

from trajopt.rgps.core import kl_divergence, quad_expectation, augment_cost
from trajopt.rgps.core import cubature_forward_pass, policy_backward_pass, parameter_backward_pass


# Model-free Robust Guided Policy Search
class MFRGPS:

    def __init__(self, env, nb_steps,
                 init_state, init_action_sigma=1.,
                 policy_kl_bound=0.1,
                 param_kl_bound=100):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_noise = self.env.unwrapped.noise
        self.env_cost = self.env.unwrapped.cost
        self.env_init = init_state

        self.ulim = self.env.action_space.high

        self.dm_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]
        self.dm_param = self.dm_state * (self.dm_act + self.dm_state + 1)

        self.nb_steps = nb_steps

        self.policy_kl_bound = policy_kl_bound
        self.param_kl_bound = param_kl_bound

        self.alpha = np.array([-1e4])
        self.beta = np.array([1e3])

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init

        self.udist = Gaussian(self.dm_act, self.nb_steps)
        self.xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        self.dyn = LearnedLinearGaussianDynamics(self.dm_state, self.dm_act, self.nb_steps)
        self.ctl = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps, init_action_sigma)

        self.param = MatrixNormalParameters(self.dm_state, self.dm_act, self.nb_steps)

        _A = self.dyn.A.reshape(self.dm_state ** 2, self.nb_steps, order='F')
        _B = self.dyn.B.reshape(self.dm_state * self.dm_act, self.nb_steps, order='F')
        _c = self.dyn.c
        self.param.mu = np.vstack((_A, _B, _c))
        for t in range(self.nb_steps):
            self.param.sigma[..., t] = 1e-8 * np.eye(self.dm_param)

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.dm_state, self.dm_act, self.nb_steps + 1)

        self.last_return = - np.inf

        self.data = {}

    def sample_param(self, t, worst=False):
        mu = self.mu_param if worst else self.mu_param_nom
        sigma = self.sigma_param if worst else self.sigma_param_nom

        param_vec = np.random.multivariate_normal(mu[:, t], sigma[:, :, t])
        mat = param_vec.reshape((self.dm_state, -1), order='F')
        A = mat[:, :self.dm_state]
        B = mat[:, self.dm_state:self.dm_state + self.dm_act]
        c = mat[:, -1]

        return A, B, c

    def sample(self, nb_episodes, stoch=True, use_model=False, worst=False):
        data = {'x': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'u': np.zeros((self.dm_act, self.nb_steps, nb_episodes)),
                'xn': np.zeros((self.dm_state, self.nb_steps, nb_episodes)),
                'c': np.zeros((self.nb_steps + 1, nb_episodes))}

        for n in range(nb_episodes):
            x = self.env.reset()

            for t in range(self.nb_steps):
                u = self.ctl.sample(x, t, stoch)
                u = np.clip(u, -self.ulim, self.ulim)
                data['u'][..., t, n] = u

                # expose true reward function
                c = self.env_cost(x, u)
                data['c'][t] = c

                data['x'][..., t, n] = x

                if use_model:
                    A, B, c = self.sample_param(t, worst)
                    x = A @ x + B @ u + c
                else:
                    x, _, _, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))

                data['xn'][..., t, n] = x

            c = self.env_cost(x, np.zeros((self.dm_act, )))
            data['c'][-1, n] = c

        return data

    def forward_pass(self, lgc, param):
        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)
        xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        mu = param.mu.reshape((self.dm_state, -1, self.nb_steps), order='F')
        A = mu[:, :self.dm_state, :]
        B = mu[:, self.dm_state:self.dm_state + self.dm_act, :]
        c = mu[:, -1, :]

        xdist.mu, xdist.sigma,\
        udist.mu, udist.sigma,\
        xudist.mu, xudist.sigma = cubature_forward_pass(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                                        A, B, c, sigma_dyn, param.sigma,
                                                        lgc.K, lgc.kff, lgc.sigma,
                                                        self.dm_state, self.dm_act, self.nb_steps)

        return xdist, udist, xudist

    def backward_pass(self, alpha, agcost):
        lgc = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu, xuvalue.q0, xuvalue.q0_softmax,\
        xvalue.V, xvalue.v, xvalue.v0, xvalue.v0_softmax,\
        lgc.K, lgc.kff, lgc.sigma, diverge = robust_backward_pass(agcost.Cxx, agcost.cx, agcost.Cuu,
                                                           agcost.cu, agcost.Cxu, agcost.c0,
                                                           self.mu_param, self.sigma_param,
                                                           alpha, self.dm_state, self.dm_act, self.nb_steps)
        return lgc, xvalue, xuvalue, diverge

    def augment_cost(self, alpha):
        agcost = QuadraticCost(self.dm_state, self.dm_act, self.nb_steps + 1)
        agcost.Cxx, agcost.cx, agcost.Cuu,\
        agcost.cu, agcost.Cxu, agcost.c0 = augment_cost(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                                        self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                        self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                        alpha, self.dm_state, self.dm_act, self.nb_steps)
        return agcost

    def parameter_backward_pass(self, alpha, xudist):
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        xvalue.V, xvalue.v, xvalue.v0,\
        mu_param, sigma_param, diverge = parameter_backward_pass(xudist.mu, xudist.sigma,
                                                                 self.cost.cx, self.cost.Cxx, self.cost.Cuu,
                                                                 self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                                 self.mu_param_nom, self.sigma_param_nom,
                                                                 self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                                 alpha, self.dm_state, self.dm_act, self.nb_steps)
        return mu_param, sigma_param, xvalue, diverge

    def parameter_kldiv(self, mu_param, sigma_param):
        return self.gaussians_kldiv(mu_param, sigma_param, self.mu_param_nom, self.sigma_param_nom)

    def gaussians_kldiv(self, mu1, sigma1, mu2, sigma2):
        prec2 = np.linalg.inv(sigma2.T).T

        delta_mu = (mu2 - mu1)
        quad_term = np.einsum('it,ijt,jt->t', delta_mu, prec2, delta_mu)
        trace_term = np.trace(np.einsum('ijt,jlt->ilt', prec2, sigma1))
        log_det_term = np.log(np.linalg.det(sigma2.T)/np.linalg.det(sigma1.T))

        return 0.5*(trace_term + quad_term + log_det_term - mu1.shape[0])

    def parameter_dual(self, alpha, verbose=False):
        from copy import deepcopy

        xudist = deepcopy(self.xudist)
        xvalue, mu_param, sigma_param = None, None, None
        for _ in range(5):
            mu_param, sigma_param, xvalue, diverge = self.parameter_backward_pass(alpha, xudist)
            xdist, udist, xudist = self.forward_pass(self.ctl, mu_param, sigma_param)

        # dual expectation
        dual = quad_expectation(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        dual += alpha * (np.sum(self.parameter_kldiv(mu_param, sigma_param)[:-1]) - self.param_kl_bound)

        # gradient
        grad = np.sum(self.parameter_kldiv(mu_param, sigma_param)[:-1]) - self.param_kl_bound

        return np.array([dual]),  np.array([grad])

    def dual(self, alpha):
        # augmented cost
        agcost = self.augment_cost(alpha)

        # backward pass
        lgc, xvalue, xuvalue, diverge = self.backward_pass(alpha, agcost)

        # forward pass
        xdist, udist, xudist = self.forward_pass(lgc, self.mu_param, self.sigma_param)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0_softmax[..., 0])
        dual += alpha * self.kl_bound

        # gradient
        grad = self.kl_bound - self.kldiv(lgc, xdist)

        return -1. * np.array([dual]), -1. * np.array([grad])

    def kldiv(self, lgc, xdist):
        return kl_divergence(lgc.K, lgc.kff, lgc.sigma,
                             self.ctl.K, self.ctl.kff, self.ctl.sigma,
                             xdist.mu, xdist.sigma,
                             self.dm_state, self.dm_act, self.nb_steps)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, k + 1)
            plt.plot(t, self.xdist.mu[k, :], '-b')
            lb = self.xdist.mu[k, :] - 2. * np.sqrt(self.xdist.sigma[k, k, :])
            ub = self.xdist.mu[k, :] + 2. * np.sqrt(self.xdist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + k + 1)
            plt.plot(t, self.udist.mu[k, :], '-g')
            lb = self.udist.mu[k, :] - 2. * np.sqrt(self.udist.sigma[k, k, :])
            ub = self.udist.mu[k, :] + 2. * np.sqrt(self.udist.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='green', alpha=0.1)

        plt.show()

    def run(self, nb_episodes, nb_iter=10, verbose=False, plot_dual=False):
        _trace = []

        # run init controller
        self.data = self.sample(nb_episodes)

        # fit time-variant linear dynamics
        self.dyn.learn(self.data)

        # update nominal parameter distribution
        self.update_param_nom()

        # initialize worst-case distribution with nominal parameters
        self.mu_param = self.mu_param_nom
        self.sigma_param = self.sigma_param_nom

        # current state distribution
        self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl, self.pdist)

        # get quadratic cost around mean traj.
        self.cost.taylor_expansion(self.xdist.mu, self.udist.mu)

        # mean objective under current ctrl.
        self.last_return = np.mean(np.sum(self.data['c'], axis=0))
        _trace.append(self.last_return)

        dv, kv = [], []
        self.alpha_param = np.array([-1e8])
        for iter in range(nb_iter):
            param_kl = 0.
            # while param_kl < self.param_kl_bound:
            for _ in range(1):
                param_res = sc.optimize.minimize(lambda a: self.parameter_dual(a)[0].flatten(), self.alpha_param,
                                                 method='SLSQP', jac=False,
                                                 bounds=((-1e8, -1e-8), ),
                                                 options={'disp': False, 'maxiter': 10,
                                                          'ftol': 1e-6})

                # param_res = sc.optimize.minimize(self.parameter_dual, self.alpha_param,
                #                                  method='L-BFGS-B',
                #                                  jac=True, callback=None,
                #                                  bounds=((-1e8, -1e-8), ),
                #                                  options={'disp': False, 'maxiter': 1000,
                #                                           'ftol': 1e-6})

                self.alpha_param = param_res.x
                _mu_param, _sigma_param, _, _ = self.parameter_backward_pass(self.alpha_param, self.xudist)
                self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl, _mu_param, _sigma_param)

                param_kl = np.sum(self.parameter_kldiv(_mu_param, _sigma_param)[:-1])
                # print(param_res.x, param_res.fun, param_res.jac, param_kl)
                # dv.append(param_res.fun.flatten().item())
                # kv.append(param_kl)

            # use scipy optimizer
            res = sc.optimize.minimize(self.dual, np.array([-1e8]),
                                       method='SLSQP',
                                       jac=True, callback=None,
                                       bounds=((-1e8, -1e-8), ),
                                       options={'disp': False, 'maxiter': 100,
                                                'ftol': 1e-6})
            self.alpha = res.x

            if plot_dual:
                try:
                    self.plot_dual(self.parameter_dual, np.log10(-0.5 * param_res.x)[0], np.log10(-10 * param_res.x)[0])
                except ValueError:
                    self.plot_dual(self.parameter_dual, np.log10(-0.8 * param_res.x)[0], np.log10(-2 * param_res.x)[0])

            # re-compute after opt.
            agcost = self.augment_cost(self.alpha)
            lgc, xvalue, xuvalue, diverge = self.backward_pass(self.alpha, agcost)

            # current return
            _return = np.mean(np.sum(self.data['c'], axis=0))

            # get expected improvment:
            xdist, udist, xudist = self.forward_pass(lgc, self.mu_param, self.sigma_param)
            _expected_return = self.cost.evaluate(xdist.mu, udist.mu)

            # expected vs actual improvement
            _expected_imp = self.last_return - _expected_return
            _actual_imp = self.last_return - _return

            # check kl constraint
            policy_kl = self.kldiv(lgc, xdist)
            if (policy_kl - self.policy_kl_bound) < 0.25 * self.policy_kl_bound:
                # update controller
                self.ctl = lgc

                # update value functions
                self.vfunc, self.qfunc = xvalue, xuvalue

                # run current controller
                self.data = self.sample(nb_episodes)

                # fit time-variant linear dynamics
                self.dyn.learn(self.data)

                # update nominal parameter distribution
                self.update_param_nom()

                # current state distribution
                self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl, self.mu_param, self.sigma_param)

                # get quadratic cost around mean traj.
                self.cost.taylor_expansion(self.xdist.mu, self.udist.mu)

                # mean objective under last dists.
                _trace.append(_return)

                # update last return to current
                self.last_return = _return

            else:
                print("Something is wrong, KL not satisfied")
                self.alpha = np.array([-1e4])

            if verbose:
                if iter == 0:
                    print("%6s %12s %12s %12s" %("", "policy kl", "param kl", ""))
                    print("%6s %6s %6s %6s %6s %12s" %("iter", "req.", "act.",  "req.", "act.", "return"))

                print("%6i %6.2f %6.2f %6.2f %6.2f %12.2f" %(iter, self.kl_bound, kl,  self.param_kl_bound, param_kl, _return))

        return _trace

    def plot_dual(self, dual_fun, elow=0,ehigh=8,logax=True):
        import matplotlib.pyplot as plt
        import scipy as sc

        res = sc.optimize.minimize(dual_fun, np.array([-1*10**((ehigh-elow)/2)]),
                                        method='L-BFGS-B',
                                        jac=True,
                                        callback=None,
                                        bounds=((-10**(ehigh), -10**(elow)), ),
                                        options={'disp': True, 'maxiter': 100,
                                                    'ftol': 1e-6})

        print(res.x)
        fig, ax1 = plt.subplots()
        if logax:
            alphas = np.logspace(elow,ehigh).flatten()
            ax1.set_xscale('log')
        else:
            alphas = np.linspace(10**elow,10**ehigh).flatten()

        dual_obj = lambda alpha: dual_fun(alpha)[0]
        eps = np.sqrt(np.finfo(float).eps)
        grad_findiff = -np.hstack([sc.optimize.approx_fprime(np.array([-alpha]), dual_obj, [eps]) for alpha in alphas])

        obj, grad = zip(*[dual_fun(-alpha) for alpha in alphas])
        obj = np.hstack(obj)
        grad = np.hstack(grad)
        grad *= -1
        ax1.plot(alphas, obj, 'b')
        ax1.set_ylabel("objective",color='b')
        ax1.set_xlabel("alpha")
        ax1.axvline(-res.x, color='k', ls="--")
        ax2 = ax1.twinx()
        ax2.set_ylabel("gradient",color='r')
        ax2.plot(alphas,grad,'r')
        ax2.plot(alphas,grad_findiff,'r--')
        ax2.axhline(0,color='k')
        plt.show()