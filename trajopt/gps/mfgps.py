import autograd.numpy as np

import scipy as sc
from scipy import optimize

from trajopt.gps.objects import Gaussian, QuadraticCost
from trajopt.gps.objects import LearnedLinearGaussianDynamics, AnalyticalQuadraticCost
from trajopt.gps.objects import LearnedLinearGaussianDynamicsWithKnownNoise
from trajopt.gps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.gps.objects import LinearGaussianControl
from trajopt.gps.objects import pass_alpha_as_vector

from trajopt.gps.core import kl_divergence, quad_expectation, augment_cost
from trajopt.gps.core import forward_pass, backward_pass


class MFGPS:

    def __init__(self, env, nb_steps,
                 init_state, init_action_sigma=1.,
                 kl_bound=0.1, kl_adaptive=False,
                 kl_stepwise=False, activation=None,
                 slew_rate=False, action_penalty=None,
                 dyn_prior=None):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_noise = self.env.unwrapped.sigma
        self.env_cost = self.env.unwrapped.cost
        self.env_init = init_state

        self.ulim = self.env.action_space.high

        self.dm_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # use slew rate penalty or not
        self.env.unwrapped.slew_rate = slew_rate
        if action_penalty is not None:
            self.env.unwrapped.uw = action_penalty * np.ones((self.dm_act, ))

        self.kl_stepwise = kl_stepwise
        if self.kl_stepwise:
            self.kl_base = kl_bound * np.ones((self.nb_steps, ))
            self.kl_bound = kl_bound * np.ones((self.nb_steps, ))
            self.alpha = 1e8 * np.ones((self.nb_steps, ))
        else:
            self.kl_base = kl_bound * np.ones((1, ))
            self.kl_bound = kl_bound * np.ones((1, ))
            self.alpha = 1e8 * np.ones((1, ))

        # kl mult.
        self.kl_adaptive = kl_adaptive
        self.kl_mult = 1.
        self.kl_mult_min = 0.1
        self.kl_mult_max = 5.0

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init

        self.udist = Gaussian(self.dm_act, self.nb_steps)
        self.xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        # self.dyn = LearnedLinearGaussianDynamics(self.dm_state, self.dm_act, self.nb_steps, dyn_prior)
        self.dyn = LearnedLinearGaussianDynamicsWithKnownNoise(self.dm_state, self.dm_act, self.nb_steps,
                                                               self.env_noise, dyn_prior)
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

    def rollout(self, nb_episodes, stoch=True):
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

    def forward_pass(self, lgc):
        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)
        xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        xdist.mu, xdist.sigma,\
        udist.mu, udist.sigma,\
        xudist.mu, xudist.sigma = forward_pass(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                               self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                               lgc.K, lgc.kff, lgc.sigma,
                                               self.dm_state, self.dm_act, self.nb_steps)
        return xdist, udist, xudist

    @pass_alpha_as_vector
    def backward_pass(self, alpha, agcost):
        lgc = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu, xuvalue.q0, \
        xvalue.V, xvalue.v, xvalue.v0, \
        lgc.K, lgc.kff, lgc.sigma, diverge = backward_pass(agcost.Cxx, agcost.cx, agcost.Cuu,
                                                           agcost.cu, agcost.Cxu, agcost.c0,
                                                           self.dyn.A, self.dyn.B, self.dyn.c, self.dyn.sigma,
                                                           alpha, self.dm_state, self.dm_act, self.nb_steps)
        return lgc, xvalue, xuvalue, diverge

    @pass_alpha_as_vector
    def augment_cost(self, alpha):
        agcost = QuadraticCost(self.dm_state, self.dm_act, self.nb_steps + 1)
        agcost.Cxx, agcost.cx, agcost.Cuu,\
        agcost.cu, agcost.Cxu, agcost.c0 = augment_cost(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                                        self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                        self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                        alpha, self.dm_state, self.dm_act, self.nb_steps)
        return agcost

    def dual(self, alpha):
        # augmented cost
        agcost = self.augment_cost(alpha)

        # backward pass
        lgc, xvalue, xuvalue, diverge = self.backward_pass(alpha, agcost)

        # forward pass
        xdist, udist, xudist = self.forward_pass(lgc)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])

        if self.kl_stepwise:
            dual = np.array([dual]) - np.sum(alpha * self.kl_bound)
            grad = self.kldiv(lgc, xdist) - self.kl_bound
        else:
            dual = np.array([dual]) - alpha * self.kl_bound
            grad = np.sum(self.kldiv(lgc, xdist)) - self.kl_bound

        return -1. * dual, -1. * grad

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

    def run(self, nb_learning_episodes,
            nb_evaluation_episodes, nb_iter=10,
            verbose=False):
        _trace = []

        # run init controller
        self.data = self.rollout(nb_learning_episodes)
        eval = self.rollout(nb_evaluation_episodes)

        # fit time-variant linear dynamics
        self.dyn.learn(self.data)

        # current state distribution
        self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl)

        # get quadratic cost around mean traj.
        self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)

        # mean objective under current ctrl.
        self.last_return = np.mean(np.sum(eval['c'], axis=0))
        _trace.append(self.last_return)

        for iter in range(nb_iter):
            if self.kl_stepwise:
                init = 1e8 * np.ones((self.nb_steps,))
                bounds = ((1e-16, 1e16), ) * self.nb_steps
            else:
                init = 1e8 * np.ones((1,))
                bounds = ((1e-16, 1e16), ) * 1

            res = sc.optimize.minimize(self.dual, init,
                                       method='SLSQP',
                                       jac=True,
                                       bounds=bounds,
                                       options={'disp': False, 'maxiter': 10000,
                                                'ftol': 1e-6})
            self.alpha = res.x

            # re-compute after opt.
            agcost = self.augment_cost(self.alpha)
            lgc, xvalue, xuvalue, diverge = self.backward_pass(self.alpha, agcost)

            # get expected improvment:
            xdist, udist, xudist = self.forward_pass(lgc)
            _expected_return = self.cost.evaluate(xdist.mu, udist.mu)

            # check kl constraint
            kl = self.kldiv(lgc, xdist)
            if not self.kl_stepwise:
                kl = np.sum(kl)

            if np.all(np.abs(kl - self.kl_bound) < 0.25 * self.kl_bound):
                # update controller
                self.ctl = lgc

                # update value functions
                self.vfunc, self.qfunc = xvalue, xuvalue

                # run current controller
                self.data = self.rollout(nb_learning_episodes)
                eval = self.rollout(nb_evaluation_episodes)

                # current return
                _return = np.mean(np.sum(eval['c'], axis=0))

                # expected vs actual improvement
                _expected_imp = self.last_return - _expected_return
                _actual_imp = self.last_return - _return

                # update kl multiplier
                if self.kl_adaptive:
                    _mult = _expected_imp / (2. * np.maximum(1e-4, _expected_imp - _actual_imp))
                    _mult = np.maximum(0.1, np.minimum(5.0, _mult))
                    self.kl_mult = np.maximum(np.minimum(_mult * self.kl_mult, self.kl_mult_max), self.kl_mult_min)

                # fit time-variant linear dynamics
                self.dyn.learn(self.data)

                # current state distribution
                self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl)

                # get quadratic cost around mean traj.
                self.cost.taylor_expansion(self.xdist.mu, self.udist.mu, self.weighting)

                # mean objective under last dists.
                _trace.append(_return)

                # update last return to current
                self.last_return = _return

                # update kl bound
                if self.kl_adaptive:
                    self.kl_bound = self.kl_base * self.kl_mult

                if verbose:
                    if iter == 0:
                        print("%6s %8s %8s" % ("", "kl", ""))
                        print("%6s %6s %6s %12s" % ("iter", "req.", "act.", "return"))
                    print("%6i %6.2f %6.2f %12.2f" % (iter, np.sum(self.kl_bound), np.sum(kl), _return))
            else:
                print("Something is wrong, KL not satisfied")
                if self.kl_stepwise:
                    self.alpha = 1e8 * np.ones((self.nb_steps,))
                else:
                    self.alpha = 1e8 * np.ones((1,))

        return _trace
