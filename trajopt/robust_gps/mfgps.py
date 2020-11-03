import autograd.numpy as np

import scipy as sc
from scipy import optimize

from trajopt.robust_gps.objects import Gaussian, QuadraticCost
from trajopt.robust_gps.objects import LearnedLinearGaussianDynamics, AnalyticalQuadraticCost
from trajopt.robust_gps.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.robust_gps.objects import LinearGaussianControl

from trajopt.robust_gps.core import kl_divergence, quad_expectation, augment_cost
from trajopt.robust_gps.core import forward_pass, backward_pass, cubature_forward_pass, robust_backward_pass, parameter_backward_pass


class MFROBGPS:

    def __init__(self, env, nb_steps,
                 init_state, init_action_sigma=1.,
                 kl_bound=0.1, kl_adaptive=False,
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

        self.dm_param = self.dm_state*(self.dm_act + self.dm_state + 1)

        self.nb_steps = nb_steps

        self.param_kl_bound = param_kl_bound

        # total kl over traj.
        self.kl_base = kl_bound
        self.kl_bound = kl_bound

        # kl mult.
        self.kl_adaptive = kl_adaptive
        self.kl_mult = 1.
        self.kl_mult_min = 0.1
        self.kl_mult_max = 5.0

        self.alpha = np.array([-1e4])

        self.alpha_param = np.array([1e3])

        # create state distribution and initialize first time step
        self.xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        self.xdist.mu[..., 0], self.xdist.sigma[..., 0] = self.env_init

        self.udist = Gaussian(self.dm_act, self.nb_steps)
        self.xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        self.dyn = LearnedLinearGaussianDynamics(self.dm_state, self.dm_act, self.nb_steps)
        self.ctl = LinearGaussianControl(self.dm_state, self.dm_act, self.nb_steps, init_action_sigma)

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.dm_state, self.dm_act, self.nb_steps + 1)

        self.last_return = - np.inf

        self.data = {}

        # TODO: get these from dynamics and keep updating
        sigma_param_row = np.repeat(1e-2*np.eye(self.dm_state + self.dm_act + 1)[:,:,np.newaxis], self.nb_steps, axis=2) 
        sigma_param_col = np.repeat(1e-2*np.eye(self.dm_state)[:,:,np.newaxis], self.nb_steps, axis=2) 
        
        self.sigma_param_nom = np.transpose(np.array([np.kron(sigma_param_col[:,:,t], sigma_param_row[:,:,t]) for t in range(nb_steps)]), (1,2,0)) 

        A_flat = self.dyn.A.reshape(self.dm_state**2, self.nb_steps, order='F')
        B_flat = self.dyn.B.reshape(self.dm_state*self.dm_act, self.nb_steps, order='F')

        self.mu_param_nom = np.vstack((A_flat, B_flat, self.dyn.c))

    def sample(self, nb_episodes, stoch=True):
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
                x, _, _, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))
                data['xn'][..., t, n] = x

            c = self.env_cost(x, np.zeros((self.dm_act, )))
            data['c'][-1, n] = c

        return data

    def forward_pass(self, lgc, mu_param, sigma_param):
        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)
        xudist = Gaussian(self.dm_state + self.dm_act, self.nb_steps + 1)

        # TODO: rewrite cubature_forward_pass to take mu_param directly
        mat = self.mu_param.reshape((self.dm_state,-1,self.nb_steps),order='F')
        A = mat[:,:self.dm_state,:]
        B = mat[:,self.dm_state:self.dm_state+self.dm_act,:]
        c = mat[:,-1,:]

        xdist.mu, xdist.sigma,\
        udist.mu, udist.sigma,\
        xudist.mu, xudist.sigma = cubature_forward_pass(self.xdist.mu[..., 0], self.xdist.sigma[..., 0],
                                               A, B, c, sigma_param,
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
                                                           self.dyn.A, self.dyn.B, self.dyn.c, self.sigma_param,
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

    def parameter_backward_pass(self, alpha):
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        xvalue.V, xvalue.v, xvalue.v0,\
        mu_param, sigma_param, diverge = parameter_backward_pass(self.xudist.mu, self.xudist.sigma,
                                                                     self.cost.cx, self.cost.Cxx, self.cost.Cuu,
                                                                     self.cost.cu, self.cost.Cxu, self.cost.c0,
                                                                     self.dyn.A, self.dyn.B, self.dyn.c, self.sigma_param_nom,
                                                                     self.ctl.K, self.ctl.kff, self.ctl.sigma,
                                                                     alpha, self.dm_state, self.dm_act, self.nb_steps)
        return mu_param, sigma_param, xvalue, diverge

    def parameter_kldiv(self, mu_param, sigma_param):
        prec_param_nom = np.linalg.inv(self.sigma_param_nom.T).T

        delta_mu = (self.mu_param_nom - mu_param)
        quad_term = np.einsum('it,ijt,jt->t', delta_mu, prec_param_nom, delta_mu)
        trace_term = np.trace(np.einsum('ijt,jlt->ilt', prec_param_nom, sigma_param))
        log_det_term = np.log(np.linalg.det(self.sigma_param_nom.T)/np.linalg.det(sigma_param.T))
        
        return 0.5*(trace_term + quad_term + log_det_term - self.dm_param)


    def parameter_dual(self, alpha):
        # backward pass
        mu_param, sigma_param, xvalue, diverge = self.parameter_backward_pass(alpha)

        # forward pass
        xdist, udist, xudist = self.forward_pass(self.ctl, mu_param, sigma_param)

        # dual expectation
        dual = quad_expectation(xdist.mu[..., 0], xdist.sigma[..., 0],
                                xvalue.V[..., 0], xvalue.v[..., 0],
                                xvalue.v0[..., 0])
                            
        dual += alpha * (np.sum(self.parameter_kldiv(mu_param, sigma_param)[:-1]) - self.param_kl_bound)

        # gradient
        grad = np.sum(self.parameter_kldiv(mu_param, sigma_param)[:-1]) - self.param_kl_bound

        # TODO sign?
        return  np.array([dual]),  np.array([grad])

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

    def run(self, nb_episodes, nb_iter=10, verbose=False):
        _trace = []

        # run init controller
        self.data = self.sample(nb_episodes)

        # fit time-variant linear dynamics
        self.dyn.learn(self.data)


        # sigma_param_row = np.repeat(1e-3*np.eye(self.dm_state + self.dm_act + 1)[:,:,np.newaxis], self.nb_steps, axis=2) 
        # sigma_param_col = np.repeat(1e-8*np.eye(self.dm_state)[:,:,np.newaxis], self.nb_steps, axis=2) 
        
        # self.sigma_param_nom = np.transpose(np.array([np.kron(sigma_param_col[:,:,t], sigma_param_row[:,:,t]) for t in range(self.nb_steps)]), (1,2,0)) 

        self.sigma_param_nom = np.repeat(1e-4*np.eye(self.dm_state*(self.dm_state + self.dm_act + 1))[:,:,np.newaxis], self.nb_steps, axis=2) 

        A_flat = self.dyn.A.reshape(self.dm_state**2, self.nb_steps, order='F')
        B_flat = self.dyn.B.reshape(self.dm_state*self.dm_act, self.nb_steps, order='F')

        self.mu_param_nom = np.vstack((A_flat, B_flat, self.dyn.c))
        self.mu_param = self.mu_param_nom
        self.sigma_param = self.sigma_param_nom


        # current state distribution
        self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl, self.mu_param, self.sigma_param)

        # get quadratic cost around mean traj.
        self.cost.taylor_expansion(self.xdist.mu, self.udist.mu)

        # mean objective under current ctrl.
        self.last_return = np.mean(np.sum(self.data['c'], axis=0))
        _trace.append(self.last_return)

        for iter in range(nb_iter):
            # param_res = sc.optimize.minimize(self.parameter_dual, np.array([-1e5]),
            #                            method='SLSQP',
            #                            jac=True,
            #                            bounds=((-1e8, -1e-8), ),
            #                            options={'disp': True, 'maxiter': 10000,
            #                                     'ftol': 1e-6})
            # use scipy optimizer
            res = sc.optimize.minimize(self.dual, np.array([-1e8]),
                                       method='SLSQP',
                                       jac=True,
                                       callback=None,
                                       bounds=((-1e8, -1e-8), ),
                                       options={'disp': True, 'maxiter': 100,
                                                'ftol': 1e-6})
            self.alpha = res.x

         
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

            # update kl multiplier
            if self.kl_adaptive:
                _mult = _expected_imp / (2. * np.maximum(1e-4, _expected_imp - _actual_imp))
                _mult = np.maximum(0.1, np.minimum(5.0, _mult))
                self.kl_mult = np.maximum(np.minimum(_mult * self.kl_mult, self.kl_mult_max), self.kl_mult_min)

            # check kl constraint
            kl = self.kldiv(lgc, xdist)
            if (kl - self.kl_bound) < 0.25 * self.kl_bound:
                # update controller
                self.ctl = lgc

                # update value functions
                self.vfunc, self.qfunc = xvalue, xuvalue

                # run current controller
                self.data = self.sample(nb_episodes)

                # fit time-variant linear dynamics
                self.dyn.learn(self.data)

                # current state distribution
                self.xdist, self.udist, self.xudist = self.forward_pass(self.ctl, self.mu_param, self.sigma_param)

                # get quadratic cost around mean traj.
                self.cost.taylor_expansion(self.xdist.mu, self.udist.mu)

                # mean objective under last dists.
                _trace.append(_return)

                # update last return to current
                self.last_return = _return

                # update kl bound
                if self.kl_adaptive:
                    self.kl_bound = self.kl_base * self.kl_mult
            else:
                print("Something is wrong, KL not satisfied")
                self.alpha = np.array([-1e4])

            if verbose:
                print("iter: ", iter,
                      " req. kl: ", self.kl_bound,
                      " act. kl: ", kl,
                      " return: ", _return)

        return _trace
