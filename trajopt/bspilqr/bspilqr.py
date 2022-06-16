import autograd.numpy as np

from trajopt.bspilqr.objects import Gaussian
from trajopt.bspilqr.objects import AnalyticalLinearBeliefDynamics, AnalyticalQuadraticCost
from trajopt.bspilqr.objects import QuadraticBeliefValue
from trajopt.bspilqr.objects import LinearControl

from trajopt.bspilqr.core import backward_pass


class BSPiLQR:

    def __init__(self, env, nb_steps, init_belief,
                 alphas=np.power(10., np.linspace(0, -3, 11)),
                 lmbda=1., dlmbda=1., min_lmbda=1e-6,
                 max_lmbda=1e6, mult_lmbda=1.6, tolfun=1e-8,
                 tolgrad=1e-6, min_imp=0., reg=1):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_obs = self.env.unwrapped.observe
        self.env_cost = self.env.unwrapped.cost
        self.env_init = init_belief

        self.env_dyn_noise = self.env.unwrapped.dyn_noise
        self.env_obs_noise = self.env.unwrapped.obs_noise

        self.ulim = self.env.action_space.high

        self.belief_dim = self.env.unwrapped.state_space.shape[0]
        self.obs_dim = self.env.unwrapped.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.nb_steps = nb_steps

        # backtracking
        self.alphas = alphas
        self.alpha = None

        self.lmbda = lmbda
        self.dlmbda = dlmbda
        self.min_lmbda = min_lmbda
        self.max_lmbda = max_lmbda
        self.mult_lmbda = mult_lmbda

        # regularization type
        self.reg = reg

        # minimum relative improvement
        self.min_imp = min_imp

        # stopping criterion
        self.tolfun = tolfun
        self.tolgrad = tolgrad

        # reference belief trajectory
        self.bref = Gaussian(self.belief_dim, self.nb_steps + 1)
        self.bref.mu[..., 0], self.bref.sigma[..., 0] = self.env_init

        self.uref = np.zeros((self.act_dim, self.nb_steps))

        self.vfunc = QuadraticBeliefValue(self.belief_dim, self.nb_steps + 1)

        self.dyn = AnalyticalLinearBeliefDynamics(self.env_dyn, self.env_obs,
                                                  self.env_dyn_noise, self.env_obs_noise,
                                                  self.belief_dim, self.obs_dim, self.act_dim, self.nb_steps)

        self.ctl = LinearControl(self.belief_dim, self.act_dim, self.nb_steps)
        self.ctl.kff = 1e-8 * np.random.randn(self.act_dim, self.nb_steps)

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.belief_dim, self.act_dim, self.nb_steps + 1)

        self.last_return = - np.inf

    def forward_pass(self, ctl, alpha):
        belief = Gaussian(self.belief_dim, self.nb_steps + 1)
        action = np.zeros((self.act_dim, self.nb_steps))
        cost = np.zeros((self.nb_steps + 1, ))

        belief.mu[..., 0], belief.sigma[..., 0] = self.env_init
        for t in range(self.nb_steps):
            action[..., t] = ctl.action(belief, alpha, self.bref, self.uref, t)
            cost[..., t] = self.env_cost(belief.mu[..., t], belief.sigma[..., t], action[..., t])
            belief.mu[..., t + 1], belief.sigma[..., t + 1] = self.dyn.forward(belief, action, t)

        cost[..., -1] = self.env_cost(belief.mu[..., -1], belief.sigma[..., -1], np.zeros((self.act_dim, )))
        return belief, action, cost

    def backward_pass(self):
        lc = LinearControl(self.belief_dim, self.act_dim, self.nb_steps)
        bvalue = QuadraticBeliefValue(self.belief_dim, self.nb_steps + 1)

        bvalue.S, bvalue.s, bvalue.tau,\
        dS, lc.K, lc.kff, diverge = backward_pass(self.cost.Q, self.cost.q,
                                                  self.cost.R, self.cost.r,
                                                  self.cost.P, self.cost.p,
                                                  self.dyn.F, self.dyn.G,
                                                  self.dyn.T, self.dyn.U,
                                                  self.dyn.V, self.dyn.X,
                                                  self.dyn.Y, self.dyn.Z,
                                                  self.lmbda, self.reg,
                                                  self.belief_dim, self.act_dim, self.nb_steps)
        return lc, bvalue, dS, diverge

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.belief_dim):
            plt.subplot(self.belief_dim + self.act_dim, 1, k + 1)
            plt.plot(t, self.bref.mu[k, :], '-b')
            lb = self.bref.mu[k, :] - 2. * np.sqrt(self.bref.sigma[k, k, :])
            ub = self.bref.mu[k, :] + 2. * np.sqrt(self.bref.sigma[k, k, :])
            plt.fill_between(t, lb, ub, color='blue', alpha=0.1)

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.act_dim):
            plt.subplot(self.belief_dim + self.act_dim, 1, self.belief_dim + k + 1)
            plt.plot(t, self.uref[k, :], '-g')

        plt.show()

    def run(self, nb_iter=25, verbose=False):
        _trace = []
        # init trajectory
        for alpha in self.alphas:
            _belief, _action, _cost = self.forward_pass(self.ctl, alpha)
            if np.all(_belief.mu < 1e8):
                self.bref = _belief
                self.uref = _action
                self.last_return = np.sum(_cost)
                break
            else:
                print("Initial trajectory diverges")

        _trace.append(self.last_return)

        for iter in range(nb_iter):
            # get linear system dynamics around ref traj.
            self.dyn.taylor_expansion(self.bref, self.uref)

            # get quadratic cost around ref traj.
            self.cost.taylor_expansion(self.bref, self.uref)

            bvalue = None
            lc, dvalue = None, None
            # execute a backward pass
            backpass_done = False
            while not backpass_done:
                lc, bvalue, dvalue, diverge = self.backward_pass()
                if np.any(diverge):
                    # increase lmbda
                    self.dlmbda = np.maximum(self.dlmbda * self.mult_lmbda, self.mult_lmbda)
                    self.lmbda = np.maximum(self.lmbda * self.dlmbda, self.min_lmbda)
                    if self.lmbda > self.max_lmbda:
                        break
                    else:
                        continue
                else:
                    backpass_done = True

            # terminate if gradient too small
            _g_norm = np.mean(np.max(np.abs(lc.kff) / (np.abs(self.uref) + 1.), axis=1))
            if _g_norm < self.tolgrad and self.lmbda < 1e-5:
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)
                break

            _belief, _action = None, None
            _return, _dreturn = None, None
            # execute a forward pass
            fwdpass_done = False
            if backpass_done:
                for alpha in self.alphas:
                    # apply on actual system
                    _belief, _action, _cost = self.forward_pass(ctl=lc, alpha=alpha)

                    # summed mean return
                    _return = np.sum(_cost)

                    # check return improvement
                    _dreturn = self.last_return - _return
                    _expected = - 1. * alpha * (dvalue[0] + alpha * dvalue[1])
                    _imp = _dreturn / _expected
                    if _imp > self.min_imp:
                        fwdpass_done = True
                        break

            # accept or reject
            if fwdpass_done:
                # decrease lmbda
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)

                self.bref = _belief
                self.uref = _action
                self.last_return = _return

                self.vfunc = bvalue

                self.ctl = lc

                _trace.append(self.last_return)

                # terminate if reached objective tolerance
                if _dreturn < self.tolfun:
                    break
            else:
                # increase lmbda
                self.dlmbda = np.maximum(self.dlmbda * self.mult_lmbda, self.mult_lmbda)
                self.lmbda = np.maximum(self.lmbda * self.dlmbda, self.min_lmbda)
                if self.lmbda > self.max_lmbda:
                    break
                else:
                    continue

            if verbose:
                print("iter: ", iter,
                      " return: ", _return)

        return _trace
