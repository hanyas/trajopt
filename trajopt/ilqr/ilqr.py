import autograd.numpy as np

from trajopt.ilqr.objects import AnalyticalLinearDynamics, AnalyticalQuadraticCost
from trajopt.ilqr.objects import QuadraticStateValue, QuadraticStateActionValue
from trajopt.ilqr.objects import LinearControl

from trajopt.ilqr.core import backward_pass


class iLQR:

    def __init__(self, env, nb_steps,
                 init_state, init_action=None,
                 alphas=np.power(10., np.linspace(0, -3, 11)),
                 lmbda=1., dlmbda=1.,
                 min_lmbda=1e-6, max_lmbda=1e6, mult_lmbda=1.6,
                 tolfun=1e-6, tolgrad=1e-4, min_imp=0., reg=1):

        self.env = env

        # expose necessary functions
        self.env_dyn = self.env.unwrapped.dynamics
        self.env_cost = self.env.unwrapped.cost
        self.env_init = init_state

        self.ulim = self.env.action_space.high

        self.dm_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]
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

        # reference trajectory
        self.xref = np.zeros((self.dm_state, self.nb_steps + 1))
        self.xref[..., 0] = self.env_init

        self.uref = np.zeros((self.dm_act, self.nb_steps))

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.qfunc = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        self.dyn = AnalyticalLinearDynamics(self.env_dyn, self.dm_state, self.dm_act, self.nb_steps)

        self.ctl = LinearControl(self.dm_state, self.dm_act, self.nb_steps)
        if init_action is None:
            self.ctl.kff = 1e-8 * np.random.randn(self.dm_act, self.nb_steps)
        else:
            assert init_action.shape[1] == self.nb_steps
            self.ctl.kff = init_action

        self.cost = AnalyticalQuadraticCost(self.env_cost, self.dm_state, self.dm_act, self.nb_steps + 1)

        self.last_return = - np.inf

    def forward_pass(self, ctl, alpha):
        state = np.zeros((self.dm_state, self.nb_steps + 1))
        action = np.zeros((self.dm_act, self.nb_steps))
        cost = np.zeros((self.nb_steps + 1, ))

        state[..., 0] = self.env_init
        for t in range(self.nb_steps):
            _act = ctl.action(state, alpha, self.xref, self.uref, t)
            action[..., t] = np.clip(_act, -self.ulim, self.ulim)
            cost[..., t] = self.env_cost(state[..., t], action[..., t], action[..., t - 1])
            state[..., t + 1] = self.env_dyn(state[..., t], action[..., t])

        cost[..., -1] = self.env_cost(state[..., -1], np.zeros((self.dm_act, )), np.zeros((self.dm_act, )))
        return state, action, cost

    def backward_pass(self):
        lc = LinearControl(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        xuvalue = QuadraticStateActionValue(self.dm_state, self.dm_act, self.nb_steps)

        xuvalue.Qxx, xuvalue.Qux, xuvalue.Quu,\
        xuvalue.qx, xuvalue.qu,\
        xvalue.V, xvalue.v, dV,\
        lc.K, lc.kff, diverge = backward_pass(self.cost.Cxx, self.cost.cx, self.cost.Cuu,
                                              self.cost.cu, self.cost.Cxu,
                                              self.dyn.A, self.dyn.B,
                                              self.lmbda, self.reg,
                                              self.dm_state, self.dm_act, self.nb_steps)
        return lc, xvalue, xuvalue, dV, diverge

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, k + 1)
            plt.plot(t, self.xref[k, :], '-b')

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + k + 1)
            plt.plot(t, self.uref[k, :], '-g')

        plt.show()

    def run(self, nb_iter=25, verbose=False):
        _trace = []
        # init trajectory
        for alpha in self.alphas:
            _state, _action, _cost = self.forward_pass(self.ctl, alpha)
            if np.all(_state < 1e8):
                self.xref = _state
                self.uref = _action
                self.last_return = np.sum(_cost)
                break
            else:
                print("Initial trajectory diverges")

        _trace.append(self.last_return)

        for iter in range(nb_iter):
            # get linear system dynamics around ref traj.
            self.dyn.taylor_expansion(self.xref, self.uref)

            # get quadratic cost around ref traj.
            self.cost.taylor_expansion(self.xref, self.uref)

            xvalue, xuvalue = None, None
            lc, dvalue = None, None
            # execute a backward pass
            backpass_done = False
            while not backpass_done:
                lc, xvalue, xuvalue, dvalue, diverge = self.backward_pass()
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

            _state, _action = None, None
            _return, _dreturn = None, None
            # execute a forward pass
            fwdpass_done = False
            if backpass_done:
                for alpha in self.alphas:
                    self.alpha = alpha

                    # apply on actual system
                    _state, _action, _cost = self.forward_pass(ctl=lc, alpha=self.alpha)

                    # summed mean return
                    _return = np.sum(_cost)

                    # check return improvement
                    _dreturn = self.last_return - _return
                    _expected = - 1. * alpha * (dvalue[0] + alpha * dvalue[1])
                    _imp = _dreturn / _expected
                    if _imp >= self.min_imp:
                        fwdpass_done = True
                        break

            # accept or reject
            if fwdpass_done:
                # decrease lmbda
                self.dlmbda = np.minimum(self.dlmbda / self.mult_lmbda, 1. / self.mult_lmbda)
                self.lmbda = self.lmbda * self.dlmbda * (self.lmbda > self.min_lmbda)

                self.xref = _state
                self.uref = _action
                self.last_return = _return

                self.vfunc = xvalue
                self.qfunc = xuvalue

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
