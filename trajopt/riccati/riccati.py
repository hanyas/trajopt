import autograd.numpy as np

from trajopt.riccati.objects import AnalyticalLinearDynamics, AnalyticalQuadraticCost
from trajopt.riccati.objects import QuadraticStateValue
from trajopt.riccati.objects import LinearControl


class Riccati:

    def __init__(self, env, nb_steps,
                 init_state, activation=None):

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

        # reference trajectory
        self.xref = np.zeros((self.dm_state, self.nb_steps + 1))
        self.xref[..., 0] = self.env_init[0]

        self.uref = np.zeros((self.dm_act, self.nb_steps))

        self.vfunc = QuadraticStateValue(self.dm_state, self.nb_steps + 1)
        self.dyn = AnalyticalLinearDynamics(self.env_dyn, self.dm_state, self.dm_act, self.nb_steps)
        self.ctl = LinearControl(self.dm_state, self.dm_act, self.nb_steps)

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

    def forward_pass(self, ctl):
        state = np.zeros((self.dm_state, self.nb_steps + 1))
        action = np.zeros((self.dm_act, self.nb_steps))
        cost = np.zeros((self.nb_steps + 1, ))

        state[..., 0] = self.env.reset()
        for t in range(self.nb_steps):
            action[..., t] = ctl.action(state[..., t], t)
            cost[..., t] = self.env_cost(state[..., t], action[..., t], action[..., t - 1], self.weighting[t])
            state[..., t + 1], _, _, _ = self.env.step(action[..., t])

        cost[..., -1] = self.env_cost(state[..., -1], np.zeros((self.dm_act, )),
                                      np.zeros((self.dm_act, )), self.weighting[-1])
        return state, action, cost

    def backward_pass(self):
        lc = LinearControl(self.dm_state, self.dm_act, self.nb_steps)
        xvalue = QuadraticStateValue(self.dm_state, self.nb_steps + 1)

        xvalue.V[..., -1] = self.cost.Cxx[..., -1]
        xvalue.v[..., -1] = self.cost.cx[..., -1]

        for t in range(self.nb_steps - 2, -1, -1):
            Qxx = self.cost.Cxx[..., t] + self.dyn.A[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.A[..., t]
            Quu = self.cost.Cuu[..., t] + self.dyn.B[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.B[..., t]
            Qux = self.cost.Cxu[..., t].T + self.dyn.B[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.A[..., t]

            qx = self.cost.cx[..., t] + 2.0 * self.dyn.A[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.c[..., t] +\
                  self.dyn.A[..., t].T @ xvalue.v[..., t + 1]

            qu = self.cost.cu[..., t] + 2.0 * self.dyn.B[..., t].T @ xvalue.V[..., t + 1] @ self.dyn.c[..., t] +\
                  self.dyn.B[..., t].T @ xvalue.v[..., t + 1]

            Quu_inv = np.linalg.inv(Quu)

            lc.K[..., t] = - Quu_inv @ Qux
            lc.kff[..., t] = - 0.5 * Quu_inv @ qu

            xvalue.V[..., t] = Qxx + Qux.T * lc.K[..., t]
            xvalue.v[..., t] = qx + 2. * lc.kff[..., t].T @ Qux

        return lc, xvalue

    def plot(self, xref=None, uref=None):
        xref = self.xref if xref is None else xref
        uref = self.uref if uref is None else uref

        import matplotlib.pyplot as plt

        plt.figure()

        t = np.linspace(0, self.nb_steps, self.nb_steps + 1)
        for k in range(self.dm_state):
            plt.subplot(self.dm_state + self.dm_act, 1, k + 1)
            plt.plot(t, xref[k, :], '-b')

        t = np.linspace(0, self.nb_steps, self.nb_steps)
        for k in range(self.dm_act):
            plt.subplot(self.dm_state + self.dm_act, 1, self.dm_state + k + 1)
            plt.plot(t, uref[k, :], '-g')

        plt.show()

    def run(self):
        # get linear system dynamics around ref traj.
        self.dyn.taylor_expansion(self.xref, self.uref)

        # get quadratic cost around ref traj.
        self.cost.taylor_expansion(self.xref, self.uref, self.weighting)

        # backward pass to get ctrl.
        self.ctl, self.vfunc = self.backward_pass()

        # forward pass to get cost and traj.
        self.xref, self.uref, _cost = self.forward_pass(self.ctl)

        return np.sum(_cost)
