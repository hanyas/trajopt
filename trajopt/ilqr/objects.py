import autograd.numpy as np
from autograd import jacobian, hessian

from pathos.multiprocessing import ProcessingPool as Pool


class QuadraticStateValue:
    def __init__(self, dm_state, nb_steps):
        self.dm_state = dm_state
        self.nb_steps = nb_steps

        self.V = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.v = np.zeros((self.dm_state, self.nb_steps, ))


class QuadraticStateActionValue:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.Qxx = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.Quu = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        self.Qux = np.zeros((self.dm_act, self.dm_state, self.nb_steps))

        self.qx = np.zeros((self.dm_state, self.nb_steps, ))
        self.qu = np.zeros((self.dm_act, self.nb_steps, ))


class QuadraticCost:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act

        self.nb_steps = nb_steps

        self.Cxx = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.cx = np.zeros((self.dm_state, self.nb_steps))

        self.Cuu = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        self.cu = np.zeros((self.dm_act, self.nb_steps))

        self.Cxu = np.zeros((self.dm_state, self.dm_act, self.nb_steps))

    @property
    def params(self):
        return self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu

    @params.setter
    def params(self, values):
        self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu = values


class AnalyticalQuadraticCost(QuadraticCost):
    def __init__(self, f, dm_state, dm_act, nb_steps):
        super(AnalyticalQuadraticCost, self).__init__(dm_state, dm_act, nb_steps)

        self.f = f

        self.dcdxx = hessian(self.f, 0)
        self.dcduu = hessian(self.f, 1)
        self.dcdxu = jacobian(jacobian(self.f, 0), 1)

        self.dcdx = jacobian(self.f, 0)
        self.dcdu = jacobian(self.f, 1)

    def evalf(self, x, u, a):
        return self.f(x, u, a)

    def taylor_expansion(self, x, u):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))

        pool = Pool(processes=-1)

        def _loop(t):
            _in = tuple([x[..., t], _u[..., t], _u[..., t - 1]])
            Cxx = self.dcdxx(*_in)
            Cuu = self.dcduu(*_in)
            Cxu = self.dcdxu(*_in)
            cx = self.dcdx(*_in)
            cu = self.dcdu(*_in)

            return Cxx, Cuu, Cxu, cx, cu

        res = pool.map(_loop, range(self.nb_steps))
        for t in range(self.nb_steps):
            self.Cxx[..., t], self.Cuu[..., t], self.Cxu[..., t],\
                self.cx[..., t], self.cu[..., t] = res[t]

        # for t in range(self.nb_steps):
        #     _in = tuple([x[..., t], _u[..., t], a[t]])
        #     self.Cxx[..., t] = self.dcdxx(*_in)
        #     self.Cuu[..., t] = self.dcduu(*_in)
        #     self.Cxu[..., t] = self.dcdxu(*_in)
        #     self.cx[..., t] = self.dcdx(*_in)
        #     self.cu[..., t] = self.dcdu(*_in)


class LinearDynamics:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.A = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.B = np.zeros((self.dm_state, self.dm_act, self.nb_steps))

    @property
    def params(self):
        return self.A, self.B

    @params.setter
    def params(self, values):
        self.A, self.B = values

    def sample(self, x, u):
        pass


class AnalyticalLinearDynamics(LinearDynamics):
    def __init__(self, f_dyn, dm_state, dm_act, nb_steps):
        super(AnalyticalLinearDynamics, self).__init__(dm_state, dm_act, nb_steps)

        self.f = f_dyn

        self.dfdx = jacobian(self.f, 0)
        self.dfdu = jacobian(self.f, 1)

    def evalf(self, x, u):
        return self.f(x, u)

    def taylor_expansion(self, x, u):
        for t in range(self.nb_steps):
            self.A[..., t] = self.dfdx(x[..., t], u[..., t])
            self.B[..., t] = self.dfdu(x[..., t], u[..., t])


class LinearControl:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.K = np.zeros((self.dm_act, self.dm_state, self.nb_steps))
        self.kff = np.zeros((self.dm_act, self.nb_steps))

    @property
    def params(self):
        return self.K, self.kff

    @params.setter
    def params(self, values):
        self.K, self.kff = values

    def action(self, x, alpha, xref, uref, t):
        dx = x[..., t] - xref[:, t]
        return uref[:, t] + alpha * self.kff[..., t] + self.K[..., t] @ dx
