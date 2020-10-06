import autograd.numpy as np
from autograd import jacobian, hessian


class QuadraticStateValue:
    def __init__(self, dm_state, nb_steps):
        self.dm_state = dm_state
        self.nb_steps = nb_steps

        self.V = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.v = np.zeros((self.dm_state, self.nb_steps, ))


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

    def evalf(self, x, u):
        return self.f(x, u)

    def taylor_expansion(self, x, u):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))
        for t in range(self.nb_steps):
            _in = tuple([x[..., t], _u[..., t]])
            self.Cxx[..., t] = self.dcdxx(*_in)
            self.Cuu[..., t] = self.dcduu(*_in)
            self.Cxu[..., t] = self.dcdxu(*_in)
            self.cx[..., t] = self.dcdx(*_in)
            self.cu[..., t] = self.dcdu(*_in)


class LinearDynamics:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.A = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.B = np.zeros((self.dm_state, self.dm_act, self.nb_steps))
        self.c = np.zeros((self.dm_state, self.nb_steps))

    @property
    def params(self):
        return self.A, self.B, self.c

    @params.setter
    def params(self, values):
        self.A, self.B, self.c = values

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
            # residual of taylor expansion
            self.c[..., t] = self.evalf(x[..., t], u[..., t]) -\
                             self.A[..., t] @ x[..., t] - self.B[..., t] @ u[..., t]


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

    def action(self, x, t):
        return self.kff[..., t] + self.K[..., t] @ x
