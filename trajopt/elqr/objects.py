import autograd.numpy as np
from autograd import jacobian, hessian
from copy import deepcopy


class QuadraticStateValue:
    def __init__(self, dm_state, nb_steps):
        self.dm_state = dm_state
        self.nb_steps = nb_steps

        self.V = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.v = np.zeros((self.dm_state, self.nb_steps, ))
        self.v0 = np.zeros((self.nb_steps, ))


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
        self.c0 = np.zeros((self.nb_steps, ))

    @property
    def params(self):
        return self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu, self.c0

    @params.setter
    def params(self, values):
        self.Cxx, self.cx, self.Cuu, self.cu, self.Cxu, self.c0 = values


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

    def taylor_expansion(self, x, u, a):
        _in = tuple([x, u, a])
        _Cxx = 0.5 * self.dcdxx(*_in)
        _Cuu = 0.5 * self.dcduu(*_in)
        _Cxu = self.dcdxu(*_in)

        _cx = self.dcdx(*_in) - self.dcdxx(*_in) @ x - self.dcdxu(*_in) @ u
        _cu = self.dcdu(*_in) - self.dcduu(*_in) @ u - x.T @ self.dcdxu(*_in)

        # residual of taylor expansion
        _c0 = self.f(*_in) - x.T @ _Cxx @ x -\
              u.T @ _Cuu @ u - x.T @ _Cxu @ u -\
              _cx.T @ x - _cu.T @ u

        return _Cxx, _Cuu, _Cxu, _cx, _cu, _c0


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
        _A = self.dfdx(x, u)
        _B = self.dfdu(x, u)
        # residual of taylor expansion
        _c = self.evalf(x, u) - _A @ x - _B @ u

        return _A, _B, _c


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
