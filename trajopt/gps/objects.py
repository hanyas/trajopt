import autograd.numpy as np
from autograd import jacobian, hessian

from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart


class Gaussian:
    def __init__(self, nb_dim, nb_steps):
        self.nb_dim = nb_dim
        self.nb_steps = nb_steps

        self.mu = np.zeros((self.nb_dim, self.nb_steps))
        self.sigma = np.zeros((self.nb_dim, self.nb_dim, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = np.eye(self.nb_dim)

    @property
    def params(self):
        return self.mu, self.sigma

    @params.setter
    def params(self, values):
        self.mu, self.sigma = values

    def sample(self, x):
        pass


class QuadraticStateValue:
    def __init__(self, dm_state, nb_steps):
        self.dm_state = dm_state
        self.nb_steps = nb_steps

        self.V = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.v = np.zeros((self.dm_state, self.nb_steps, ))
        self.v0 = np.zeros((self.nb_steps, ))
        self.v0_softmax = np.zeros((self.nb_steps, ))


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

        self.q0 = np.zeros((self.nb_steps, ))
        self.q0_common = np.zeros((self.nb_steps, ))
        self.q0_softmax = np.zeros((self.nb_steps, ))


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

    def evaluate(self, x, u):
        _ret = 0.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))
        for t in range(self.nb_steps):
            _ret += x[..., t].T @ self.Cxx[..., t] @ x[..., t] +\
                    _u[..., t].T @ self.Cuu[..., t] @ _u[..., t] +\
                    x[..., t].T @ self.Cxu[..., t] @ _u[..., t] +\
                    self.cx[..., t].T @ x[..., t] +\
                    self.cu[..., t].T @ _u[..., t] + self.c0[..., t]
        return _ret


class AnalyticalQuadraticCost(QuadraticCost):
    def __init__(self, f, dm_state, dm_act, nb_steps):
        super(AnalyticalQuadraticCost, self).__init__(dm_state, dm_act, nb_steps)

        self.f = f

        self.dcdxx = hessian(self.f, 0)
        self.dcduu = hessian(self.f, 1)
        self.dcdxu = jacobian(jacobian(self.f, 0), 1)

        self.dcdx = jacobian(self.f, 0)
        self.dcdu = jacobian(self.f, 1)

    def evalf(self, x, u, u_last, a):
        return self.f(x, u, u_last, a)

    def taylor_expansion(self, x, u, a):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))

        for t in range(self.nb_steps):
            _in = tuple([x[..., t], _u[..., t], _u[..., t - 1], a[t]])
            self.Cxx[..., t] = 0.5 * self.dcdxx(*_in)
            self.Cuu[..., t] = 0.5 * self.dcduu(*_in)
            self.Cxu[..., t] = self.dcdxu(*_in)

            self.cx[..., t] = self.dcdx(*_in) - self.dcdxx(*_in) @ x[..., t] - self.dcdxu(*_in) @ _u[..., t]
            self.cu[..., t] = self.dcdu(*_in) - self.dcduu(*_in) @ _u[..., t] - x[..., t].T @ self.dcdxu(*_in)

            # residual of taylor expansion
            self.c0[..., t] = self.f(*_in)\
                              - x[..., t].T @ self.Cxx[..., t] @ x[..., t]\
                              - _u[..., t].T @ self.Cuu[..., t] @ _u[..., t]\
                              - x[..., t].T @ self.Cxu[..., t] @ _u[..., t]\
                              - self.cx[..., t].T @ x[..., t]\
                              - self.cu[..., t].T @ _u[..., t]


class LinearGaussianDynamics:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.A = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.B = np.zeros((self.dm_state, self.dm_act, self.nb_steps))
        self.c = np.zeros((self.dm_state, self.nb_steps))
        self.sigma = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = 1e-8 * np.eye(self.dm_state)

    @property
    def params(self):
        return self.A, self.B, self.c, self.sigma

    @params.setter
    def params(self, values):
        self.A, self.B, self.c, self.sigma = values

    def sample(self, x, u):
        pass


class AnalyticalLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, f_dyn, noise, dm_state, dm_act, nb_steps):
        super(AnalyticalLinearGaussianDynamics, self).__init__(dm_state, dm_act, nb_steps)

        self.f = f_dyn
        self.noise = noise

        self.dfdx = jacobian(self.f, 0)
        self.dfdu = jacobian(self.f, 1)

    def evalf(self, x, u):
        return self.f(x, u)

    def taylor_expansion(self, x, u):
        _A = self.dfdx(x, u)
        _B = self.dfdu(x, u)
        # residual of taylor expansion
        _c = self.evalf(x, u) - _A @ x - _B @ u
        _sigma = self.noise(x, u)
        return _A, _B, _c, _sigma

    def extended_kalman(self, init_state, lgc, ulim):
        lgd = LinearGaussianDynamics(self.dm_state, self.dm_act, self.nb_steps)

        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)

        # forward propagation of mean dynamics
        xdist.mu[..., 0], xdist.sigma[..., 0] = init_state
        for t in range(self.nb_steps):
            udist.mu[..., t] = np.clip(lgc.mean(xdist.mu[..., t], t), - ulim, ulim)
            xdist.mu[..., t + 1] = self.evalf(xdist.mu[..., t], udist.mu[..., t])

        for t in range(self.nb_steps):
            lgd.A[..., t], lgd.B[..., t], lgd.c[..., t], lgd.sigma[..., t] =\
                self.taylor_expansion(xdist.mu[..., t], udist.mu[..., t])

            # construct variace of next time step with extend Kalman filtering
            mu_x, sigma_x = xdist.mu[..., t], xdist.sigma[..., t]
            K, kff, _ctl_sigma = lgc.K[..., t], lgc.kff[..., t], lgc.sigma[..., t]

            # propagate variance of action dist.
            u_sigma = _ctl_sigma + K @ sigma_x @ K.T
            u_sigma = 0.5 * (u_sigma + u_sigma.T)
            udist.sigma[..., t] = u_sigma

            AB = np.hstack((lgd.A[..., t], lgd.B[..., t]))
            sigma_xu = np.vstack((np.hstack((sigma_x, sigma_x @ K.T)),
                                  np.hstack((K @ sigma_x, u_sigma))))

            sigma_xn = lgd.sigma[..., t] + AB @ sigma_xu @ AB.T
            sigma_xn = 0.5 * (sigma_xn + sigma_xn.T)
            xdist.sigma[..., t + 1] = sigma_xn

        return xdist, udist, lgd


# This part is still under construction
class LearnedLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, dm_state, dm_act, nb_steps, prior):
        super(LearnedLinearGaussianDynamics, self).__init__(dm_state, dm_act, nb_steps)

        hypparams = dict(M=np.zeros((self.dm_state, self.dm_state + self.dm_act + 1)),
                         K=prior['K'] * np.eye(self.dm_state + self.dm_act + 1),
                         psi=prior['psi'] * np.eye(self.dm_state),
                         nu=self.dm_state + prior['nu'])
        self.prior = MatrixNormalWishart(**hypparams)

    def learn(self, data, stepwise=True):
        if stepwise:

            for t in range(self.nb_steps):
                input = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T))
                target = data['xn'][:, t, :].T

                model = LinearGaussianWithMatrixNormalWishart(self.prior, affine=True)
                model = model.max_aposteriori(y=target, x=input)

                self.A[..., t] = model.likelihood.A[:, :self.dm_state]
                self.B[..., t] = model.likelihood.A[:, self.dm_state:self.dm_state + self.dm_act]
                self.c[..., t] = model.likelihood.A[:, -1]
                self.sigma[..., t] = model.likelihood.sigma
        else:
            raise NotImplementedError


class LinearGaussianControl:
    def __init__(self, dm_state, dm_act, nb_steps, init_ctl_sigma=1.):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.K = np.zeros((self.dm_act, self.dm_state, self.nb_steps))
        self.kff = np.zeros((self.dm_act, self.nb_steps))

        self.sigma = np.zeros((self.dm_act, self.dm_act, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = init_ctl_sigma * np.eye(self.dm_act)

    @property
    def params(self):
        return self.K, self.kff, self.sigma

    @params.setter
    def params(self, values):
        self.K, self.kff, self.sigma = values

    def mean(self, x, t):
        return np.einsum('kh,h->k', self.K[..., t], x) + self.kff[..., t]

    def sample(self, x, t, stoch=True):
        mu = self.mean(x, t)
        if stoch:
            return np.random.multivariate_normal(mean=mu, cov=self.sigma[..., t])
        else:
            return mu

    def forward(self, xdist, t):
        x_mu, x_sigma = xdist.mu[..., t], xdist.sigma[..., t]
        K, kff, ctl_sigma = self.K[..., t], self.kff[..., t], self.sigma[..., t]

        u_mu = K @ x_mu + kff
        u_sigma = ctl_sigma + K @ x_sigma @ K.T
        u_sigma = 0.5 * (u_sigma + u_sigma.T)

        return u_mu, u_sigma
