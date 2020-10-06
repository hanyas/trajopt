import autograd.numpy as np
from autograd import jacobian, hessian

from pathos.multiprocessing import ProcessingPool as Pool


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

    def evalf(self, x, u, a):
        return self.f(x, u, a)

    def taylor_expansion(self, x, u):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.dm_act, 1))))

        pool = Pool(processes=-1)

        def _loop(t):
            _in = tuple([x[..., t], _u[..., t], _u[..., t - 1]])
            _dcdxx = self.dcdxx(*_in)
            _dcduu = self.dcduu(*_in)
            _dcdxu = self.dcdxu(*_in)

            Cxx = 0.5 * _dcdxx
            Cuu = 0.5 * _dcduu
            Cxu = _dcdxu

            cx = self.dcdx(*_in) - _dcdxx @ x[..., t] - _dcdxu @ _u[..., t]
            cu = self.dcdu(*_in) - _dcduu @ _u[..., t] - x[..., t].T @ _dcdxu

            # residual of taylor expansion
            c0 = self.f(*_in) - x[..., t].T @ Cxx @ x[..., t]\
                              - _u[..., t].T @ Cuu @ _u[..., t]\
                              - x[..., t].T @ Cxu @ _u[..., t]\
                              - cx.T @ x[..., t]\
                              - cu.T @ _u[..., t]

            return Cxx, Cuu, Cxu, cx, cu, c0

        res = pool.map(_loop, range(self.nb_steps))
        for t in range(self.nb_steps):
            self.Cxx[..., t], self.Cuu[..., t], self.Cxu[..., t],\
                self.cx[..., t], self.cu[..., t], self.c0[..., t] = res[t]

        # for t in range(self.nb_steps):
        #     _in = tuple([x[..., t], _u[..., t], a[t]])
        #     self.Cxx[..., t] = 0.5 * self.dcdxx(*_in)
        #     self.Cuu[..., t] = 0.5 * self.dcduu(*_in)
        #     self.Cxu[..., t] = self.dcdxu(*_in)
        #
        #     self.cx[..., t] = self.dcdx(*_in) - self.dcdxx(*_in) @ x[..., t] - self.dcdxu(*_in) @ _u[..., t]
        #     self.cu[..., t] = self.dcdu(*_in) - self.dcduu(*_in) @ _u[..., t] - x[..., t].T @ self.dcdxu(*_in)
        #
        #     # residual of taylor expansion
        #     self.c0[..., t] = self.f(*_in)\
        #                       - x[..., t].T @ self.Cxx[..., t] @ x[..., t]\
        #                       - _u[..., t].T @ self.Cuu[..., t] @ _u[..., t]\
        #                       - x[..., t].T @ self.Cxu[..., t] @ _u[..., t]\
        #                       - self.cx[..., t].T @ x[..., t]\
        #                       - self.cu[..., t].T @ _u[..., t]


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

        pool = Pool(processes=-1)

        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)

        # forward propagation of mean dynamics
        xdist.mu[..., 0], xdist.sigma[..., 0] = init_state
        for t in range(self.nb_steps):
            udist.mu[..., t] = np.clip(lgc.mean(xdist.mu[..., t], t), - ulim, ulim)
            xdist.mu[..., t + 1] = self.evalf(xdist.mu[..., t], udist.mu[..., t])

        # parallel autograd linearization around mean traj.
        def _loop(t):
            return self.taylor_expansion(xdist.mu[..., t], udist.mu[..., t])

        res = pool.map(_loop, range(self.nb_steps))
        for t in range(self.nb_steps):
            lgd.A[..., t], lgd.B[..., t], lgd.c[..., t], lgd.sigma[..., t] = res[t]

            # construct variace of next time step with extend Kalman filtering
            _mu_x, _sigma_x = xdist.mu[..., t], xdist.sigma[..., t]
            _K, _kff, _ctl_sigma = lgc.K[..., t], lgc.kff[..., t], lgc.sigma[..., t]

            # propagate variance of action dist.
            _u_sigma = _ctl_sigma + _K @ _sigma_x @ _K.T
            _u_sigma = 0.5 * (_u_sigma + _u_sigma.T)
            udist.sigma[..., t] = _u_sigma

            _AB = np.hstack((lgd.A[..., t], lgd.B[..., t]))
            _sigma_xu = np.vstack((np.hstack((_sigma_x, _sigma_x @ _K.T)),
                                   np.hstack((_K @ _sigma_x, _u_sigma))))

            _sigma_xn = lgd.sigma[..., t] + _AB @ _sigma_xu @ _AB.T
            _sigma_xn = 0.5 * (_sigma_xn + _sigma_xn.T)
            xdist.sigma[..., t + 1] = _sigma_xn

        return xdist, udist, lgd


# This part is still under construction
class LearnedLinearGaussianDynamics(LinearGaussianDynamics):
    def __init__(self, dm_state, dm_act, nb_steps):
        super(LearnedLinearGaussianDynamics, self).__init__(dm_state, dm_act, nb_steps)

    def learn(self, data, pointwise=False):
        if pointwise:
            from mimo import distributions
            _hypparams = dict(M=np.zeros((self.dm_state, self.dm_state + self.dm_act + 1)),
                              V=1e6 * np.eye(self.dm_state + self.dm_act + 1),
                              affine=True,
                              psi=np.eye(self.dm_state), nu=self.dm_state + 2)
            _prior = distributions.MatrixNormalInverseWishart(**_hypparams)

            for t in range(self.nb_steps):
                _data = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T, data['xn'][:, t, :].T))

                _model = distributions.BayesianLinearGaussian(_prior)
                _model = _model.MAP(_data)

                self.A[..., t] = _model.A[:, :self.dm_state]
                self.B[..., t] = _model.A[:, self.dm_state:self.dm_state + self.dm_act]
                self.c[..., t] = _model.A[:, -1]
                self.sigma[..., t] = _model.sigma
        else:
            _obs = [data['x'][..., n].T for n in range(data['x'].shape[-1])]
            _input = [data['u'][..., n].T for n in range(data['u'].shape[-1])]

            from sds.rarhmm_ls import rARHMM
            rarhmm = rARHMM(nb_states=5, dim_obs=self.dm_state, dim_act=self.dm_act)
            rarhmm.initialize(_obs, _input)
            rarhmm.em(_obs, _input, nb_iter=50, prec=1e-12, verbose=False)

            _mean_obs = np.mean(data['x'], axis=-1).T
            _mean_input = np.mean(data['u'], axis=-1).T
            _, _mean_z = rarhmm.viterbi([_mean_obs], [_mean_input])

            for t in range(self.nb_steps):
                self.A[..., t] = rarhmm.observations.A[_mean_z[0][t], ...]
                self.B[..., t] = rarhmm.observations.B[_mean_z[0][t], ...]
                self.c[..., t] = rarhmm.observations.c[_mean_z[0][t], ...]


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
        _x_mu, _x_sigma = xdist.mu[..., t], xdist.sigma[..., t]
        _K, _kff, _ctl_sigma = self.K[..., t], self.kff[..., t], self.sigma[..., t]

        _u_mu = _K @ _x_mu + _kff
        _u_sigma = _ctl_sigma + _K @ _x_sigma @ _K.T
        _u_sigma = 0.5 * (_u_sigma + _u_sigma.T)

        return _u_mu, _u_sigma
