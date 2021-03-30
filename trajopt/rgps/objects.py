import autograd.numpy as np
from autograd import jacobian, hessian


from mimo.distributions import MatrixNormalWishart
from mimo.distributions import LinearGaussianWithMatrixNormalWishart
from mimo.distributions import MatrixNormalWithKnownPrecision
from mimo.distributions import LinearGaussianWithMatrixNormal
from mimo.distributions import LinearGaussianWithKnownPrecision

import scipy as sc
from scipy import stats


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


class MatrixNormalParameters:
    def __init__(self, dm_state, dm_act, nb_steps):
        self.dm_state = dm_state
        self.dm_act = dm_act
        self.nb_steps = nb_steps

        self.dm_param = self.dm_state * (self.dm_state + self.dm_act + 1)

        self.mu = np.zeros((self.dm_param, self.nb_steps))
        self.sigma = np.zeros((self.dm_param, self.dm_param, self.nb_steps))
        for t in range(self.nb_steps):
            self.sigma[..., t] = 1e0 * np.eye(self.dm_param)

    def sample(self, t):
        return np.random.multivariate_normal(self.mu[:, t], self.sigma[:, :, t])

    def matrices(self, t):
        A = np.reshape(self.mu[:self.dm_state * self.dm_state, t], (self.dm_state, self.dm_state), order='F')
        B = np.reshape(self.mu[self.dm_state * self.dm_state: self.dm_state * self.dm_state
                               + self.dm_state * self.dm_act, t], (self.dm_state, self.dm_act), order='F')
        c = np.reshape(self.mu[- self.dm_state:, t], (self.dm_state, 1), order='F')
        return A, B, c

    def entropy(self, t):
        return sc.stats.multivariate_normal(mean=self.mu[:, t], cov=self.sigma[..., t]).entropy()

    def plot(self, t, axs=None, color='b'):
        import matplotlib.pyplot as plt

        if axs is None:
            _, axs = plt.subplots(self.dm_param, figsize=(8, 12))
        for k, ax in enumerate(axs):
            mu, sigma = self.mu[k, t], self.sigma[k, k, t]
            plot_gaussian(mu, sigma, ax, color=color)

        plt.tight_layout()
        plt.show()

        return axs


class QuadraticStateValue:
    def __init__(self, dm_state, nb_steps):
        self.dm_state = dm_state
        self.nb_steps = nb_steps

        self.V = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        self.v = np.zeros((self.dm_state, self.nb_steps, ))
        self.v0 = np.zeros((self.nb_steps, ))


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

    def evaluate(self, x, u, stoch=True):
        ret = 0.
        _u = np.hstack((u.mu, np.zeros((self.dm_act, 1))))
        for t in range(self.nb_steps):
            ret += x.mu[..., t].T @ self.Cxx[..., t] @ x.mu[..., t] +\
                   _u[..., t].T @ self.Cuu[..., t] @ _u[..., t] +\
                   x.mu[..., t].T @ self.Cxu[..., t] @ _u[..., t] +\
                   self.cx[..., t].T @ x.mu[..., t] +\
                   self.cu[..., t].T @ _u[..., t] + self.c0[..., t]
            if stoch:
                # does not consider cross terms for now
                ret += np.trace(self.Cxx[..., t] @ x.sigma[..., t])
                if t < self.nb_steps - 1:
                    ret += np.trace(self.Cuu[..., t] @ u.sigma[..., t])
        return ret


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
            self.Cxu[..., t] = 0.5 * self.dcdxu(*_in)

            self.cx[..., t] = self.dcdx(*_in) - self.dcdxx(*_in) @ x[..., t] - 2. * self.dcdxu(*_in) @ _u[..., t]
            self.cu[..., t] = self.dcdu(*_in) - self.dcduu(*_in) @ _u[..., t] - 2. * x[..., t].T @ self.dcdxu(*_in)

            # residual of taylor expansion
            self.c0[..., t] = self.f(*_in)\
                              - x[..., t].T @ self.Cxx[..., t] @ x[..., t]\
                              - _u[..., t].T @ self.Cuu[..., t] @ _u[..., t]\
                              - 2. * x[..., t].T @ self.Cxu[..., t] @ _u[..., t]\
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
        A = self.dfdx(x, u)
        B = self.dfdu(x, u)
        # residual of taylor expansion
        c = self.evalf(x, u) - A @ x - B @ u
        sigma = self.noise(x, u)
        return A, B, c, sigma

    def extended_kalman(self, init_state, lgc, ulim):
        lgd = LinearGaussianDynamics(self.dm_state, self.dm_act, self.nb_steps)

        xdist = Gaussian(self.dm_state, self.nb_steps + 1)
        udist = Gaussian(self.dm_act, self.nb_steps)

        # forward propagation of mean dynamics
        xdist.mu[..., 0], xdist.sigma[..., 0] = init_state
        for t in range(self.nb_steps):
            udist.mu[..., t] = np.clip(lgc.K[..., t] @ xdist.mu[..., t] + lgc.kff[..., t], -ulim, ulim)
            xdist.mu[..., t + 1] = self.evalf(xdist.mu[..., t], udist.mu[..., t])

        for t in range(self.nb_steps):
            lgd.A[..., t], lgd.B[..., t], lgd.c[..., t], lgd.sigma[..., t] =\
                self.taylor_expansion(xdist.mu[..., t], udist.mu[..., t])

            # construct variace of next time step with extend Kalman filtering
            mu_x, sigma_x = xdist.mu[..., t], xdist.sigma[..., t]
            K, kff, ctl_sigma = lgc.K[..., t], lgc.kff[..., t], lgc.sigma[..., t]

            # propagate variance of action dist.
            u_sigma = ctl_sigma + K @ sigma_x @ K.T
            u_sigma = 0.5 * (u_sigma + u_sigma.T)
            udist.sigma[..., t] = u_sigma

            AB = np.hstack((lgd.A[..., t], lgd.B[..., t]))
            sigma_xu = np.vstack((np.hstack((sigma_x, sigma_x @ K.T)),
                                  np.hstack((K @ sigma_x, u_sigma))))

            sigma_xn = lgd.sigma[..., t] + AB @ sigma_xu @ AB.T
            sigma_xn = 0.5 * (sigma_xn + sigma_xn.T)
            xdist.sigma[..., t + 1] = sigma_xn

        return xdist, udist, lgd


class LearnedProbabilisticLinearDynamics(MatrixNormalParameters):
    def __init__(self, dm_state, dm_act, nb_steps, prior):
        super(LearnedProbabilisticLinearDynamics, self).__init__(dm_state, dm_act, nb_steps)

        hypparams = dict(M=np.zeros((self.dm_state, self.dm_state + self.dm_act + 1)),
                         K=prior['K'] * np.eye(self.dm_state + self.dm_act + 1),
                         psi=prior['psi'] * np.eye(self.dm_state),
                         nu=self.dm_state + prior['nu'])
        self.prior = MatrixNormalWishart(**hypparams)

    def learn(self, data):
        noise = np.zeros((self.dm_state, self.dm_state, self.nb_steps))
        for t in range(self.nb_steps):
            input = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T))
            target = data['xn'][:, t, :].T

            model = LinearGaussianWithMatrixNormalWishart(self.prior, affine=True)
            model = model.meanfield_update(y=target, x=input)

            self.mu[..., t] = np.reshape(model.posterior.matnorm.M, self.mu[..., t].shape, order='F')
            self.sigma[..., t] = np.linalg.inv(np.kron(model.posterior.matnorm.K, model.posterior.wishart.mode()))
            noise[..., t] = np.linalg.inv(model.posterior.wishart.mode())

        return noise


class LearnedProbabilisticLinearDynamicsWithKnownNoise(MatrixNormalParameters):
    def __init__(self, dm_state, dm_act, nb_steps, noise, prior):
        super(LearnedProbabilisticLinearDynamicsWithKnownNoise, self).__init__(dm_state, dm_act, nb_steps)

        hypparams = dict(M=np.zeros((self.dm_state, self.dm_state + self.dm_act + 1)),
                         K=prior['K'] * np.eye(self.dm_state + self.dm_act + 1),
                         V=np.linalg.inv(noise))
        self.prior = MatrixNormalWithKnownPrecision(**hypparams)
        self.noise = noise  # assumed stationary over all time steps

    def learn(self, data):
        for t in range(self.nb_steps):
            input = np.hstack((data['x'][:, t, :].T, data['u'][:, t, :].T))
            target = data['xn'][:, t, :].T

            likelihood = LinearGaussianWithKnownPrecision(lmbda=np.linalg.inv(self.noise), affine=True)
            model = LinearGaussianWithMatrixNormal(self.prior, likelihood=likelihood, affine=True)
            model = model.meanfield_update(y=target, x=input)

            self.mu[..., t] = np.reshape(model.posterior.M, self.mu[..., t].shape, order='F')
            self.sigma[..., t] = model.posterior.sigma


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


def pass_alpha_as_vector(f):
    def wrapper(self, alpha, *args):
        assert alpha is not None

        if alpha.shape[0] == 1:
            alpha = alpha * np.ones((self.nb_steps, ))

        return f(self, alpha, *args)
    return wrapper


def plot_gaussian(mu, sigma, ax, color='b', points=250):
    from scipy.stats import norm

    min = mu - 5. * sigma
    max = mu + 5. * sigma
    x = np.linspace(min, max, points)
    p = norm.pdf(x, mu, sigma)

    ax.plot(x, p, color=color, linewidth=2.)
    ax.fill_between(x, 0, p, color=color, alpha=0.1)
