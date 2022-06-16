import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.misc import flatten


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

class EKF:

    def __init__(self, env):

        self.f = env.unwrapped.dynamics
        self.h = env.unwrapped.observe

        self.dfdx = jacobian(self.f, 0)
        self.dhdx = jacobian(self.h, 0)

        self.dyn_noise = env.unwrapped.dyn_noise
        self.obs_noise = env.unwrapped.obs_noise

        self.belief_dim = env.belief_dim
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

    def evalf(self, mu_b, u):
        return self.f(mu_b, u)

    def evalh(self, mu_b):
        return self.h(mu_b)

    def predict(self, mu_b, sigma_b, u):
        A = self.dfdx(mu_b, u)
        sigma_dyn = self.dyn_noise(mu_b, u)

        sigma_b_pr = A @ sigma_b @ A.T + sigma_dyn
        sigma_b_pr = 0.5 * (sigma_b_pr + sigma_b_pr.T)

        mu_b_pr = self.evalf(mu_b, u)

        return mu_b_pr, sigma_b_pr

    def innovate(self, mu_b, sigma_b, z):
        H = self.dhdx(mu_b)
        sigma_obs = self.obs_noise(mu_b)

        K = sigma_b @ H.T @ np.linalg.inv(H @ sigma_b @ H.T + sigma_obs)

        mu_b_in = mu_b + K @ (z - self.evalh(mu_b))
        sigma_b_in = sigma_b - K @ H @ sigma_b
        sigma_b_in = 0.5 * (sigma_b_in + sigma_b_in.T)

        return mu_b_in, sigma_b_in

    def inference(self, mu_b, sigma_b, u, z):
        mu_b, sigma_b = self.predict(mu_b, sigma_b, u)
        mu_b, sigma_b = self.innovate(mu_b, sigma_b, z)
        return mu_b, sigma_b


class QuadraticBeliefValue:
    def __init__(self, belief_dim, nb_steps):
        self.belief_dim = belief_dim
        self.nb_steps = nb_steps

        self.S = np.zeros((self.belief_dim, self.belief_dim, self.nb_steps))
        self.s = np.zeros((self.belief_dim, self.nb_steps, ))
        self.tau = np.zeros((self.belief_dim, self.nb_steps, ))


class QuadraticCost:
    def __init__(self, belief_dim, act_dim, nb_steps):
        self.belief_dim = belief_dim
        self.act_dim = act_dim

        self.nb_steps = nb_steps

        self.Q = np.zeros((self.belief_dim, self.belief_dim, self.nb_steps))
        self.q = np.zeros((self.belief_dim, self.nb_steps))

        self.R = np.zeros((self.act_dim, self.act_dim, self.nb_steps))
        self.r = np.zeros((self.act_dim, self.nb_steps))

        self.P = np.zeros((self.belief_dim, self.act_dim, self.nb_steps))
        self.p = np.zeros((self.belief_dim * self.belief_dim, self.nb_steps))

    @property
    def params(self):
        return self.Q, self.q, self.R, self.r, self.P, self.p

    @params.setter
    def params(self, values):
        self.Q, self.q, self.R, self.r, self.P, self.p = values


class AnalyticalQuadraticCost(QuadraticCost):
    def __init__(self, f_cost, belief_dim, act_dim, nb_steps):
        super(AnalyticalQuadraticCost, self).__init__(belief_dim, act_dim, nb_steps)

        self.f = f_cost

        self.fQ = hessian(self.f, 0)
        self.fq = jacobian(self.f, 0)

        self.fR = hessian(self.f, 2)
        self.fr = jacobian(self.f, 2)

        self.fP = jacobian(jacobian(self.f, 0), 2)
        self.fp = jacobian(self.f, 1)

    def evalf(self, mu_b, sigma_b, u):
        return self.f(mu_b, sigma_b, u)

    def taylor_expansion(self, b, u):
        # padd last time step of action traj.
        _u = np.hstack((u, np.zeros((self.act_dim, 1))))

        for t in range(self.nb_steps):
            _in = tuple([b.mu[..., t], b.sigma[..., t], _u[..., t]])

            self.Q[..., t] = self.fQ(*_in)
            self.q[..., t] = self.fq(*_in)

            self.R[..., t] = self.fR(*_in)
            self.r[..., t] = self.fr(*_in)

            self.P[..., t] = self.fP(*_in)
            self.p[..., t] = np.reshape(self.fp(*_in),
                                        (self.belief_dim * self.belief_dim), order='F')


class LinearBeliefDynamics:
    def __init__(self, belief_dim, obs_dim, act_dim, nb_steps):
        self.belief_dim = belief_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.nb_steps = nb_steps

        # Linearization of dynamics
        self.A = np.zeros((self.belief_dim, self.belief_dim, self.nb_steps))
        self.H = np.zeros((self.obs_dim, self.obs_dim, self.nb_steps))

        # EKF matrices
        self.K = np.zeros((self.belief_dim, self.obs_dim, self.nb_steps))
        self.D = np.zeros((self.belief_dim, self.obs_dim, self.nb_steps))

        # Linearization of belief dynamics
        self.F = np.zeros((self.belief_dim, self.belief_dim, self.nb_steps))
        self.G = np.zeros((self.belief_dim, self.act_dim, self.nb_steps))

        self.T = np.zeros((self.belief_dim * self.belief_dim, self.belief_dim, self.nb_steps))
        self.U = np.zeros((self.belief_dim * self.belief_dim, self.belief_dim * self.belief_dim, self.nb_steps))
        self.V = np.zeros((self.belief_dim * self.belief_dim, self.act_dim, self.nb_steps))

        self.X = np.zeros((self.belief_dim * self.belief_dim, self.belief_dim, self.nb_steps))
        self.Y = np.zeros((self.belief_dim * self.belief_dim, self.belief_dim * self.belief_dim, self.nb_steps))
        self.Z = np.zeros((self.belief_dim * self.belief_dim, self.act_dim, self.nb_steps))

        self.sigma_x = np.zeros((self.belief_dim, self.belief_dim, self.nb_steps))
        self.sigma_z = np.zeros((self.obs_dim, self.obs_dim, self.nb_steps))

    @property
    def params(self):
        return self.A, self.H, self.K, self.D,\
               self.F, self.G, self.T, self.U,\
               self.V, self.X, self.Y, self.Z, self.y

    @params.setter
    def params(self, values):
        self.A, self.H, self.K, self.D,\
        self.F, self.G, self.T, self.U,\
        self.V, self.X, self.Y, self.Z, self.y = values


class AnalyticalLinearBeliefDynamics(LinearBeliefDynamics):
    def __init__(self, f_dyn, f_obs,
                 dyn_noise, obs_noise,
                 belief_dim, obs_dim, act_dim, nb_steps):
        super(AnalyticalLinearBeliefDynamics, self).__init__(belief_dim, obs_dim, act_dim, nb_steps)

        self.f = f_dyn
        self.h = f_obs

        self.dyn_noise = dyn_noise
        self.obs_noise = obs_noise

        self.dfdx = jacobian(self.f, 0)
        self.dhdx = jacobian(self.h, 0)

    def evalf(self, mu_b, u):
        return self.f(mu_b, u)

    def evalh(self, mu_b):
        return self.h(mu_b)

    # belief state dynamics
    def dynamics(self, mu_b, sigma_b, u):
        A = self.dfdx(mu_b, u)
        H = self.dhdx(self.evalf(mu_b, u))

        sigma_dyn = self.dyn_noise(mu_b, u)
        sigma_obs = self.obs_noise(self.evalf(mu_b, u))

        D = A @ sigma_b @ A.T + sigma_dyn
        D = 0.5 * (D + D.T)

        K = D @ H.T @ np.linalg.inv(H @ D @ H.T + sigma_obs)

        # stochastic mean dynamics
        f = self.evalf(mu_b, u)
        W = K @ H @ D

        # covariance dynamics
        phi = D - K @ H @ D
        phi = 0.5 * (phi + phi.T)

        return f, W, phi

    def taylor_expansion(self, b, u):
        for t in range(self.nb_steps):
            _in = tuple([b.mu[..., t], b.sigma[..., t], u[..., t]])

            _in_flat, _unflatten = flatten(_in)

            def _dyn_flat(_in_flat):
                return flatten(self.dynamics(*_unflatten(_in_flat)))[0]

            _dyn_jac = jacobian(_dyn_flat)

            _grads = _dyn_jac(_in_flat)
            self.F[..., t] = _grads[:self.belief_dim, :self.belief_dim]
            self.G[..., t] = _grads[:self.belief_dim, -self.act_dim:]

            self.X[..., t] = _grads[self.belief_dim:self.belief_dim + self.belief_dim * self.belief_dim, :self.belief_dim]
            self.Y[..., t] = _grads[self.belief_dim:self.belief_dim + self.belief_dim * self.belief_dim, self.belief_dim:self.belief_dim + self.belief_dim * self.belief_dim]
            self.Z[..., t] = _grads[self.belief_dim:self.belief_dim + self.belief_dim * self.belief_dim, -self.act_dim:]

            self.T[..., t] = _grads[self.belief_dim + self.belief_dim * self.belief_dim:, :self.belief_dim]
            self.U[..., t] = _grads[self.belief_dim + self.belief_dim * self.belief_dim:, self.belief_dim:self.belief_dim + self.belief_dim * self.belief_dim]
            self.V[..., t] = _grads[self.belief_dim + self.belief_dim * self.belief_dim:, -self.act_dim:]

    def forward(self, b, u, t):
        mu_bn, _, sigma_bn = self.dynamics(b.mu[..., t], b.sigma[..., t], u[..., t])
        return mu_bn, sigma_bn


class LinearControl:
    def __init__(self, belief_dim, act_dim, nb_steps):
        self.belief_dim = belief_dim
        self.act_dim = act_dim
        self.nb_steps = nb_steps

        self.K = np.zeros((self.act_dim, self.belief_dim, self.nb_steps))
        self.kff = np.zeros((self.act_dim, self.nb_steps))

    @property
    def params(self):
        return self.K, self.kff

    @params.setter
    def params(self, values):
        self.K, self.kff = values

    def action(self, b, alpha, bref, uref, t):
        db = b.mu[:, t] - bref.mu[:, t]
        return uref[:, t] + alpha * self.kff[..., t] + self.K[..., t] @ db
