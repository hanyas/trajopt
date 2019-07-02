import autograd.numpy as np
import time

from trajopt.envs.quanser.cartpole.base import CartpoleDynamics


class PDCtrl:
    """
    PD controller for the cartpole environment.
    Flag `done` is set when `|x_des - x| < tol`.
    """

    def __init__(self, K=None, s_des=np.zeros(4), tol=5e-4):
        self.K = K if K is not None else np.array([20.0, 0.0, 0.0, 0.0])

        self.done = False
        self.s_des = s_des
        self.tol = tol

    def __call__(self, s):

        # Compute the voltage:
        err = self.s_des - s
        v = np.dot(self.K.transpose(), err)

        # Check for completion:
        if np.sum(err**2) <= self.tol:
            self.done = True

        return np.array([v], dtype=np.float32)


class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, s_init, positive=True):
        self.done = False
        self.success = False
        self.x_init = s_init[0]
        self.x_lim = 0.0
        self.xd_max = 1e-4
        self.delta_x_min = 0.1

        self.sign = 1 if positive else -1
        self.u_max = self.sign * np.array([1.5])

        self._t_init = False
        self._t0 = 0.0
        self._t_max = 10.0
        self._t_min = 2.0

    def __call__(self, s):
        x, _, xd, _ = s

        # Initialize the time:
        if not self._t_init:
            self._t0 = time.time()
            self._t_init = True

        # Compute voltage:
        if (time.time() - self._t0) < self._t_min:
            u = self.u_max

        elif np.abs(xd) < self.xd_max:
            u = np.zeros(1)
            self.success = True
            self.done = True

        elif (time.time() - self._t0) > self._t_max:
            u = np.zeros(1)
            self.success = False
            self.done = True

        else:
            u = self.u_max

        return u


class SwingUpCtrl:
    """Swing up and balancing controller"""

    def __init__(self, long=False, mu=18.5, u_max=18, v_max=24):
        self.dynamics = CartpoleDynamics(long=long)

        self.u_max = u_max
        self.v_max = v_max

        if long:
            self.Kpd = np.array([41.833, 189.8393, -47.8483, 28.0941])
        else:
            # Simulation:
            self.kp = 8.5                                              # Standard Value: 8.5
            self.ke = mu                                               # Standard Value: 21.5
            self.Kpd = np.array([41.83, -173.44, 46.14, -16.27])       # Standard Value: [41.8, -173.4, 46.1, -16.2]

    def __call__(self, state):
        x, theta, x_dot, theta_dot = state

        if theta > np.pi:
            alpha = - (2. * np.pi - theta)
        else:
            alpha = theta

        dyna = self.dynamics
        Mp = self.dynamics.mp
        pl = self.dynamics.pl
        Jp = self.dynamics.Jp

        Ek = Jp/2. * theta_dot**2
        Ep = Mp * dyna.g * pl * (1. - np.cos(theta + np.pi))     # E(pi) = 0., E(0) = E(2pi) = 2 mgl
        Er = 2 * Mp * dyna.g * pl                                # = 2 mgl

        if np.abs(alpha) < 0.1745:
            u = np.matmul(self.Kpd, (np.array([x, alpha, x_dot, theta_dot])))
        else:
            self.u_max = 180
            u = np.clip(self.ke * ((Ep + Ek) - Er) * np.sign(theta_dot * np.cos(theta + np.pi)) +
                        self.kp * (0.0 - x), -self.u_max, self.u_max)

        Vm = (dyna.Jeq * dyna.Rm * dyna.r_mp * u) / (dyna.eta_g * dyna.Kg * dyna.eta_m * dyna.Kt)\
              + dyna.Kg * dyna.Km * x_dot / dyna.r_mp
        Vm = np.clip(Vm, -self.v_max, self.v_max)

        return np.array([Vm])
