import numpy as np
from .base import QubeDynamics


class PDCtrl:
    """
    Slightly tweaked PD controller (increases gains if `x_des` not reachable).

    Accepts `th_des` and drives Qube to `x_des = (th_des, 0.0, 0.0, 0.0)`

    Flag `done` is set when `|x_des - x| < tol`.

    Tweak: increase P-gain on `th` if velocity is zero but the goal is still
    not reached (useful for counteracting resistance from the power cord).
    """

    def __init__(self, K=None, th_des=0.0, tol=5e-2):
        self.done = False
        self.K = K if K is not None else [2.5, 0.0, 0.5, 0.0]
        self.th_des = th_des
        self.tol = tol

    def __call__(self, x):
        th, al, thd, ald = x
        K, th_des, tol = self.K, self.th_des, self.tol
        all_but_th_squared = al ** 2 + thd ** 2 + ald ** 2
        err = np.sqrt((th_des - th) ** 2 + all_but_th_squared)
        if not self.done and err < tol:
            self.done = True
        elif th_des and np.sqrt(all_but_th_squared) < tol / 5.0:
            # Increase P-gain on `th` when struggling to reach `th_des`
            K[0] += 0.1 * K[0]
        return np.array([K[0]*(th_des - th) - K[1]*al - K[2]*thd - K[3]*ald])


class GoToLimCtrl:
    """Go to joint limits by applying `u_max`; save limit value in `th_lim`."""

    def __init__(self, fs_ctrl, positive=True):
        self.done = False
        self.th_lim = 10.0
        self.sign = 1 if positive else -1
        self.u_max = 1.0
        self.cnt = 0
        self.cnt_done = int(.4 * fs_ctrl)

    def __call__(self, x):
        th, _, thd, _ = x
        if np.abs(th - self.th_lim) > 0:
            self.cnt = 0
            self.th_lim = th
        else:
            self.cnt += 1
        self.done = self.cnt == self.cnt_done
        return np.array([self.sign * self.u_max])


class CalibrCtrl:
    """Go to joint limits, find midpoint, go to the midpoint."""

    def __init__(self, fs_ctrl):
        self.done = False
        self.go_right = GoToLimCtrl(fs_ctrl, positive=True)
        self.go_left = GoToLimCtrl(fs_ctrl, positive=False)
        self.go_center = PDCtrl()

    def __call__(self, x):
        u = np.array([0.0])
        if not self.go_right.done:
            u = self.go_right(x)
        elif not self.go_left.done:
            u = self.go_left(x)
        elif not self.go_center.done:
            if self.go_center.th_des == 0.0:
                self.go_center.th_des = \
                    (self.go_left.th_lim + self.go_right.th_lim) / 2
            u = self.go_center(x)
        elif not self.done:
            self.done = True
        return u


class EnergyCtrl:
    """PD controller on energy."""

    def __init__(self, Er, mu, a_max):
        self.Er = Er  # reference energy (J)
        self.mu = mu  # P-gain on the energy (m/s/J)
        self.a_max = a_max  # max acceleration of the pendulum pivot (m/s^2)
        self._dyn = QubeDynamics()  # dynamics parameters of the robot

    def __call__(self, x):
        _, al, _, ald = x
        Jp = self._dyn.Mp * self._dyn.Lp ** 2 / 12
        Ek = 0.5 * Jp * ald ** 2
        Ep = 0.5 * self._dyn.Mp * self._dyn.g * self._dyn.Lp * (1 - np.cos(al))
        E = Ek + Ep
        acc = np.clip(self.mu * (self.Er - E) * np.sign(ald * np.cos(al)),
                      -self.a_max, self.a_max)
        trq = self._dyn.Mr * self._dyn.Lr * acc
        voltage = -self._dyn.Rm / self._dyn.km * trq
        return np.array([voltage])


class SwingUpCtrl:
    """Hybrid controller (EnergyCtrl, PDCtrl) switching based on alpha."""

    def __init__(self, ref_energy=0.04, energy_gain=25.0, acc_max=5.0,
                 alpha_max_pd_enable=10.0, pd_gain=None):
        # Set up the energy pumping controller
        self.en_ctrl = EnergyCtrl(ref_energy, energy_gain, acc_max)
        # Set up the PD controller
        cos_al_delta = 1.0 + np.cos(np.pi - np.deg2rad(alpha_max_pd_enable))
        self.pd_enabled = lambda cos_al: np.abs(1.0 + cos_al) < cos_al_delta
        pd_gain = pd_gain if pd_gain is not None else [-1.0, 20.0, -0.5, 1.25]
        self.pd_ctrl = PDCtrl(K=pd_gain)

    def __call__(self, obs):
        th, cos_al, sin_al, th_d, al_d = obs

        x = np.r_[th,
                  np.arctan2(sin_al, cos_al),
                  th_d, al_d]
        if self.pd_enabled(cos_al):
            x[1] = x[1] % (2 * np.pi) - np.pi
            return self.pd_ctrl(x)
        else:
            return self.en_ctrl(x)
