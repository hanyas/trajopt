import autograd.numpy as np
import warnings

from trajopt.envs.quanser.common import Base, LabeledBox, Timing

X_LIM = 0.814


class QCartpoleBase(Base):
    def __init__(self, fs, fs_ctrl):
        super(QCartpoleBase, self).__init__(fs, fs_ctrl)
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        self.safe_range = 0.1
        self._x_lim = X_LIM / 2.      # [m]

        act_max = np.array([24.0])
        state_max = np.array([self._x_lim, np.inf, np.inf, np.inf])
        sens_max = np.array([np.inf, np.inf])
        obs_max = np.array([self._x_lim, np.inf, np.inf, np.inf])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('x', 'theta'),
            low=-sens_max, high=sens_max, dtype=np.float32)

        self.state_space = LabeledBox(
            labels=('x', 'theta', 'x_dot', 'theta_dot'),
            low=-state_max, high=state_max, dtype=np.float32)

        self.observation_space = LabeledBox(
            labels=('x',  'theta', 'x_dot', 'th_dot'),
            low=-obs_max, high=obs_max, dtype=np.float32)

        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)

        self.reward_range = (0., 2.)

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        return self._sim_step([0.0])

    def _lim_act(self, action):
        if np.abs(action) > 24.:
            warnings.warn("Control signal a = {0:.2f} should be between -24V and 24V.".format(action))
        return np.clip(action, -24., 24.)

    def _rwd(self, x, u):
        return np.array([0.], np.float32), False

    def _observation(self, state):
        """
        A observation is provided given the internal state.

        :param state: (x, theta, x_dot, theta_dot)
        :type state: np.array
        :return: (x, sin(theta), cos(theta), x_dot, theta_dot)
        :rtype: np.array
        """
        return np.array([state[0], state[1], state[2], state[3]])


class CartpoleDynamics:
    def __init__(self, long=False):

        self.g = 9.81              # Gravitational acceleration [m/s^2]
        self.eta_m = 1.            # Motor efficiency  []
        self.eta_g = 1.            # Planetary Gearbox Efficiency []
        self.Kg = 3.71             # Planetary Gearbox Gear Ratio
        self.Jm = 3.9E-7           # Rotor inertia [kg.m^2]
        self.r_mp = 6.35E-3        # Motor Pinion radius [m]
        self.Rm = 2.6              # Motor armature Resistance [Ohm]
        self.Kt = .00767           # Motor Torque Constant [N.zz/A]
        self.Km = .00767           # Motor Torque Constant [N.zz/A]
        self.mc = 0.37             # Mass of the cart [kg]

        if long:
            self.mp = 0.23         # Mass of the pole [kg]
            self.pl = 0.641 / 2.   # Half of the pole length [m]

            self.Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
            self.Bp = 0.0024       # Viscous coefficient at the pole 0.0024
            self.gain = 1.3
            self.scale = np.array([0.45, 1.])
        else:
            self.mp = 0.127        # Mass of the pole [kg]
            self.pl = 0.3365 / 2.  # Half of the pole length [m]
            self.Beq = 5.4         # Equivalent Viscous damping Coefficient 5.4
            self.Bp = 0.0024       # Viscous coefficient at the pole 0.0024
            self.gain = 1.5
            self.scale = np.array([1., 1.])

        # Compute Inertia:
        self.Jp = self.pl ** 2 * self.mp / 3.   # Pole inertia [kg.m^2]
        self.Jeq = self.mc + (self.eta_g * self.Kg ** 2 * self.Jm) / (self.r_mp ** 2)

    def __call__(self, s, v_m):
        x, theta, x_dot, theta_dot = s

        # Compute force acting on the cart:
        F = (self.eta_g * self.Kg * self.eta_m * self.Kt) / (self.Rm * self.r_mp) *\
            (-self.Kg * self.Km * x_dot / self.r_mp + self.eta_m * v_m)

        # Compute acceleration:
        A = np.array([[self.mp + self.Jeq, self.mp * self.pl * np.cos(theta + np.pi)],
                      [self.mp * self.pl * np.cos(theta + np.pi), self.Jp + self.mp * self.pl ** 2]])

        b = np.array([F[0] - self.Beq * x_dot - self.mp * self.pl * np.sin(theta + np.pi) * theta_dot ** 2,
                      0. - self.Bp * theta_dot - self.mp * self.pl * self.g * np.sin(theta + np.pi)])

        s_ddot = np.linalg.solve(A, b)
        return s_ddot
