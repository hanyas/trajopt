import autograd.numpy as np
import warnings

from trajopt.envs.quanser.common import Base, LabeledBox, Timing


class QubeBase(Base):
    def __init__(self, fs, fs_ctrl):
        super(QubeBase, self).__init__(fs, fs_ctrl)
        self._state = None
        self._vel_filt = None
        self.timing = Timing(fs, fs_ctrl)

        # Limits
        act_max = np.array([3.0])
        state_max = np.array([2.0, 4.0 * np.pi, 30.0, 40.0])
        sens_max = np.array([2.0, np.inf])
        obs_max = np.array([2.0, 4.0 * np.pi, state_max[2], state_max[3]])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('theta', 'alpha'),
            low=-sens_max, high=sens_max, dtype=np.float32)
        self.state_space = LabeledBox(
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-state_max, high=state_max, dtype=np.float32)
        self.observation_space = LabeledBox(
            labels=('theta', 'alpha', 'th_d', 'al_d'),
            low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float32)
        self.reward_range = (0.0, self.timing.dt_ctrl)

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        return self._sim_step([0.0])

    def _lim_act(self, action):
        if np.abs(action) > 5.:
            warnings.warn("Control signal a = {0:.2f} should be between -5V and 5V.".format(action))
        return np.clip(action, -5., 5.)

    def _rwd(self, x, u):
        th, al, thd, ald = x
        cost = al**2 + 5e-3*ald**2 + 1e-1*th**2 + 2e-2*thd**2 + 3e-3*u[0]**2
        done = not self.state_space.contains(x)
        rwd = - cost * self.timing.dt_ctrl
        return np.float32(rwd), False

    def _observation(self, state):
        obs = np.float32([state[0], state[1],
                          state[2], state[3]])
        return obs


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(self):
        # Gravity
        self.g = 9.81

        # Motor
        self.Rm = 8.4    # resistance
        self.km = 0.042  # back-emf constant (V-s/rad)

        # Rotary arm
        self.Mr = 0.095  # mass (kg)
        self.Lr = 0.085  # length (m)
        self.Dr = 5e-5   # viscous damping (N-m-s/rad), original: 0.0015

        # Pendulum link
        self.Mp = 0.024  # mass (kg)
        self.Lp = 0.129  # length (m)
        self.Dp = 1e-5   # viscous damping (N-m-s/rad), original: 0.0005

        # Init constants
        self._init_const()

    def _init_const(self):
        # Moments of inertia
        Jr = self.Mr * self.Lr ** 2 / 12  # inertia about COM (kg-m^2)
        Jp = self.Mp * self.Lp ** 2 / 12  # inertia about COM (kg-m^2)

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + self.Mp * self.Lr ** 2
        self._c[1] = 0.25 * self.Mp * self.Lp ** 2
        self._c[2] = 0.5 * self.Mp * self.Lp * self.Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * self.Mp * self.Lp * self.g

    @property
    def params(self):
        params = self.__dict__.copy()
        params.pop('_c')
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, s, u):
        th, al, thd, ald = s
        voltage = u[0]

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * np.sin(al + np.pi) ** 2
        b = self._c[2] * np.cos(al + np.pi)
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * np.sin(2 * (al + np.pi)) * thd * ald \
            - self._c[2] * np.sin(al + np.pi) * ald * ald
        c1 = - 0.5 * self._c[1] * np.sin(2 * (al + np.pi)) * thd * thd +\
             self._c[4] * np.sin(al + np.pi)
        x = trq - self.Dr * thd - c0
        y = - self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
