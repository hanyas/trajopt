import autograd.numpy as np
import time

from trajopt.envs.quanser.common import QSocket, VelocityFilter
from trajopt.envs.quanser.cartpole.base import QCartpoleBase
from trajopt.envs.quanser.cartpole.ctrl import GoToLimCtrl


class QCartpoleRR(QCartpoleBase):
    def __init__(self, ip, fs_ctrl):
        super(QCartpoleRR, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)

        # Initialize Socket:
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0], u_len=self.action_space.shape[0])

        # Save the relative limits:
        self._calibrated = False
        self.c_lim = 0.075
        self._norm_x_lim = np.zeros(2, dtype=np.float32)

    def __del__(self):
        if self._qsoc.is_open():
            self.close()

    def _calibrate(self, verbose=False):
        if self._calibrated:
            return

        if verbose:
            print("\n\nCalibrate Cartpole:")

        # Retrieve current state:
        sensor = self._qsoc.snd_rcv(np.array([0.0]))

        # Reset calibration
        wcf = 62.8318
        zetaf = 0.9
        self._vel_filt_x = VelocityFilter(1, num=(wcf ** 2, 0), den=(1, 2 * wcf * zetaf, wcf ** 2),
                x_init=sensor[0:1], dt=self.timing.dt)
        self._vel_filt_th = VelocityFilter(1, num=(wcf ** 2, 0), den=(1, 2*wcf*zetaf, wcf**2),
                x_init=sensor[1:2], dt=self.timing.dt)

        # Go to the left:
        if verbose:
            print("\tGo to the Left:\t\t\t", end="")

        state = self._zero_sim_step()
        ctrl = GoToLimCtrl(state, positive=True)

        while not ctrl.done:
            u = ctrl(state)
            state = self._sim_step(u)

        if ctrl.success:
            self._norm_x_lim[1] = state[0]
            if verbose:
                print("\u2713")

        else:
            if verbose:
                print("\u274C ")
            raise RuntimeError("Going to the left limit failed.")

        # Go to the right
        if verbose:
            print("\tGo to the Right:\t\t", end="")

        state = self._zero_sim_step()
        ctrl = GoToLimCtrl(state, positive=False)

        while not ctrl.done:
            u = ctrl(state)
            state = self._sim_step(u)

        if ctrl.success:
            self._norm_x_lim[0] = state[0]
            if verbose:
                print("\u2713")
        else:
            if verbose:
                print("\u274C")
            raise RuntimeError("Going to the right limit failed.")

        # Activate the absolute cart position:
        self._calibrated = True

    def _center_cart(self, verbose=False):
        t_max = 10.0

        if verbose:
            print("\tCentering the Cart:\t\t", end="")

        # Center the cart:
        t0 = time.time()
        state = self._zero_sim_step()
        while (time.time() - t0) < t_max:
            a = -np.sign(state[0]) * 1.5 * np.ones(1)
            state = self._sim_step(a)

            if np.abs(state[0]) <= self.c_lim/10.:
                break

        # Stop the Cart:
        state = self._zero_sim_step()
        time.sleep(0.5)

        if np.abs(state[0]) > self.c_lim:
            if verbose: print("\u274C")
            time.sleep(0.1)
            raise RuntimeError("Centering of the cart failed. |x| = {0:.2f} > {1:.2f}".format(np.abs(state[0]), self.c_lim))

        elif verbose:
            print("\u2713")

    def _wait_for_upright_pole(self, verbose=False):
        if verbose:
            print("\tCentering the Pole:\t\t", end="")

        t_max = 15.0
        upright = False

        pos_th = np.array([self.c_lim, 2. * np.pi / 180.])
        vel_th = 0.1 * np.ones(2)
        th = np.hstack((pos_th, vel_th))

        # Wait until the pole is upright:
        t0 = time.time()
        while (time.time() - t0) <= t_max:
            state = self._zero_sim_step()

            transformed_state = np.array(state, copy=True)
            transformed_state[1] -= np.sign(transformed_state[1]) * np.pi
            if np.all(np.abs(transformed_state) <= th):
                upright = True
                break

        if not upright:
            if verbose:
                print("\u274C")
            time.sleep(0.1)

            state_str = np.array2string(np.abs(state), suppress_small=True,
                                        precision=2, formatter={'float_kind': lambda x: "{0:+05.2f}".format(x)})
            th_str = np.array2string(th, suppress_small=True, precision=2,
                                     formatter={'float_kind': lambda x: "{0:+05.2f}".format(x)})
            raise TimeoutError("The Pole is not upright, i.e., {0} > {1}".format(state_str, th_str))

        elif verbose:
            print("\u2713")

        return

    def _sim_step(self, a):
        pos = self._qsoc.snd_rcv(a)

        # Transform the relative cart position to [-0.4, +0.4]
        if self._calibrated:
            pos[0] = (pos[0] - self._norm_x_lim[0]) - 1./2. * (self._norm_x_lim[1] - self._norm_x_lim[0])

        x_dot = self._vel_filt_x(pos[0:1])
        th_dot = self._vel_filt_th(pos[1:2])

        # Normalize the angle from 0. to 2.*pi
        pos[1] = np.mod(pos[1], 2. * np.pi) - np.pi
        return np.concatenate([pos, x_dot, th_dot])

    def reset(self, verbose=True):

        # Reconnect to the system:
        self._qsoc.close()
        self._qsoc.open()

        # The system only needs to be calibrated once, as this is a bit time consuming:
        self._calibrate(verbose=verbose)

        # Center the cart in the middle @ x = 0.0
        self._center_cart(verbose=verbose)

        self._state = self._zero_sim_step()
        return self.step(np.array([0.0]))[0]

    def render(self, mode='human'):
        return

    def close(self):
        self.step(np.array([0.]))
        self._qsoc.close()
