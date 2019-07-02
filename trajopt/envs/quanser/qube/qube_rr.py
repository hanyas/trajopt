import autograd.numpy as np

from trajopt.envs.quanser.common import QSocket, VelocityFilter
from trajopt.envs.quanser.qube.base import QubeBase
from trajopt.envs.quanser.qube.ctrl import CalibrCtrl


class QubeRR(QubeBase):
    def __init__(self, ip, fs_ctrl):
        super(QubeRR, self).__init__(fs=500.0, fs_ctrl=fs_ctrl)
        self._qsoc = QSocket(ip, x_len=self.sensor_space.shape[0],
                             u_len=self.action_space.shape[0])
        self._sens_offset = None

    def _calibrate(self):
        # Reset calibration
        self._vel_filt = VelocityFilter(self.sensor_space.shape[0])
        self._sens_offset = np.zeros(self.sensor_space.shape[0],
                                     dtype=np.float32)

        # Record alpha offset if alpha == k * 2pi (happens upon reconnect)
        x = self._zero_sim_step()
        if np.abs(x[1]) > np.pi:
            diff = 1.0
            while diff > 0.0:
                xn = self._zero_sim_step()
                diff = np.linalg.norm(xn - x)
                x = xn
            self._sens_offset[1] = x[1]

        # Find theta offset by going to joint limits
        x = self._zero_sim_step()
        act = CalibrCtrl()
        while not act.done:
            x = self._sim_step(act(x))
        self._sens_offset[0] = (act.go_right.th_lim + act.go_left.th_lim) / 2

        # Set current state
        self._state = self._zero_sim_step()

    def _sim_step(self, x, u):
        pos = self._qsoc.snd_rcv(u)
        pos -= self._sens_offset
        return np.concatenate([pos, self._vel_filt(pos)])

    def reset(self):
        self._qsoc.close()
        self._qsoc.open()
        self._calibrate()
        return self.step(np.array([0.0]))[0]

    def render(self, mode='human'):
        return

    def close(self):
        self._zero_sim_step()
        self._qsoc.close()
