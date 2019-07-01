import numpy as np

from trajopt.envs.quanser.common import VelocityFilter
from trajopt.envs.quanser.cartpole.base import QCartpoleBase, X_LIM, CartpoleDynamics


class QCartpole(QCartpoleBase):
    def __init__(self, fs, fs_ctrl, long_pole=False):
        super(QCartpole, self).__init__(fs, fs_ctrl)
        self.dyn = CartpoleDynamics(long_pole)
        self._sim_state = None
        self.viewer = None

        # Transformations for the visualization:
        self.cart_trans = None
        self.pole_trans = None
        self.track = None
        self.axle = None

    def _calibrate(self):
        _wcf, _zetaf = 62.8318, 0.9  # filter params
        self._vel_filt = VelocityFilter(x_len=self.sensor_space.shape[0],
                                        x_init=np.array([0., np.pi]), dt=self.timing.dt,
                                        num=(_wcf ** 2, 0),
                                        den=(1, 2. * _wcf * _zetaf, _wcf ** 2))

        self._sim_state = np.array([0., np.pi + 0.01 * self._np_random.randn(), 0., 0.])
        self._state = self._zero_sim_step()

    def _sim_step(self, u):
        # Add a bit of noise to action for robustness
        u_noisy = u + 1e-6 * np.float32(
            self._np_random.randn(self.action_space.shape[0]))

        acc = self.dyn(self._sim_state, u_noisy)

        # Update internal simulation state
        self._sim_state[3] += self.timing.dt * acc[-1]
        self._sim_state[2] += self.timing.dt * acc[-2]
        self._sim_state[1] += self.timing.dt * self._sim_state[3]
        self._sim_state[0] += self.timing.dt * self._sim_state[2]

        # Pretend to only observe position and obtain velocity by filtering
        pos = self._sim_state[:2]
        # vel = self._sim_state[2:]
        vel = self._vel_filt(pos)
        return np.concatenate([pos, vel])

    def reset(self):
        self._calibrate()
        return self.step(np.array([0.0]))[0]

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = X_LIM
        scale = screen_width / world_width

        cart_y = 100  # Top of the cart
        pole_width = scale * 0.01
        pole_len = scale * self.dyn.pl
        cart_width = scale * 0.1
        cart_height = scale * 0.05

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = - cart_width/2., cart_width/2., cart_height/2., - cart_height/2.
            axle_offset = cart_height/4.

            # Plot cart:
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # Plot pole:
            l, r, t, b = - pole_width/2., pole_width/2., pole_len - pole_width/2., -pole_width/2.
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.pole_trans = rendering.Transform(translation=(0, axle_offset))

            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            # Plot axle:
            self.axle = rendering.make_circle(pole_width/2.)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            # Plot track:
            self.track = rendering.Line((0, cart_y), (screen_width, cart_y))
            self.track.set_color(0., 0., 0.)
            self.viewer.add_geom(self.track)

        # Update the visualization:
        if self._sim_state is None:
            return None

        x = self._sim_state
        cart_x = x[0] * scale + screen_width/2.0
        self.cart_trans.set_translation(cart_x, cart_y)
        self.pole_trans.set_rotation(x[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
