import numpy as np
from scipy import optimize
import importlib
vp = None


class QubeEnv:
    # Sampling frequency
    fs = 500.0

    # Gravity
    g = 9.81

    # Motor
    Rm = 8.4  # resistance
    kt = 0.042  # current-torque (N-m/A)
    km = 0.042  # back-emf constant (V-s/rad)

    # Rotary arm
    Mr = 0.095  # mass (kg)
    Lr = 0.085  # length (m)
    Jr = Mr * Lr ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dr = - km ** 2 / Rm  # equivalent viscous damping coefficient (N-m-s/rad)

    # Pendulum link
    Mp = 0.024  # mass (kg)
    Lp = 0.129  # length (m)
    Jp = Mp * Lp ** 2 / 12  # moment of inertia about COM (kg-m^2)
    Dp = 0.0002  # equivalent viscous damping coefficient (N-m-s/rad)

    # Joint angles
    alpha = 0
    alpha_d = 0
    alpha_dd = 0
    theta = 0
    theta_d = 0
    theta_dd = 0

    theta_min = -2.3
    theta_max = 2.3
    u_max = 5.0

    u_dim = 1
    x_dim = 4
    x_labels = ('th', 'alpha', 'th_dot', 'alpha_dot')
    u_labels = ('Vm',)
    x_min = (-2.3, -np.inf, -np.inf, -np.inf)
    x_max = (2.3, np.inf, np.inf, np.inf)

    range = 0.2  # shows a 0.4m x 0.4m excerpt of the scene
    arm_radius = 0.003
    arm_length = 0.085
    pole_radius = 0.0045
    pole_length = 0.129

    shown_points = []
    scene = None

    def __init__(self, show_gui=True, center_camera=False, scene=None):
        self.show_gui = show_gui
        if show_gui:
            self.set_gui(scene, center_camera)

    # Define the GUI given a camera position [centered / normal]
    def set_gui(self, scene, center_camera):
        # import vpython globally
        global vp
        # Vpython scene: http://www.glowscript.org/docs/VPythonDocs/canvas.html
        vp = importlib.import_module("vpython")
        if scene is None:
            scene = vp.scene
        self.scene = scene
        self.scene.width = 800
        self.scene.height = 800
        self.scene.background = vp.color.gray(0.95)  # equals browser background -> higher = brighter
        self.scene.lights = []
        vp.distant_light(direction=vp.vector(0.2,  0.2,  0.5), color=vp.color.white)
        self.scene.up = vp.vector(0, 0, 1)  # z-axis is showing up
        # front view for projections
        if center_camera:
            self.scene.range = self.range
            self.scene.center = vp.vector(0, 0, 0)
            self.scene.forward = vp.vector(-1, 0, 0)
        # ...or custom view for better observation
        else:
            self.scene.range = self.range
            self.scene.center = vp.vector(0.04, 0, 0)
            self.scene.forward = vp.vector(-2, 1.2, -1)

        vp.box(pos=vp.vector(0, 0, -0.07), length=0.09, width=0.1, height=0.09, color=vp.color.gray(0.5))
        vp.cylinder(axis=vp.vector(0, 0, -1), radius=0.005, length=0.03, color=vp.color.gray(0.5))

        # Arm
        self.arm = vp.cylinder()
        self.arm.radius = self.arm_radius
        self.arm.length = self.arm_length
        self.arm.color = vp.color.blue
        # Pole
        self.pole = vp.cylinder()
        self.pole.radius = self.pole_radius
        self.pole.length = self.pole_length
        self.pole.color = vp.color.red

        self.curve = vp.curve(color=vp.color.white, radius=0.0005, retain=2000)

        self.render()

    # Reset global joint angles and re-render the scene
    def reset(self):
        # Joint angles
        self.alpha = 0
        self.alpha_d = 0
        self.alpha_dd = 0
        self.theta = 0
        self.theta_d = 0
        self.theta_dd = 0
        if self.show_gui:
            self.curve.clear()
        self.clear_rendered_points()
        self.render()

    def rk4(self, u, q):
        c1 = self.Mp * self.Lr**2 + 1./4 * self.Mp * self.Lp ** 2 - 1./4*self.Mp * self.Lp**2 * np.cos(q[1])**2 + self.Jr
        c2 = 1./2 * self.Mp * self.Lp * self.Lr * np.cos(q[1])
        c3 = u - self.Dr * q[2] - 0.5 * self.Mp * self.Lp**2 * np.sin(q[1]) * np.cos(q[1]) * q[2] * q[3] -\
             0.5 * self.Mp * self.Lp * self.Lr * np.sin(q[1]) * q[3]**2

        c4 = 0.5 * self.Mp * self.Lp * self.Lr * np.cos(q[1])
        c5 = self.Jp + 1./4 * self.Mp * self.Lp ** 2
        c6 = - self.Dp * q[3] + 1./4 * self.Mp * self.Lp ** 2 * np.cos(q[1]) * np.sin(q[1]) * q[2] ** 2 -\
             0.5 * self.Mp * self.Lp * self.g * np.sin(q[1])

        a = np.array([[c1, -c2], [c4, c5]])
        b = np.array([c3, c6])
        [th_dd, al_dd] = np.linalg.solve(a, b)
        return np.array([q[2], q[3], th_dd, al_dd])

    # Execute one step for a given action
    def step(self, action):
        # compute the applied torque from the control voltage (action)
        u = self.km * (action - self.km * self.theta_d) / self.Rm
        # clip the angle theta
        self.theta = np.clip(self.theta, self.theta_min, self.theta_max)
        # set torque to zero if pendulum is at it's max angle
        if self.theta == self.theta_min or self.theta == self.theta_max:
            self.theta_d = 0

        u_i = np.array([self.theta, self.alpha, self.theta_d, self.alpha_d])
        k1 = self.rk4(u, u_i)
        k2 = self.rk4(u, u_i + 1/(2.*self.fs) * k1)
        k3 = self.rk4(u, u_i + 1/(2.*self.fs) * k2)
        k4 = self.rk4(u, u_i + 1/self.fs * k3)

        [self.theta, self.alpha, self.theta_d, self.alpha_d] = u_i + 1 / (6 * self.fs) * (k1 + 2 * k2 + 2 * k3 + k4)

        return np.array([self.theta, self.alpha, self.theta_d, self.alpha_d])

    # Render global state
    def render(self, rate=500):
        if not self.show_gui:
            return

        # Position: End of the arm
        x_pos_arm = self.Lr * np.cos(self.theta)
        y_pos_arm = self.Lr * np.sin(self.theta)
        z_pos_arm = 0

        pos_axis, reachable = self.fwd_kin([self.theta, self.alpha])
        if not reachable:
            return

        # Direction: End of the arm -to- End of the pole (normalization not needed)
        x_axis_pole = pos_axis[0] - x_pos_arm
        y_axis_pole = pos_axis[1] - y_pos_arm
        z_axis_pole = pos_axis[2] - z_pos_arm

        # Render the computed positions
        self.arm.axis = vp.vector(x_pos_arm, y_pos_arm, z_pos_arm)
        self.pole.pos = vp.vector(x_pos_arm, y_pos_arm, z_pos_arm)
        self.pole.axis = vp.vector(x_axis_pole, y_axis_pole, z_axis_pole)

        self.curve.append(self.pole.pos + self.pole.axis)

        vp.rate(rate)

    # Forward kinematics
    def fwd_kin(self, q):
        th, al = q
        reachable = False
        # check if theta is in reachable space
        if self.theta_min <= th <= self.theta_max:
            reachable = True
            x = - self.Lp * np.sin(al) * np.sin(th) + self.Lr * np.cos(th)
            y = self.Lp * np.sin(al) * np.cos(th) + self.Lr * np.sin(th)
            z = - self.Lp * np.cos(al)
        # if theta not reachable, return some default values
        else:
            x, y, z = [0, 0, 0]

        return [x, y, z], reachable

    # Inverse kinematics
    def inv_kin(self, des, init=[0, 0]):
        """
        :param des: desired cartesian state
        :param init: initial guess for the joint angles
        :return:
        """
        x_des, y_des, z_des = des

        def f(x):
            th, al = x  # theta, alpha
            return np.array([
                - self.Lp * np.sin(al) * np.sin(th) + self.Lr * np.cos(th) - x_des,
                self.Lp * np.sin(al) * np.cos(th) + self.Lr * np.sin(th) - y_des,
                - self.Lp * np.cos(al) - z_des
            ])

        # Compute joint angles via least squares minimization given boundaries for theta
        boundaries = [[self.theta_min, -np.inf], [self.theta_max, np.inf]]
        res = optimize.least_squares(f, init, bounds=boundaries)
        if res.cost > 10e-8:
            res = optimize.least_squares(f, [1, 1], bounds=boundaries)
        return res.x, res.cost < 10e-8

    def inv_kin_pseudo(self, des, init=[0.01, 0.01], err_thold=0.01, max_steps=500, retries=100, alpha=0.01):
        """
        Compute the joint angles for a given cartesian position using the pseudo inverse
        :param des: desired cartesian position
        :param init: initial guess for theta & alpha
        :param err_thold: treshold for the error [abort condition]
        :param max_steps:
        :param retries:
        :param alpha: the rate of the gradient descent
        :return: q=[alpha, theta], succeeded flag
        """

        def j(x):
            al, th = x  # alpha , theta
            return np.array(
                [[-self.Lp * np.cos(al) * np.sin(th), -self.Lp * np.sin(al) * np.cos(th) - self.Lr * np.sin(th)],
                 [self.Lp * np.cos(al) * np.cos(th), -self.Lp * np.sin(al) * np.sin(th) + self.Lr * np.cos(th)],
                 [self.Lp * np.sin(al), 0]])

        def jt(x):
            return np.linalg.pinv(j(x))

        q_found = None
        for i in range(retries):
            # compute joint angles via jacobian pseudo inverse iterations
            q = np.concatenate([np.random.uniform(-np.pi, np.pi, 1), np.random.uniform(self.theta_min, self.theta_max, 1)])
            err = np.inf
            for i in range(max_steps):
                # Check if error is small enough
                if err < err_thold:
                    q_found = q
                    break
                # update the joint value
                J = j(q)
                J_t = J.T
                J_pinv = np.dot(np.linalg.inv(np.dot(J_t, J)), J_t)
                forw_cart = self.fwd_kin(np.flip(q, 0))[0]
                d_cart = np.array(des) - np.array(forw_cart)
                q_grad = np.dot(J_pinv, d_cart)
                q += alpha * q_grad
                # get the cartesian position for current q
                point, reachable = self.fwd_kin(np.flip(q, 0))
                # check if q is reachable
                if not reachable:
                    continue
                # compute the current cartesian error
                err = np.linalg.norm(np.array(des) - np.array(point))
            if q_found is not None:
                break

        q_output = np.zeros(2)
        if q_found is not None:
            # return joint angles and whether the error is small enough
            q_flip = np.flip(q, 0)
            q_circle = np.fmod(q_flip, 2*np.pi)
            q_circle_abs = np.abs(q_circle)
            q_clip = np.greater(q_circle_abs, np.pi)
            q_output = q_circle - np.multiply(np.multiply(q_clip, 2 * np.pi), np.sign(q_circle))
            print("q " + str(q_flip) + str(q_output))
        return q_output, q_found is not None

    # draw a projection line [-0.2 < y,z < 0.2]
    def line(self, y, z):
        if not self.show_gui:
            return
        line = vp.curve(color=vp.color.green, radius=0.001)
        line.append(0.5 * self.scene.camera.pos + vp.vector(0, 0.5 * y, 0.5 * z))
        line.append(vp.vector(0, y, z))
        line.append(- 0.5 * self.scene.camera.pos + vp.vector(0, 1.5 * y, 1.5 * z))

    # plot a point cloud for reachable end effector positions
    def point_cloud(self):
        if not self.show_gui:
            return
        all_thetas = np.arange(self.theta_min, self.theta_max, 0.1)
        all_alphas = np.arange(0, 2 * np.pi, 0.1)
        point_list = []
        for theta in all_thetas:
            for alpha in all_alphas:
                [x, y, z], reachable = self.fwd_kin([theta, alpha])
                # add the cartesian point coordinate
                point_list.append(vp.vector(x, y, z))

        # Plot all computed points
        vp.points(pos=point_list, radius=1, color=vp.color.gray(0.5))

    # Compute all possible joint angle states for a given 2-D position [condition: centered camera position]
    def projection(self, y, z):
        # equation for cut points between the projection line and the reachable sphere

        camera_pos = np.zeros(3)
        if self.show_gui:
            camera_pos[0] = self.scene.camera.pos.x
            camera_pos[1] = self.scene.camera.pos.y
            camera_pos[2] = self.scene.camera.pos.z
        else:
            camera_pos[0] = 0.34
            camera_pos[1] = 0
            camera_pos[2] = 0

        def f(k):
            return np.sqrt(((1-k) * camera_pos[0])**2 + (k*y)**2 + (k*z)**2) - np.sqrt(self.arm_length**2 + self.pole_length**2)

        # compute the possible cartesian positions
        k1 = optimize.fsolve(f, 0)
        k2 = optimize.fsolve(f, 2)

        x1 = camera_pos + (np.array((0, y, z)) - camera_pos) * k1[0]
        x2 = camera_pos + (np.array((0, y, z)) - camera_pos) * k2[0]

        # compute the angle positions if z-coordinate is reachable
        q = []
        if - self.pole_length <= x1[2] <= self.pole_length:
            q1, success = self.inv_kin([x1[0], x1[1], x1[2]])
            if success:
                q.append(q1)
                # print('Q1', q1)
        if - self.pole_length <= x2[2] <= self.pole_length:
            q2, success = self.inv_kin([x2[0], x2[1], x2[2]])
            if success:
                q.append(q2)
                # print('Q2', q2)
        # return a list of possible joint position [0-2 solutions possible]
        return q

    def render_point(self, point, color=None):
        if not self.show_gui:
            return
        if color is None:
            color = vp.color.white
        cart_point, reachable = self.fwd_kin(point)
        point = vp.points(pos=vp.vector(cart_point[0], cart_point[1], cart_point[2]), radius=5, color=color)
        self.shown_points.append(point)

    def clear_rendered_points(self):
        if not self.show_gui:
            return
        for point in self.shown_points:
            point.visible = False
            del point
