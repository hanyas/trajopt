import casadi as ca
import numpy as np


class NLPBuildingBlocks:
    @staticmethod
    def buildRobotDynamicalSystem(qube):
        # dynamic first grade ode vars
        z1 = ca.SX.sym("z1")  # alpha
        z2 = ca.SX.sym("z2")  # alpha_d
        z3 = ca.SX.sym("z3")  # theta
        z4 = ca.SX.sym("z4")  # theta_d

        # action
        u = ca.SX.sym("u")

        # formulating the dynamic equations with the symbolic variables

        tau = qube.km * (u - qube.km * z4) / qube.Rm

        c1 = qube.Mp * qube.Lr ** 2 + 1 / 4 * qube.Mp * qube.Lp ** 2 - 1 / 4 * qube.Mp * qube.Lp ** 2 * np.cos(z1) ** 2 + qube.Jr
        c2 = 1 / 2 * qube.Mp * qube.Lp * qube.Lr * np.cos(z1)
        c3 = tau - qube.Dr * z4 - 0.5 * qube.Mp * qube.Lp ** 2 * np.sin(z1) * np.cos(z1) * z4 * z2 - 0.5 * qube.Mp * qube.Lp * qube.Lr * np.sin(z1) * z2 ** 2

        c4 = 0.5 * qube.Mp * qube.Lp * qube.Lr * np.cos(z1)
        c5 = qube.Jp + 1 / 4 * qube.Mp * qube.Lp ** 2
        c6 = - qube.Dp * z2 + 1 / 4 * qube.Mp * qube.Lp ** 2 * np.cos(z1) * np.sin(z1) * z4 ** 2 - 0.5 * qube.Mp * qube.Lp * qube.g * np.sin(z1)

        # first order ODE
        z1_d = z2
        z2_d = 1./c2 * (c1 * (c2*c6+c3*c5)/(c2*c4+c1*c5) - c3)
        z3_d = z4
        z4_d = (c2*c6+c3*c5)/(c2*c4+c1*c5)

        z = ca.vertcat(z1, z2, z3, z4)
        z_d = ca.vertcat(z1_d, z2_d, z3_d, z4_d)

        robot = {
            "Lp": qube.Lp,
            "Lr": qube.Lr,
            "theta_min": qube.theta_min,
            "theta_max": 2.3
        }
        return z, z_d, u, robot

    @staticmethod
    def euler_integration(z, z_d, u, integrator_dt, integrator_steps):
        z_d_generator = ca.Function("z_d_generator", [z, u], [z_d])

        current_z = z
        for i in range(integrator_steps):
            current_z_d = z_d_generator(current_z, u)
            current_z += integrator_dt * current_z_d
        return current_z

    @staticmethod
    def rk_integration(z, z_d, u, integrator_dt, integrator_steps):
        f = ca.Function('z_dot', [z, u], [z_d])
        h = integrator_dt * integrator_steps

        k1 = f(z, u)
        k2 = f(z + h/2. * k1, u)
        k3 = f(z + h/2. * k2, u)
        k4 = f(z + h    * k3, u)

        return z + h/6. * (k1 + 2.*k2 + 2.*k3 + k4)

    @staticmethod
    def assemble_sym_traj(z_init_state, z, u, z_integrated, control_steps):
        control_step_generator = ca.Function("control_step_generator", [z, u], [z_integrated])

        u_outputs = ca.SX.sym("u_outputs", control_steps)

        init_z_state = ca.SX(z_init_state)
        z_states = [init_z_state]
        for i in range(control_steps):
            current_z = control_step_generator(z_states[-1], u_outputs[i])
            z_states.append(current_z)
        return z_states, u_outputs

    @staticmethod
    def build_z_constraints(robot, z_states):
        G_Z = None
        lbg_Z = None
        ubg_Z = None
        lbg_Z_base = np.array([
            float("-Inf"),
            float("-Inf"),
            robot["theta_min"],
            float("-Inf")
        ])
        ubg_Z_base = np.array([
            float("Inf"),
            float("Inf"),
            robot["theta_max"],
            float("Inf")
        ])
        for z_state in z_states:
            if G_Z is None:
                G_Z = z_state
                lbg_Z = lbg_Z_base
                ubg_Z = ubg_Z_base
            else:
                G_Z = ca.vertcat(G_Z, z_state)
                lbg_Z = np.concatenate((lbg_Z, lbg_Z_base))
                ubg_Z = np.concatenate((ubg_Z, ubg_Z_base))
        return G_Z, lbg_Z, ubg_Z

    @staticmethod
    def quadr_task_space_loss(robot, z_state, des):
        # Quadratic loss for distance to target point in task space
        J_x = - robot["Lp"] * np.sin(z_state[0]) * np.sin(z_state[2]) + robot["Lr"] * np.cos(z_state[2])
        J_y = robot["Lp"] * np.sin(z_state[0]) * np.cos(z_state[2]) + robot["Lr"] * np.sin(z_state[2])
        J_z = - robot["Lp"] * np.cos(z_state[0])
        return (J_x - des[0])**2 + (J_y - des[1])**2 + (J_z - des[2])**2

    @staticmethod
    def linear_task_space_loss(robot, z_state, des):
        # Quadratic loss for distance to target point in task space
        J_x = - robot["Lp"] * np.sin(z_state[0]) * np.sin(z_state[2]) + robot["Lr"] * np.cos(z_state[2])
        J_y = robot["Lp"] * np.sin(z_state[0]) * np.cos(z_state[2]) + robot["Lr"] * np.sin(z_state[2])
        J_z = - robot["Lp"] * np.cos(z_state[0])
        return ca.sqrt((J_x - des[0])**2 + (J_y - des[1])**2 + (J_z - des[2])**2)

    @staticmethod
    def gen_RBF(t, step, step_count):
        """generate RBF to activate each single via-point loss at variable time-steps"""
        return ca.exp(-((t * step_count - step) ** 2))

    @staticmethod
    def stretch_traj(traj, control_step_time, robot_step_rate):
        control_step_multiple = np.round(control_step_time * robot_step_rate)
        print("Control step multiple " + str(control_step_multiple))

        stretched_traj = traj
        if control_step_multiple > 1:
            stretched_traj = np.repeat(traj, control_step_multiple)
        else:
            if control_step_multiple < 1:
                combination_count = int(np.round(1/(control_step_time * robot_step_rate)))
                print("combination count " + str(combination_count) + ", fds " + str(traj.shape[0]))
                if combination_count > 1:
                    traj = traj if traj.shape[0] % combination_count == 0 else np.append(traj, traj[-(combination_count - traj.shape[0] % combination_count)])
                    combined_traj = []
                    print("combination count " + str(combination_count) + ", fds " + str(traj.shape[0]))
                    max_index = int(traj.shape[0] / combination_count)
                    print("max index " + str(max_index))
                    for i in range(max_index):
                        combined_traj.append(np.mean(traj[i * combination_count:min(i * combination_count + combination_count, traj.shape[0])]))
                    stretched_traj = np.array(combined_traj)
        return stretched_traj
