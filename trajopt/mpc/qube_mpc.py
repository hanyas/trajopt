import math
import numpy as np
import matplotlib.pyplot as plt

from qube_env import QubeEnv
from multiple_point_rbf_activated_loss import gen_multi_point_rbf_loss_traj


def query_float_input(caption, relentless=True, default=None):
    number = None
    while number is None:
        try:
            print(str(caption) + ":", end="")
            number = float(input())
        except Exception:
            number = default if default is not None else (None if relentless is True else float("nan"))
            pass
    return number


def play_trajectory(qube, traj):
    qube.reset()
    n_cycles = traj.shape[0]

    print("Playing trajectory of length " + str(n_cycles))
    for i in range(n_cycles):
        print("[" + str(round((i + 1)/n_cycles * 100)) + "%] Qube-Loss " + str([loss(via_point, qube.alpha, qube.theta) for via_point in via]))
        u = traj[i]
        x = qube.step(u)
        qube.render()


def exact_point_tracking(qube):
    traj = None
    via = None

    while True:
        cmd = None
        if traj is not None:
            print("New Trajectory [n] or Replay [r]: ", end="")
            cmd = input()

        if cmd == 'n' or traj is None:
            via = []
            via_points = []
            add_point_cmd = None
            while len(via) == 0 or add_point_cmd == 'y':

                y_des = query_float_input("Target y (horizontal)")
                z_des = query_float_input("Target z (vertical)  ")

                projected_joint_ang = qube.projection(y_des, z_des)
                if len(projected_joint_ang) > 0:
                    print("Projected joint angles " + str(projected_joint_ang))
                    forw_cart = qube.fwd_kin(projected_joint_ang[0])
                    if(forw_cart[1] == True):
                        via_points.append(projected_joint_ang[0])
                        via.append(forw_cart[0])
                else:
                    print("No projectedJointAngles found")

                try:
                    print("Add another point? [y/n]:", end="")
                    add_point_cmd = input()
                except Exception:
                    add_point_cmd = None
                    pass

            integrator_dt = query_float_input("integrator dt", default=0.001)

            integrator_steps = math.floor(query_float_input("integrator steps", default=20))

            control_steps = math.floor(query_float_input("control steps", default=100))

            qube.reset()

            for via_point in via_points:
                qube.render_point(via_point)

            (traj, t_choice) = gen_multi_point_rbf_loss_traj(qube,
                                                             via,
                                                             integrator_dt=integrator_dt,
                                                             integrator_steps=integrator_steps,
                                                             control_steps=control_steps,
                                                             robot_step_rate=500)
            plt.plot(traj)
            plt.show()

        qube.reset()

        # Plot the via points with vPython
        for via_point in via_points:
            qube.render_point(via_point)

        def loss(via_point, alpha, theta):
            J_x = - qube.Lp * np.sin(alpha) * np.sin(theta) + qube.Lr * np.cos(theta)
            J_y = qube.Lp * np.sin(alpha) * np.cos(theta) + qube.Lr * np.sin(theta)
            J_z = - qube.Lp * np.cos(alpha)
            return (via_point[0] - J_x)**2 + (via_point[1] - J_y)**2 + (via_point[2] - J_z)**2

        n_cycles = traj.shape[0]
        real_traj = []
        for i in range(n_cycles):
            print(str(round((i+1)/n_cycles*100)) + '% | Loss = ' + str([loss(via_point, qube.alpha, qube.theta)
                                                                        for via_point in via]))
            u = traj[i]
            x = qube.step(u)
            real_traj.append(np.array(x))
            qube.render()
        real_traj = np.array(real_traj)


if __name__ == "__main__":
    print("Start")
    print("Hit enter to use the default config")
    qube = QubeEnv(center_camera=True)
    exact_point_tracking(qube)
    print("Finish")
