import casadi as ca
import numpy as np

from nlp_building_blocks import NLPBuildingBlocks as bb


def gen_multi_point_rbf_loss_traj(qube_robot, via_points, integrator_dt=0.001, integrator_steps=20, control_steps=50,
                                  robot_step_rate=500):

    print("Preparing NLP for " + str(len(via_points)) + " via points")
    z, z_d, u, robot = bb.buildRobotDynamicalSystem(qube_robot)

    # Integrating to get trajectory for constant u in a small time-interval
    z_integrated = bb.rk_integration(z, z_d, u, integrator_dt, integrator_steps)

    # Assembling multiple of the constant u, small time-interval fractions for the trajectory model (independent parameters: U)
    z_init_state = [0, 0, 0, 0]
    z_states, u_outputs = bb.assemble_sym_traj(z_init_state, z, u, z_integrated, control_steps)

    num_vp = len(via_points)
    t = ca.SX.sym("t", num_vp)

    J_acc = 0
    for k in range(control_steps):
        for j in range(num_vp):
            J_acc += bb.gen_RBF(t[j], k, control_steps) * bb.quadr_task_space_loss(robot, z_states[k], via_points[j])

    # Trajectory length loss
    J_time = 0
    for j in range(num_vp):
        J_time += t[j] / num_vp

    J = J_acc + J_time * 0.0001#* (J_time**2 + 1)

    # Activation-time ordering constraints
    G_t = None
    lbg_t = None
    ubg_t = None
    lbg_t_base = np.array([0])
    ubg_t_base = np.array([1])
    first = True
    for j in range(num_vp):
        if first:
            first = False
            G_t = t[j]
            lbg_t = lbg_t_base
            ubg_t = ubg_t_base
        else:
            G_t = ca.vertcat(G_t, t[j] - t[j - 1])
            lbg_t = np.concatenate((lbg_t, lbg_t_base))
            ubg_t = np.concatenate((ubg_t, ubg_t_base))

    # Robot state constraints
    (G_Z, lbg_Z, ubg_Z) = bb.build_z_constraints(robot, z_states)

    # NLP formulation
    G_full = ca.vertcat(G_Z, ca.SX(G_t))
    nlp = {"x": ca.vertcat(u_outputs, t), "f": J, "g": G_full}

    # Parameter constraints
    arg = {}
    arg["lbx"] = ca.vertcat(np.ones(control_steps) * -5.0, np.zeros(num_vp))
    arg["ubx"] = ca.vertcat(np.ones(control_steps) * 5.0, np.ones(num_vp))
    arg["x0"] = ca.vertcat(np.zeros(control_steps), np.zeros(num_vp))
    arg["lbg"] = np.concatenate((lbg_Z, lbg_t))
    arg["ubg"] = np.concatenate((ubg_Z, ubg_t))

    print("Solving NLP")

    # Solver configuration
    opts = {"ipopt.tol": 1e-12, "expand": True}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    res = solver(**arg)

    print("NLP solved")

    optimal_param = np.array(res["x"])
    traj = optimal_param[0:control_steps]
    t_choice = optimal_param[control_steps:control_steps + num_vp]

    print("T choice " + str(t_choice))

    # Stretching the trajectory to match robot control frequency
    return (bb.stretch_traj(traj, integrator_dt * integrator_steps, robot_step_rate), t_choice)
