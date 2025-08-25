import pathlib
import time
import numpy as np

import torch

from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv
from isaac_victor_envs.utils import get_assets_dir

from pydrake.all import Parser
from pyidto import FindIdtoResource

# Bridge and task helpers in idto
from ccai.idto.python_examples.isaac_bridge import (
    IdtoMpcController,
    action_sequence_from_solution_positions,
    map_isaac_to_drake_positions,
    map_isaac_to_drake_velocities,
    build_screwdriver_nominal_updater,
)
from ccai.idto.python_examples.screwdriver_task import (
    screwdriver_problem_ctor_factory,
    screwdriver_solver_params,
    screwdriver_q_guess_ctor_factory,
)


def build_drake_model_file_for_screwdriver():
    """
    Return the path to a Drake model capturing Allegro hand + screwdriver contacts.

    STUB: Uses idto/models/allegro_hand.sdf only as placeholder. You should
    create a jointed model that includes the screwdriver and the grasp for
    proper contact reasoning, or a reduced model consistent with the Isaac sim
    task's DOFs.
    """
    # Use the Isaac assets for robot and task object
    # Robot URDF (Allegro hand with sensors)
    allegro_urdf = str(pathlib.Path(get_assets_dir()) / "xela_models/allegro_hand_right.urdf")
    # Task/object URDF (screwdriver on table)
    screwdriver_urdf = str(pathlib.Path(get_assets_dir()) / "screwdriver/screwdriver.urdf")
    return [allegro_urdf, screwdriver_urdf]


def isaac_to_drake_state(env, plant):
    """
    Read Isaac env state and map to Drake q,v arrays.
    STUB: velocity mapping returns zeros; position mapping is identity.
    Implement the ordering/selection to match the Drake model.
    """
    state = env.get_state()
    # Expect state['q']: (num_envs, dof), we use env 0
    q_isaac = state['q'].reshape(-1).cpu().numpy()
    v_isaac = None  # env may not expose velocities; keep None to zero-fill
    q_drake = map_isaac_to_drake_positions(q_isaac, plant)
    v_drake = map_isaac_to_drake_velocities(v_isaac, plant)
    return q_drake, v_drake


def add_actuators_for_selected_fingers(plant):
    """
    Add actuators ONLY for index (hitosashi), middle (naka), and thumb (oya).
    Do NOT actuate ring (kusuri). Call before Finalize().
    """
    index_joints = [
        "allegro_hand_hitosashi_finger_finger_joint_0",
        "allegro_hand_hitosashi_finger_finger_joint_1",
        "allegro_hand_hitosashi_finger_finger_joint_2",
        "allegro_hand_hitosashi_finger_finger_joint_3",
    ]
    middle_joints = [
        "allegro_hand_naka_finger_finger_joint_4",
        "allegro_hand_naka_finger_finger_joint_5",
        "allegro_hand_naka_finger_finger_joint_6",
        "allegro_hand_naka_finger_finger_joint_7",
    ]
    thumb_joints = [
        "allegro_hand_oya_finger_joint_12",
        "allegro_hand_oya_finger_joint_13",
        "allegro_hand_oya_finger_joint_14",
        "allegro_hand_oya_finger_joint_15",
    ]

    actuated_joint_names = index_joints + middle_joints + thumb_joints
    for jname in actuated_joint_names:
        joint = plant.GetJointByName(jname)
        act_name = jname + "_act"
        try:
            # Avoid duplicate actuators if they exist
            if hasattr(plant, "HasJointActuatorNamed") and plant.HasJointActuatorNamed(act_name):
                continue
        except Exception:
            pass
        plant.AddJointActuator(act_name, joint)


def compute_actuated_position_indices(plant):
    """
    Return the position indices for index, middle, and thumb joints
    (in that order). Ring/kusuri is excluded.
    """
    groups = [
        [
            "allegro_hand_hitosashi_finger_finger_joint_0",
            "allegro_hand_hitosashi_finger_finger_joint_1",
            "allegro_hand_hitosashi_finger_finger_joint_2",
            "allegro_hand_hitosashi_finger_finger_joint_3",
        ],
        [
            "allegro_hand_naka_finger_finger_joint_4",
            "allegro_hand_naka_finger_finger_joint_5",
            "allegro_hand_naka_finger_finger_joint_6",
            "allegro_hand_naka_finger_finger_joint_7",
        ],
        [
            "allegro_hand_oya_finger_joint_12",
            "allegro_hand_oya_finger_joint_13",
            "allegro_hand_oya_finger_joint_14",
            "allegro_hand_oya_finger_joint_15",
        ],
    ]
    pos_inds = []
    for names in groups:
        for jname in names:
            joint = plant.GetJointByName(jname)
            pos_inds.append(joint.position_start())
    return pos_inds


def interpolate_targets_from_solution(solution, pos_indices, steps_per_action):
    q_now = np.array(solution.q[0]).reshape(-1)
    q_next = np.array(solution.q[1]).reshape(-1)
    q_now_sel = q_now[pos_indices]
    q_next_sel = q_next[pos_indices]
    seq = np.linspace(q_now_sel, q_next_sel, steps_per_action + 1)[1:]
    return seq


def make_problem_ctor(num_steps: int):
    # Initial conditions providers sized to plant
    def q_init_provider(plant):
        return np.zeros(plant.num_positions())

    def v_init_provider(plant):
        return np.zeros(plant.num_velocities())

    return screwdriver_problem_ctor_factory(num_steps, q_init_provider, v_init_provider)


def main():
    config = yaml.safe_load(pathlib.Path(f'{CCAI_PATH}/examples/config/screwdriver/allegro_screwdriver_IDTO.yaml').read_text())    
    # Isaac Gym env setup (single env)
    num_envs = 1
    steps_per_action = 60
    env = AllegroScrewdriverTurningEnv(num_envs, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=config['visualize'],
                                        steps_per_action=60,
                                        friction_coefficient=2.5,
                                        device=config['sim_device'],
                                        video_save_path=img_save_dir,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        gradual_control=False,
                                        gravity=True, 
                                        randomize_obj_start=config.get('randomize_obj_start', False),
                                        randomize_rob_start=config.get('randomize_rob_start', False),
                                        external_wrench_perturb=config.get('external_wrench_perturb', False),
                                        force_sensors=config.get('tactile_controller', False)
                                        )
    sim, gym, viewer = env.get_sim()

    # Drake model for optimization only
    model_files = build_drake_model_file_for_screwdriver()
    opt_dt = 0.05
    horizon_steps = 20

    # q_guess ctor sized to model
    q_guess_ctor = screwdriver_q_guess_ctor_factory(model_files, opt_dt, horizon_steps)

    # Problem ctor bound to plant sizes and default initial conditions
    problem_ctor = make_problem_ctor(horizon_steps)
    params_ctor = screwdriver_solver_params

    def post_parse_setup(plant):
        # STUB: If necessary, weld the base of the Allegro hand and the table frame
        # Example:
        hand_base = plant.GetFrameByName("allegro_hand_base_link")
        table = plant.GetFrameByName("table")
        plant.WeldFrames(plant.world_frame(), hand_base)
        plant.WeldFrames(plant.world_frame(), table)
        # Add actuators only for index, middle, and thumb (exclude ring/kusuri)
        add_actuators_for_selected_fingers(plant)
        # Implement as needed once you confirm frame names in the loaded URDFs.
        pass

    mpc = IdtoMpcController(
        model_file=model_files,
        opt_dt=opt_dt,
        problem_ctor=problem_ctor,
        params_ctor=params_ctor,
        q_guess_ctor=q_guess_ctor,
        post_parse_setup=post_parse_setup,
    )

    # Decide which generalized coordinate is screwdriver yaw in Drake
    # STUB: set to None; implement actual index after finalizing Drake model.
    # STUB: Identify the yaw index in Drake's generalized positions.
    screwdriver_yaw_index = -2

    # Target: turn clockwise by 90 degrees
    goal_delta_yaw = -np.pi / 2.0
    nominal_updater = build_screwdriver_nominal_updater(goal_delta_yaw, screwdriver_yaw_index)

    # Run for N cycles
    num_cycles = 50
    for i in range(num_cycles):
        # Read Isaac state and map to Drake
        q0, v0 = isaac_to_drake_state(env, mpc.plant)

        # Solve MPC in Drake
        solution, stats = mpc.step(q0, v0, maybe_update_nominal=nominal_updater)

        # Convert solution to position targets for impedance control using
        # ONLY index, middle, and thumb (exclude ring/kusuri)
        pos_indices = compute_actuated_position_indices(mpc.plant)
        seq = interpolate_targets_from_solution(
            solution, pos_indices, steps_per_action
        )

        # Apply to Isaac over steps_per_action micro-steps
        for q_target in seq:
            action = torch.tensor(q_target, dtype=torch.float32, device=env.device).reshape(1, -1)
            env.step(action)

        # Optional: logging
        if (i + 1) % 5 == 0:
            print(f"Completed MPC cycle {i+1}")

    # Cleanup
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()


