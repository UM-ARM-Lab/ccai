import pathlib
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import yaml

from ccai.constrained_svgd_trajopt import ConstrainedSteinTrajOpt
from ccai.quadrotor_env import QuadrotorEnv
from examples.quadrotor_example import QuadrotorProblem
from faster_csvto import FasterConstrainedSteinTrajOpt

SOLVE_OPTION = 'solve'
SOLVE_ITERATIONS = 2

UPDATE_OPTION = 'compute update'
UPDATE_ITERATIONS = 100

# PROFILE_OPTION = SOLVE_OPTION
PROFILE_OPTION = UPDATE_OPTION

USE_FAST = True
# USE_FAST = False


def main() -> None:
    # Initialize QuadrotorProblem.
    config = yaml.safe_load(pathlib.Path('config/dev_config.yaml').read_text())
    if config['obstacle_mode'] == 'none':
        config['obstacle_mode'] = None
    params = config.copy()
    print(params)
    env = QuadrotorEnv(False, 'surface_data.npz', obstacle_mode=config['obstacle_mode'],
                       obstacle_data_fname='obstacle_data_20.npz')
    env.reset()
    start = torch.from_numpy(env.state).to(dtype=torch.float32, device=params['device'])
    goal = torch.zeros(12, device=params['device'])
    goal[:2] = 4
    problem = QuadrotorProblem(start, goal, params['T'], device=params['device'], include_obstacle=False,
                               gp_sdf_model=None,
                               use_squared_slack=params['squared_slack'], compute_hessian=params['use_true_hess'])
    x_initial = problem.get_initial_xu(params.get('N'))
    resample = False

    # Initialize problems and check that their solutions are equivalent.
    faster_csvto = FasterConstrainedSteinTrajOpt(problem, params)
    csvto = ConstrainedSteinTrajOpt(problem, params)

    if PROFILE_OPTION == SOLVE_OPTION:
        # Check that solve results are equivalent.
        assert (torch.allclose(faster_csvto.solve(x_initial, resample), csvto.solve(x_initial, resample)))

        if USE_FAST:
            # Time a loop of solve calls for the FasterConstrainedSteinTrajopt.
            with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                         record_shapes=True,
                         profile_memory=True,
                         with_stack=True) as faster_csvto_profiler:
                with record_function("faster csvto solve()"):
                    for _ in range(SOLVE_ITERATIONS):
                        faster_csvto.solve(x_initial, resample)
        else:
            # Time a loop of solve calls for the FasterConstrainedSteinTrajopt.
            with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                         record_shapes=True,
                         profile_memory=True,
                         with_stack=True) as csvto_profiler:
                with record_function("csvto solve()"):
                    for _ in range(SOLVE_ITERATIONS):
                        csvto.solve(x_initial, resample)

    elif PROFILE_OPTION == UPDATE_OPTION:
        random_augmented_state = torch.rand(8, 192, dtype=csvto.dtype, device='cuda:0')

        # Check that update results are equivalent.
        assert (torch.allclose(faster_csvto.compute_update(random_augmented_state),
                               csvto.compute_update(random_augmented_state)))

        if USE_FAST:
            # Time a loop of compute update steps.
            with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                         record_shapes=True,
                         profile_memory=True,
                         with_stack=True) as faster_csvto_profiler:
                with record_function("faster csvto compute_update()"):
                    for _ in range(UPDATE_ITERATIONS):
                        faster_csvto.compute_update(random_augmented_state)
        else:
            # Time a loop of compute update steps.
            with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                         record_shapes=True,
                         profile_memory=True,
                         with_stack=True) as csvto_profiler:
                with record_function("csvto compute_update()"):
                    for _ in range(UPDATE_ITERATIONS):
                        csvto.compute_update(random_augmented_state)

    if USE_FAST:
        # Print profiling results for faster csvto.
        print('Profile for Faster CSVTO')
        print(faster_csvto_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(faster_csvto_profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    else:
        # Print profiling results for original csvto.
        print('Profile for Original CSVTO')
        print(csvto_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(csvto_profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))


if __name__ == '__main__':
    main()
