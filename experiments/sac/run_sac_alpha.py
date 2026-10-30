import argparse
from pathlib import Path

import gymnasium as gym
from spinup.algos.pytorch.sac.sac import sac
from spinup.utils.run_utils import ExperimentGrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-alpha')

    # Keep the environment and architecture fixed.
    eg.add('env_name', 'Pendulum-v1', '', True)
    eg.add('ac_kwargs:hidden_sizes', (256, 256), 'hid')
    
    # Compare the original fixed temperature with automatic tuning.
    eg.add('auto_alpha', [False, True], 'auto')

    # Repeat each configuration with the same set of seeds.
    eg.add('seed', [i for i in range(args.num_seeds)])

    # Keep training settings identical.
    eg.add('epochs', args.epochs)
    eg.add('steps_per_epoch', args.steps_per_epoch)
    eg.add('start_steps', 100)
    eg.add('update_after', 100)
    eg.add('update_every', 50)
    eg.add('batch_size', 32)
    eg.add('num_test_episodes', 3)
    eg.add('max_ep_len', 200)
    eg.add('alpha', 0.2)

    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    eg.run(
        sac,
        num_cpu=args.cpu,
        data_dir=str(results_dir),
    )


if __name__ == '__main__':
    main()
