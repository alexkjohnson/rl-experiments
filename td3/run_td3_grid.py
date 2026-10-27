import argparse
from pathlib import Path

import torch
from spinup import td3_pytorch
from spinup.utils.run_utils import ExperimentGrid


def train_td3(**kwargs):
    """Run TD3 with optional W&B tracking."""
    track = kwargs.pop('track', False)

    if not track:
        return td3_pytorch(**kwargs)

    import wandb

    # Remove logger settings from the W&B config.
    logger_kwargs = kwargs['logger_kwargs']

    config = {
        key: value
        for key, value in kwargs.items()
        if key not in ('logger_kwargs', 'env_fn')
    }

    # Copy the nested dictionary so the TD3 arguments remain unchanged.
    if 'ac_kwargs' in config:
        config['ac_kwargs'] = config['ac_kwargs'].copy()

    # Convert the activation class to a readable string for W&B.
    ac_kwargs = config.get('ac_kwargs')
    if ac_kwargs and 'activation' in ac_kwargs:
        ac_kwargs['activation'] = ac_kwargs['activation'].__name__

    wandb.init(
        project='td3-experiments',
        name=f"{logger_kwargs.get('exp_name')}-s{kwargs.get('seed', 0)}",
        config=config,
    )

    def log_epoch(metrics):
        wandb.log(metrics, step=metrics.get('Epoch'))

    logger_kwargs['epoch_callback'] = log_epoch

    try:
        return td3_pytorch(**kwargs)
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--param',
        choices=['policy_delay'],
        default='policy_delay',
    )
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--track', action='store_true',
                        help='Track experiments with Weights & Biases',
                        )

    args = parser.parse_args()

    values = {
        'policy_delay': [1, 2, 4],
    }
    eg = ExperimentGrid(name=f'td3-{args.param}')

    # Keep the environment and network architecture fixed.
    eg.add('env_name', 'Pendulum-v1', '', True)
    eg.add('ac_kwargs:hidden_sizes', (64, 64), 'hid')
    eg.add('ac_kwargs:activation', torch.nn.Tanh, '')

    # Repeat each hyperparameter configuration across different seeds.
    eg.add('seed', [10 * i for i in range(args.num_seeds)])

    # Keep the training budget fixed.
    eg.add('epochs', args.epochs)
    eg.add('steps_per_epoch', args.steps_per_epoch)

    # Vary only the selected hyperparameter.
    eg.add(args.param, values[args.param], args.param)

    eg.add('track', args.track)

    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    eg.run(
        train_td3,
        num_cpu=args.cpu,
        data_dir=str(results_dir),
    )


if __name__ == '__main__':
    main()
