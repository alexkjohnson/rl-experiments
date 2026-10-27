import argparse
from pathlib import Path

import torch
from spinup import ppo_pytorch
from spinup.utils.run_utils import ExperimentGrid


def train_ppo(**kwargs):
    """Run PPO with optional W&B tracking."""
    track = kwargs.pop('track', False)

    if not track:
        return ppo_pytorch(**kwargs)

    import wandb

    # Remove logger settings from the W&B config.
    logger_kwargs = kwargs['logger_kwargs']

    config = {
        key: value
        for key, value in kwargs.items()
        if key not in ('logger_kwargs', 'env_fn')
    }

    # Copy the nested dictionary so the PPO arguments remain unchanged.
    if 'ac_kwargs' in config:
        config['ac_kwargs'] = config['ac_kwargs'].copy()

    # Convert the activation class to a readable string for W&B.
    ac_kwargs = config.get('ac_kwargs')
    if ac_kwargs and 'activation' in ac_kwargs:
        ac_kwargs['activation'] = ac_kwargs['activation'].__name__

    wandb.init(
        project='ppo-hyperparameter-search',
        name=f"{logger_kwargs.get('exp_name')}-s{kwargs.get('seed', 0)}",
        config=config,
    )

    def log_epoch(metrics):
        wandb.log(metrics, step=metrics.get('Epoch'))

    logger_kwargs['epoch_callback'] = log_epoch

    try:
        return ppo_pytorch(**kwargs)
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--param',
        choices=['clip_ratio', 'train_pi_iters', 'target_kl', 'pi_lr',
            'normalize_advantages', 'anneal_lr'],
        default='clip_ratio',
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
        'clip_ratio': [0.1, 0.2, 0.3],
        'train_pi_iters': [20, 40, 80],
        'target_kl': [0.005, 0.01, 0.02],
        'pi_lr': [1e-4, 3e-4, 1e-3],
        'normalize_advantages': [True, False],
        'anneal_lr': [True, False],
    }

    eg = ExperimentGrid(name=f'ppo-{args.param}')

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
        train_ppo,
        num_cpu=args.cpu,
        data_dir=str(results_dir),
    )


if __name__ == '__main__':
    main()
