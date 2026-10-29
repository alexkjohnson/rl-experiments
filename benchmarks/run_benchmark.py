import torch
import argparse
from pathlib import Path

from spinup import ppo_pytorch, sac_pytorch, td3_pytorch
from spinup.utils.run_utils import ExperimentGrid


ALGORITHMS = {
    "ppo": ppo_pytorch,
    "sac": sac_pytorch,
    "td3": td3_pytorch,
}


def make_wandb_wrapper(algo_fn, algo_name, project):
    """Wrap an algorithm with optional W&B logging."""

    def train(**kwargs):
        import wandb

        # ExperimentGrid supplies logger_kwargs for each individual run.
        logger_kwargs = dict(kwargs.get("logger_kwargs", {}))
        experiment_name = logger_kwargs.get("exp_name", algo_name)
        seed = kwargs.get("seed", 0)

        # Record training parameters, excluding objects that W&B
        # cannot meaningfully serialize.
        config = {
            key: value
            for key, value in kwargs.items()
            if key not in ("logger_kwargs", "env_fn")
        }

        # Convert PyTorch activation classes into readable strings.
        ac_kwargs = config.get("ac_kwargs")
        if ac_kwargs is not None:
            ac_kwargs = dict(ac_kwargs)
            if "activation" in ac_kwargs:
                ac_kwargs["activation"] = (
                    ac_kwargs["activation"].__name__
                )
            config["ac_kwargs"] = ac_kwargs

        # Use one W&B run per seed, grouped by experiment.
        run = wandb.init(
            project=project,
            name=f"{experiment_name}-s{seed}",
            group=experiment_name,
            job_type=algo_name,
            config=config,
        )

        def log_epoch(metrics):
            # Epoch is a metric, so W&B can plot against it.
            run.log(metrics)

        logger_kwargs["epoch_callback"] = log_epoch
        kwargs["logger_kwargs"] = logger_kwargs

        try:
            return algo_fn(**kwargs)
        finally:
            run.finish()

    return train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        nargs="+",
        choices=ALGORITHMS.keys(),
        default=list(ALGORITHMS.keys()),
    )
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases tracking",
    )
    parser.add_argument(
        "--wandb_project",
        default="rl-algorithm-comparison",
        help="W&B project name",
    )
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for algo_name in args.algo:
        algo_fn = ALGORITHMS[algo_name]

        experiment_name = f"{algo_name}-benchmark-e{args.epochs}"

        if algo_name == "ppo":
            experiment_name += "-hid64-64"

        eg = ExperimentGrid(name=experiment_name)

        eg.add("env_name", args.env, "", True)
        eg.add("seed", args.seeds)
        eg.add("epochs", args.epochs)
        eg.add("steps_per_epoch", args.steps_per_epoch)

        if algo_name == "ppo":
            eg.add("ac_kwargs:hidden_sizes", (64, 64), "hid")
            eg.add("ac_kwargs:activation", torch.nn.Tanh, "")

        if algo_name in ("sac", "td3"):
            eg.add("start_steps", args.start_steps)
            eg.add("update_after", args.update_after)

        print(f"\n\nRunning benchmark for {algo_name.upper()}\n")

        if args.wandb:
            algo_fn = make_wandb_wrapper(algo_fn, algo_name, args.wandb_project)

        eg.run(
            algo_fn,
            num_cpu=1,
            data_dir=str(results_dir),
        )


if __name__ == "__main__":
    main()
