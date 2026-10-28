import argparse
from pathlib import Path

from spinup import ppo_pytorch, sac_pytorch, td3_pytorch
from spinup.utils.run_utils import ExperimentGrid


ALGORITHMS = {
    "ppo": ppo_pytorch,
    "sac": sac_pytorch,
    "td3": td3_pytorch,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        nargs="+",
        choices=ALGORITHMS.keys(),
        default=list(ALGORITHMS.keys()),
    )
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=1000)
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for algo_name in args.algo:
        algo_fn = ALGORITHMS[algo_name]

        eg = ExperimentGrid(name=f"{algo_name}-benchmark")

        eg.add("env_name", args.env, "", True)
        eg.add("seed", args.seeds)
        eg.add("epochs", args.epochs)
        eg.add("steps_per_epoch", args.steps_per_epoch)

        if algo_name in ("sac", "td3"):
            eg.add("start_steps", args.start_steps)
            eg.add("update_after", args.update_after)

        print(f"\n\nRunning benchmark for {algo_name.upper()}\n")

        eg.run(
            algo_fn,
            num_cpu=1,
            data_dir=str(results_dir),
        )


if __name__ == "__main__":
    main()
