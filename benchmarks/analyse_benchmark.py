from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
PLOTS_DIR = BENCHMARK_DIR / "plots"

ALGORITHMS = ("ppo", "sac", "td3")


def main():
    runs = []

    for progress_file in RESULTS_DIR.rglob("progress.txt"):
        # Example: sac-benchmark_pendulum-v1
        experiment_name = progress_file.parent.parent.name

        if experiment_name.startswith("ppo-benchmark-hid64-32"):
            algo = "PPO-64x32"
        elif experiment_name.startswith("ppo-benchmark"):
            algo = "PPO-64x64"
        elif experiment_name.startswith("sac-benchmark"):
            algo = "SAC"
        elif experiment_name.startswith("td3-benchmark"):
            algo = "TD3"
        else:
            continue

        try:
            df = pd.read_csv(progress_file, sep="\t")
        except (OSError, pd.errors.ParserError) as exc:
            print(f"Skipping unreadable file {progress_file}: {exc}")
            continue

        metric = "AverageTestEpRet"

        if "Epoch" not in df.columns or metric not in df.columns:
            print(f"Skipping {progress_file}: missing Epoch or {metric}")
            continue

        df = df[["Epoch", metric]].copy()
        df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
        df["return"] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=["Epoch", "return"])

        if df.empty:
            print(f"Skipping empty run: {progress_file}")
            continue

        df["algo"] = algo.upper()
        df["seed_run"] = progress_file.parent.name

        runs.append(df[["Epoch", "return", "algo", "seed_run"]])

    if not runs:
        print(f"No usable benchmark logs found in {RESULTS_DIR}")
        return

    data = pd.concat(runs, ignore_index=True)

    # Report the number of distinct runs found for each algorithm.
    print("\nRuns found:")
    run_counts = data.groupby("algo")["seed_run"].nunique()

    print(run_counts.rename("number_of_runs").to_string())

    # Aggregate returns across seeds at each epoch.
    summary = (
        data.groupby(["algo", "Epoch"])["return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    print("\nFinal AverageTestEpRet by algorithm:")
    for algo, group in summary.groupby("algo", sort=True):
        last_epoch = group["Epoch"].max()
        final = group[group["Epoch"] == last_epoch].iloc[0]

        print(
            f"{algo}: "
            f"epoch={int(last_epoch)}, "
            f"mean={final['mean']:.2f}, "
            f"std={final['std']:.2f}, "
            f"seeds={int(final['count'])}"
        )

    # Plot mean return with a standard-deviation band.
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, group in summary.groupby("algo", sort=True):
        group = group.sort_values("Epoch")

        x = group["Epoch"]
        mean = group["mean"]
        std = group["std"].fillna(0)

        ax.plot(x, mean, label=algo)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average test episode return")
    ax.set_title("PPO vs SAC vs TD3 — Pendulum-v1")
    ax.legend(title="Algorithm")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output = PLOTS_DIR / "algorithm_comparison.png"
    fig.savefig(output, dpi=150)
    print(f"\nSaved plot to: {output}")

    plt.show()


if __name__ == "__main__":
    main()