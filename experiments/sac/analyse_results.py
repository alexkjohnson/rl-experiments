from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / 'results'
METRIC = 'AverageTestEpRet'


def main():
    runs = []

    for progress_file in RESULTS_DIR.rglob('progress.txt'):
        config_name = progress_file.parent.parent.name
        df = pd.read_csv(progress_file, sep='\t')

        if METRIC not in df.columns:
            print(f'Skipping {progress_file}: missing {METRIC}')
            continue

        df['config'] = config_name
        df['seed_run'] = progress_file.parent.name

        runs.append(df[['Epoch', METRIC, 'config', 'seed_run']])

    if not runs:
        print(f'No progress logs found in {RESULTS_DIR}')
        return

    data = pd.concat(runs, ignore_index=True)

    summary = (
        data.groupby(['config', 'Epoch'])[METRIC]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    print(f'\nMean {METRIC} at each configuration’s final epoch:')
    print(
        summary.groupby('config', sort=True)
        .tail(1)[['config', 'Epoch', 'mean', 'std', 'count']]
        .to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for config, group in summary.groupby('config', sort=True):
        group = group.sort_values('Epoch')

        x = group['Epoch']
        mean = group['mean']
        std = group['std'].fillna(0)

        ax.plot(x, mean, marker='o', label=config)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average test episode return')
    ax.set_title('SAC: fixed alpha vs automatic entropy tuning')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output = RESULTS_DIR / 'alpha_comparison.png'
    fig.savefig(output, dpi=150)
    print(f'\nSaved plot to: {output}')

    plt.show()


if __name__ == '__main__':
    main()