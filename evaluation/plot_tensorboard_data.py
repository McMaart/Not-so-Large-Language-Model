import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable


def add_plot(filename: str, label: str, smoothing_factor: float = 0.9,
             semilogy: bool = False, skip_lines: int = 5):
    try:
        df = pd.read_csv(filename)[skip_lines:]
    except FileNotFoundError:
        print(f"File {filename} not found", file=sys.stderr)
        return

    df_smoothed = df.ewm(alpha=(1 - smoothing_factor)).mean()
    plt_func = plt.semilogy if semilogy is True else plt.plot
    plt_func(df_smoothed["Step"], df_smoothed["Value"], label=label)


def plot_tensorboard_loss(filenames: Iterable[str], labels: Iterable[str], title: str | None = None,
                          smoothing_factor: float = 0.9, skip_lines: int = 5):
    for filename, label in zip(filenames, labels):
        add_plot(filename, label, smoothing_factor, skip_lines=skip_lines)

    if title is not None:
        plt.title(title)
    plt.legend()
    plt.xlabel("Number of steps")
    plt.ylabel("Loss")
    plt.grid(alpha=0.275)
    plt.show()


if __name__ == "__main__":
    param_counts = ('1.1M', '3.7M', '8.3M')
    csv_files = [f"tensorboard_csv_data/{params}.csv" for params in param_counts]
    plot_tensorboard_loss(csv_files, param_counts, skip_lines=10,
                          title="Training loss for the Transformer models")
