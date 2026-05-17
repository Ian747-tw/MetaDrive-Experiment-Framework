from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_metric_bars(path: str | Path, rows: list[dict], metric: str) -> None:
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    ax = df.plot.bar(x="method", y=metric)
    fig = ax.get_figure()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
