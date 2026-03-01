import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_PALETTE: list[str] = [
    "#e63946",  # red
    "#2a9d8f",  # teal
    "#e9c46a",  # yellow
    "#457b9d",  # blue
    "#f4a261",  # orange
    "#8338ec",  # purple
    "#06d6a0",  # green
    "#fb5607",  # deep orange
    "#3a86ff",  # sky blue
    "#ff006e",  # pink
]


def plot_features(levels: list[tuple[pd.DataFrame, str]]) -> None:
    """
    Plots time series and histograms for each (DataFrame, label) pair.

    Args:
        levels (list[tuple[pd.DataFrame, str]]): List of (df, label) tuples to plot.
            Accepts 1 to 10 tuples. Each gets a unique color automatically.

    Returns:
        None
    """
    resolved = _resolve_levels(levels)
    max_cols = max(len(df.columns) for _, df, _ in resolved)

    fig = plt.figure(figsize=(10 * max_cols, 10 * len(resolved)))
    fig.suptitle(
        "Feature Analysis: " + " | ".join(label for label, _, _ in resolved),
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    outer = gridspec.GridSpec(len(resolved), 1, figure=fig)

    for row_idx, (label, df, color) in enumerate(resolved):
        _plot_level(fig, outer[row_idx], label, df, color)

    plt.show()


def _resolve_levels(levels: list[tuple[pd.DataFrame, str]]) -> list[tuple[str, pd.DataFrame, str]]:
    """
    Pairs each (df, label) tuple with an auto-assigned palette color.

    Args:
        levels (list[tuple[pd.DataFrame, str]]): Raw input tuples from the caller.

    Returns:
        list[tuple[str, pd.DataFrame, str]]: Ordered (label, df, color) triples.
    """
    if not levels:
        raise ValueError("At least one (DataFrame, label) tuple is required.")
    if len(levels) > len(_PALETTE):
        raise ValueError(f"Maximum {len(_PALETTE)} levels supported, got {len(levels)}.")

    return [(label, df, _PALETTE[idx]) for idx, (df, label) in enumerate(levels)]


def _plot_level(fig: plt.Figure, subplot_spec: gridspec.SubplotSpec, label: str, df: pd.DataFrame, color: str) -> None:
    """
    Plots a single level row: one time series + one histogram per column.

    Args:
        fig (plt.Figure): The parent figure.
        subplot_spec (SubplotSpec): The outer grid slot for this row.
        label (str): Display label for the row title.
        df (pd.DataFrame): Data to plot.
        color (str): Line/bar colour for this level.

    Returns:
        None
    """
    cols = df.columns.tolist()
    inner = gridspec.GridSpecFromSubplotSpec(2, len(cols), subplot_spec=subplot_spec)

    for col_idx, col in enumerate(cols):
        _plot_time_series(fig, inner[0, col_idx], label, col, df[col], color)
        _plot_histogram(fig, inner[1, col_idx], label, col, df[col], color)


def _plot_time_series(fig: plt.Figure, subplot_spec: gridspec.SubplotSpec, label: str, col: str, series: pd.Series, color: str) -> None:
    """
    Plots a single time series panel.

    Args:
        fig (plt.Figure): The parent figure.
        subplot_spec (SubplotSpec): Grid slot for this axes.
        label (str): Row label used in the panel title.
        col (str): Column name.
        series (pd.Series): Data to plot.
        color (str): Line colour.

    Returns:
        None
    """
    ax = fig.add_subplot(subplot_spec)
    ax.plot(series.index, series, color=color, linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"[{label}] {col}: Time Series", fontsize=9, fontweight="bold")
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)


def _plot_histogram(fig: plt.Figure, subplot_spec: gridspec.SubplotSpec, label: str, col: str, series: pd.Series, color: str) -> None:
    """
    Plots a single histogram panel.

    Args:
        fig (plt.Figure): The parent figure.
        subplot_spec (SubplotSpec): Grid slot for this axes.
        label (str): Row label used in the panel title.
        col (str): Column name.
        series (pd.Series): Data to histogram.
        color (str): Bar colour.

    Returns:
        None
    """
    clean = series.dropna()

    skewness = clean.skew()
    excess_kurtosis = clean.kurt()

    ax = fig.add_subplot(subplot_spec)
    ax.hist(clean, bins="auto", color=color)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"[{label}] {col}: Histogram", fontsize=9, fontweight="bold")
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

    ax.text(
        0.98,
        0.95,
        f"Skew: {skewness:.2f}\nKurt: {excess_kurtosis:.2f}",
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7),
    )
