"""The two report figures: the OOB curve and forest tuning (inner macro-F1)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from run import GRIDS, OOB, chosen_n_trees, forest_inner  # noqa: E402

OUT = str(Path(__file__).parent / "report" / "fig-{}.png")
BLUE, ORANGE, GREY = "#1f4e9a", "#d95f02", "#4d4d4d"
# class weight -> (label, colour, marker, marker face)
CW = {"none": ("none", BLUE, "o", BLUE),
      "balanced": ("balanced", ORANGE, "s", ORANGE),
      "balanced_subsample": ("balanced per bootstrap", GREY, "^", "white")}
plt.rcParams.update({"font.family": "serif", "font.size": 8})


def style(ax, xlabel, ylabel=None):
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(OUT.format(name), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT.format(name))


def oob():
    o = pd.read_csv(OOB)
    fig, ax = plt.subplots(figsize=(3.5, 2.35))      # one IEEE column
    for i, (_, f) in enumerate(o.groupby(["repeat", "outer"])):
        ax.plot(f.n_trees, f.oob_f1, color="0.78", linewidth=0.6,
                label="single outer fold" if i == 0 else None)
    m = o.groupby("n_trees").oob_f1.mean()
    n_folds = o.groupby(["repeat", "outer"]).ngroups
    ax.plot(m.index, m.values, color=BLUE, marker="o", markersize=3.5, linewidth=1.4,
            label=f"mean over {n_folds} folds")
    nt = chosen_n_trees()
    ax.axvline(nt, color="black", linestyle="--", linewidth=0.8)
    ax.text(nt + 20, ax.get_ylim()[0] + 0.0008, f"{nt} trees")
    style(ax, "Number of trees", "OOB macro-F1")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.12))
    save(fig, "oob")


def forest_tuning():
    """Mean inner macro-F1 per setting, one panel per minimum leaf size."""
    f = forest_inner()
    order = list(dict.fromkeys(p["mtry"] for p in GRIDS["forest"]))
    ticks = ["sqrt\n(6)" if m == "sqrt" else "all\n(bagging)" if m == "all" else str(m)
             for m in order]
    leaves = sorted(f.min_leaf.unique())
    fig, axes = plt.subplots(1, len(leaves), figsize=(3.5, 2.45), sharey=True)
    for ax, leaf in zip(axes, leaves):
        for cw, (label, colour, marker, face) in CW.items():
            g = f[(f.min_leaf == leaf) & (f.cw == cw)]
            m = g.groupby("mtry").macro_f1.mean().reindex(order)
            ax.plot(range(len(order)), m.values, color=colour, marker=marker, markersize=4,
                    markerfacecolor=face, linewidth=1.2, label=label)
        ax.set_xticks(range(len(order)), ticks)
        ax.set_title(f"minimum leaf size {leaf}", fontsize=8)
        style(ax, "Features per split")
    axes[0].set_ylabel("Inner macro-F1")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "maxfeat")


if __name__ == "__main__":
    oob()
    forest_tuning()
