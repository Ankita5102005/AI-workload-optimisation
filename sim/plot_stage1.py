"""Sanity-check plots for Stage 1: energy against configuration."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_workload(result, out_path) -> None:
    """Panels 1..k: energy vs power limit per precision (line per batch size).
    Last panel: energy vs runtime with the runtime limit, baseline and chosen config.
    Circles = accepted, crosses = rejected by the runtime constraint.
    """
    trials, base, best = result.trials, result.baseline, result.best
    precisions = [p for p in ("fp32", "fp16") if any(t.config.precision == p for t in trials)]
    batches = sorted({t.config.batch_size for t in trials})
    cmap = plt.get_cmap("viridis")
    color = {b: cmap(i / max(1, len(batches) - 1)) for i, b in enumerate(batches)}

    fig, axes = plt.subplots(1, len(precisions) + 1, figsize=(5.2 * (len(precisions) + 1), 4.4))

    for ax, prec in zip(axes, precisions):
        for b in batches:
            pts = sorted(
                (t for t in trials if t.config.precision == prec and t.config.batch_size == b),
                key=lambda t: t.config.power_limit_w,
            )
            if not pts:
                continue
            ax.plot([t.config.power_limit_w for t in pts], [t.energy_j for t in pts],
                    color=color[b], lw=1, alpha=0.6, label=f"bs={b}")
            for ok, marker in ((True, "o"), (False, "x")):
                sel = [t for t in pts if t.accepted == ok]
                if sel:
                    ax.scatter([t.config.power_limit_w for t in sel], [t.energy_j for t in sel],
                               color=color[b], marker=marker, s=28)
        ax.axhline(base.energy_j, ls="--", color="gray", lw=1)
        if best.config.precision == prec:
            ax.scatter([best.config.power_limit_w], [best.energy_j], marker="*", s=220,
                       color="red", zorder=5, label="chosen")
        ax.set_title(prec.upper())
        ax.set_xlabel("Power limit (W)")
        ax.set_ylabel("Energy per run (J)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    ax = axes[-1]
    for ok, marker, label in ((True, "o", "accepted"), (False, "x", "rejected")):
        sel = [t for t in trials if t.accepted == ok]
        ax.scatter([t.runtime_s for t in sel], [t.energy_j for t in sel], marker=marker, s=24,
                   alpha=0.7, label=label, color="tab:green" if ok else "tab:red")
    ax.axvline(base.runtime_s * result.max_slowdown, ls="--", color="k", lw=1,
               label=f"{result.max_slowdown:.2f}x baseline runtime")
    ax.scatter([base.runtime_s], [base.energy_j], marker="D", s=70, color="gray", zorder=5, label="baseline")
    ax.scatter([best.runtime_s], [best.energy_j], marker="*", s=220, color="red", zorder=5, label="chosen")
    ax.set_xlabel("Runtime per run (s)")
    ax.set_ylabel("Energy per run (J)")
    ax.set_title("Energy vs runtime")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"{result.workload}: chosen {best.config.label()} "
        f"({result.energy_saving_pct:.1f}% energy saved, {result.runtime_overhead_pct:+.1f}% runtime)"
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
