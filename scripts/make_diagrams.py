"""Draw the pipeline flowchart and the meta-learner architecture diagrams.

Usage: python scripts/make_diagrams.py [output_dir]   (default: figures/diagrams)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "figures" / "diagrams"
out.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc, ec=None, tc=INK, fs=8.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.2,
                                 shrinkA=0, shrinkB=0))


# ---------- Figure: pipeline ----------
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.6)
ax.axis("off")
steps = [("MegaFon data\n600,000 rows\n50 features + T + Y", "#dbe8f8"),
         ("Validate\nmissing / duplicates\ngroup sizes", "#dbe8f8"),
         ("Balance check\nSMD < 0.1\nfor 50 features", "#dbe8f8"),
         ("Stratified split\n70% train\n30% test", "#dbe8f8")]
w, h, y0, gap = 2.05, 1.05, 2.3, 0.45
for i, (t, c) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    box(ax, x, y0, w, h, t, c)
    if i:
        arrow(ax, (x - gap, y0 + h / 2), (x, y0 + h / 2))
steps2 = [("Tune base learner\n3-fold CV Qini\n(train only)", "#fde3d8"),
          ("Fit uplift models\nS / T-learner, class transf.\n+ 2 baselines", "#fde3d8"),
          ("Evaluate on test\nQini, AUUC, uplift@30%\n5 repeated splits", "#fde3d8"),
          ("Targeting list\nrank by uplift\ntop 30% = target", "#d3f0e4")]
y1 = 0.35
for i, (t, c) in enumerate(steps2):
    x = 0.15 + (3 - i) * (w + gap)
    box(ax, x, y1, w, h, t, c)
    if i:
        arrow(ax, (x + w + gap, y1 + h / 2), (x + w, y1 + h / 2))
xr = 0.15 + 3 * (w + gap) + w / 2
arrow(ax, (xr, y0), (xr, y1 + h))
fig.tight_layout(pad=0.2)
fig.savefig(out / "pipeline.png", dpi=220)
plt.close(fig)

# ---------- Figure: meta-learner architectures ----------
fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
titles = ["S-learner (one model)", "T-learner (two models)", "Class transformation"]
for ax, t in zip(axes, titles):
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(t, fontsize=10, fontweight="bold", color=INK)
B, ORANGE, G, Y = "#dbe8f8", "#fde3d8", "#d3f0e4", "#fdeec8"
# S
a = axes[0]
box(a, 0.35, 3.0, 2.5, 0.65, "Features X  +  treatment flag T", B)
box(a, 0.35, 1.8, 2.5, 0.75, "Gradient-boosted trees\nf(x, t) ≈ P(Y=1 | x, t)", ORANGE, bold=False)
box(a, 0.1, 0.55, 1.4, 0.75, "f(x, 1)\nT set to 1", Y)
box(a, 1.7, 0.55, 1.4, 0.75, "f(x, 0)\nT set to 0", Y)
arrow(a, (1.6, 3.0), (1.6, 2.55))
arrow(a, (1.2, 1.8), (0.8, 1.3))
arrow(a, (2.0, 1.8), (2.4, 1.3))
a.text(1.6, 0.18, "uplift = f(x,1) − f(x,0)", ha="center", fontsize=9, fontweight="bold")
# T
a = axes[1]
box(a, 0.1, 3.0, 1.4, 0.65, "Treated rows\n(T = 1)", B)
box(a, 1.7, 3.0, 1.4, 0.65, "Control rows\n(T = 0)", B)
box(a, 0.1, 1.75, 1.4, 0.8, "Model f₁\nP(Y=1 | x, T=1)", ORANGE)
box(a, 1.7, 1.75, 1.4, 0.8, "Model f₀\nP(Y=1 | x, T=0)", ORANGE)
box(a, 0.6, 0.55, 2.0, 0.7, "Score every customer\nwith both models", Y)
arrow(a, (0.8, 3.0), (0.8, 2.55))
arrow(a, (2.4, 3.0), (2.4, 2.55))
arrow(a, (0.8, 1.75), (1.2, 1.25))
arrow(a, (2.4, 1.75), (2.0, 1.25))
a.text(1.6, 0.18, "uplift = f₁(x) − f₀(x)", ha="center", fontsize=9, fontweight="bold")
# CT
a = axes[2]
box(a, 0.35, 3.0, 2.5, 0.65, "Features X, outcome Y, flag T", B)
box(a, 0.35, 1.95, 2.5, 0.7, "New label Z = 1 if\n(T=1, Y=1) or (T=0, Y=0)", G)
box(a, 0.35, 0.85, 2.5, 0.7, "Gradient-boosted trees\ng(x) ≈ P(Z=1 | x)", ORANGE)
arrow(a, (1.6, 3.0), (1.6, 2.65))
arrow(a, (1.6, 1.95), (1.6, 1.55))
a.text(1.6, 0.18, "uplift = 2·g(x) − 1   (P(T=1) ≈ 0.5)", ha="center", fontsize=9, fontweight="bold")
fig.tight_layout(pad=0.4)
fig.savefig(out / "meta_learners.png", dpi=220)
plt.close(fig)
print(f"Diagrams written to {out}")
