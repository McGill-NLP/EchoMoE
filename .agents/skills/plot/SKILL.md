---
name: plot
description: Help create paper-quality plots and figures with matplotlib or seaborn. Use when the user asks for plots, figures, or visualizations.
---

# Plotting Assistant

Help the user create publication-quality figures.

## Required defaults (every plot)

These are non-negotiable. Apply them to every figure produced. Don't ship
a plot without them.

- **White background.** `figure.facecolor`, `axes.facecolor`, and
  `savefig.facecolor` all set to white. `savefig.transparent: False`.
- **Axis labels** on both axes, descriptive, with units in parentheses where
  meaningful. E.g. `"Training step (k)"`, `"Validation loss"`,
  `"GPU memory (GiB)"`. Never `"x"`, `"y"`, `"value"`, or default axis names.
- **Title** describing what the plot shows in one phrase. Model + metric +
  dataset where relevant. Left-aligned above the axes
  (`ax.set_title(..., loc="left")`).
- **Legend** for every series, placed **upper-center above the axes**, outside
  the plot box, no frame:
  ```python
  ax.legend(
      loc="lower center",
      bbox_to_anchor=(0.5, 1.02),
      ncol=<num_series>,   # one row if it fits
      frameon=False,
      handlelength=1.5,
  )
  ```
  (`loc="lower center"` anchors the legend's lower edge to `y=1.02`. I.e.,
  just above the plot. This avoids overlap with the title; if you also have a
  title, increase `y` to ~`1.1` so the legend sits above the title or move the
  title with `pad=...`.)
- **Use `constrained_layout=True`** when creating the figure so the
  outside-the-box legend doesn't get clipped on save:
  ```python
  fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
  ```
- **Error bars or shaded bands** whenever data is aggregated over seeds /
  runs. State what they represent (std, SEM, 95% CI) in the legend or caption.
- **`fig, ax` API.** Never `plt.plot()` directly. Makes multi-panel and
  customization sane.

## Style baseline (Google / Anthropic aesthetic)

Modern technical figures (Anthropic papers, Google Research blog, FT/NYT
graphics) share a visual grammar: heavy whitespace, minimal chartjunk, muted
but high-contrast colors, sans-serif throughout. Aim for that. Not
matplotlib defaults.

### Color palettes

**Google Material**. Vibrant, good for categorical comparisons of ≤4 series:

```python
GOOGLE = {
    "blue": "#4285F4",
    "red": "#EA4335",
    "yellow": "#FBBC04",
    "green": "#34A853",
    "grey": "#9AA0A6",  # de-emphasized series
    "text": "#202124",  # axis labels, titles
}
```

**Anthropic-style**. Warm, paper-friendly, good for highlighting one method
against grey baselines:

```python
ANTHROPIC = {
    "clay": "#CC785C",  # primary accent
    "rust": "#8C4B2F",  # darker accent for emphasis
    "sand": "#E8DCC4",  # background tint / fill
    "slate": "#3E3E3A",  # text
    "stone": "#A8A29E",  # de-emphasized baselines
}
```

**Palette rules:**

- One accent color for the result you're highlighting; greys for baselines.
- Cap categorical palettes at 5–6 colors. Beyond that → sequential colormap
  or use shape/linestyle encoding.
- For ordered data (depth, steps, temperature) → sequential map
  (`viridis`, `cividis`, `colormaps["Blues"]`).
- For diverging data centered at zero → `RdBu_r` or `coolwarm`.
- For colorblind safety, pair color with **linestyle** (solid / dashed /
  dotted) so the plot reads in greyscale too.

### rcParams config

Drop this at the top of any plotting script. It locks in the required
defaults plus the aesthetic.

```python
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        # Typography: sans-serif throughout
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "legend.borderaxespad": 0.0,
        "legend.handlelength": 1.5,
        # White backgrounds: required, locked here
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        # Spines: only bottom + left
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#202124",
        "axes.linewidth": 0.8,
        # Ticks: short, outward, subtle
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.color": "#202124",
        "ytick.color": "#202124",
        # Grid: light horizontal only
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E8EAED",
        "grid.linewidth": 0.6,
        # Lines
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        # Figure / export
        "figure.dpi": 120,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)
```

(In figure code: `ax.grid(axis="y")` to restrict gridlines to y only.
matplotlib has no single rcParam for axis-restricted grids.)

### Style principles

- **Spines off, grid soft.** Hide top/right; horizontal-only gridlines in
  light grey (`#E8EAED`).
- **One accent, many greys.** Highlight the line/bar that matters; render
  everything else in a single mid-grey. Reader's eye should land where you want.
- **Annotate key values.** `ax.annotate(...)` with a thin leader line is
  often clearer than the eyeballed comparison. E.g. "method X is +4.2 pts
  over baseline."
- **No 3D, no shadows, no gradient fills.** Ever.
- **Aspect ratio**: 1.6:1 to 2:1 (wider than tall) for time-series and bar
  charts. Square for scatter and heatmaps.

## Reference template

Copy this as a starting point. It includes every required default:

```python
import matplotlib.pyplot as plt
import numpy as np

# Assumes rcParams above has been applied
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

steps = np.arange(0, 50_000, 100)
for name, color in [("ours", GOOGLE["blue"]), ("baseline", GOOGLE["grey"])]:
    mean = ...  # shape (n_steps,)
    std = ...  # shape (n_steps,)
    ax.plot(steps, mean, label=name, color=color)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)

ax.set_xlabel("Training step")
ax.set_ylabel("Validation loss (cross-entropy)")
ax.set_title("Method comparison on FineWeb-Edu", loc="left", pad=25)
ax.grid(axis="y")

# Tick formatter: 10,000 → "10k"
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    handlelength=1.5,
)

fig.savefig("figures/loss_comparison.pdf")
fig.savefig("figures/loss_comparison.png", dpi=300)
```

## Other guidelines

### Aggregation over seeds

- For ≥3 seeds: plot the **mean as a solid line** and a **shaded ±std band**
  with `fill_between(..., alpha=0.15–0.25, linewidth=0)`. Bands beat
  error-bars-per-point on time-series.
- For categorical (bar charts): mean bar + thin error bar (`yerr=std`,
  `capsize=3`). Show individual seeds as dots if there are ≤10
  (`ax.scatter(...)` jittered).
- Always state in the caption / legend what the band or bar represents
  (e.g. "shaded: ±1 std across 5 seeds").

### Noisy training curves

- Plot the **raw curve at `alpha=0.25`** + a **smoothed line on top** (EMA
  with α=0.05–0.1 or rolling mean with window ≈ 1% of total steps).
- Don't oversmooth. The reader should still see the noise scale.

### Axis range

- **Bar charts: y-axis starts at 0.** Always. Truncating y-axes inflates
  apparent differences.
- **Line charts: non-zero start is fine** if the data warrants it, but don't
  truncate to deceive. Show the regime that's informative.
- **Log scale** when data spans ≥2 orders of magnitude. State the scale
  explicitly in the axis label: `"Loss (log scale)"`.

### Tick formatting

- Large counts: use `k` / `M` suffixes (`10_000` → `"10k"`,
  `1_500_000` → `"1.5M"`) via `FuncFormatter`.
- Raw integer counts under 100k: comma thousand-separators
  (`f"{x:,.0f}"`).
- Percentages: `PercentFormatter(xmax=1.0)` for `[0, 1]` data.
- Avoid scientific notation unless values are \<1e-3 or >1e6.

### Multi-panel figures

- `plt.subplots(nrows, ncols, sharex=True, sharey=True)` when panels show
  the same metric / range. Saves visual cognitive load.
- **Sub-panel labels**: `(a)`, `(b)`, `(c)` in the upper-left of each axes
  (`ax.text(0.02, 0.95, "(a)", transform=ax.transAxes, fontweight="bold")`).
- One **shared legend at the figure level** for multi-panel figures:
  `fig.legend(handles, labels, loc="upper center", ncol=N, bbox_to_anchor=(0.5, 1.05))`
  rather than repeating the legend in every panel.

### Reproducibility

- **Descriptive filenames.** `loss_qwen3_30b_finewebedu_3seeds.pdf`, not
  `fig1.pdf`. Future-you will thank present-you.
- **Save both PDF and PNG.** PDF (vector) for the paper, PNG for quick
  preview in slides, GitHub READMEs, etc.
- **Save the data alongside** when possible. Pickle / parquet of the
  arrays that produced the plot, in the same directory.
- **Record the source script**: a one-line comment in the script
  identifying which experiment / W&B run the data came from.
- **Figures path**: `figures/` or `plots/` at the repo root; never
  `$HOME` (see `cluster.md`).

### Caption-readiness

The figure should make sense **on its own**, without the surrounding text.

- Title states *what* is shown.
- Axis labels state *what's being measured* with units.
- Legend states *which series is which*.
- Annotations or a footnote inside the axes state *the headline finding*
  if it's not obvious from the curves.

A reader skimming a paper looks at figures first. Design for that reader.

### When to break the rules

These defaults exist to make plots good *by default*. Override deliberately:

- Conference templates with their own style guide → match the conference.
- A repo with established plotting conventions → match the repo (see
  \[\[principles\]\] Rule 11).
- A presentation slide where the figure needs to be readable from the back
  of the room → scale fonts up to ~16pt min, simplify further.

If you're breaking a default, say so in a comment in the plotting script
and explain why.

## Common Plot Types

- **Training curves**: loss / accuracy vs step / epoch. Shaded band over
  seeds; smoothing if noisy.
- **Bar charts**: method comparisons. Y-axis at 0, error bars for variability,
  optional seed dots.
- **Scatter plots**: correlation between metrics. Include the regression
  line + correlation coefficient if claiming a relationship.
- **Heatmaps**: confusion matrices, attention maps, sweep grids. Use a
  perceptually uniform colormap (`viridis`, `magma`); annotate cell values
  for small matrices.
- **Multi-panel**: shared axes when comparing the same metric across
  conditions; sub-panel labels `(a)`, `(b)`; one figure-level legend.

## Scope

$ARGUMENTS
