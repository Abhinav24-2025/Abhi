#!/usr/bin/env python3
"""
Animate runoff components (snowmelt, glacier melt, rainfall-runoff, baseflow)
for a basin as a stacked-area time series that draws itself over time, with a
side panel showing each component's value and share on the current date.

Input: a CSV with one row per time step (daily, monthly, ...), e.g.

    date,snowmelt,glacier_melt,rainfall,baseflow,observed
    2015-01-01,1.2,0.0,0.4,8.1,10.3
    ...

`observed` (measured discharge) is optional; when present it is drawn as a
line over the stacked components for comparison.

Examples
--------
    # your model output
    python runoff_animation.py --csv my_basin.csv --basin "Beas at Bhuntar" \
        --units "m³/s" --out runoff.mp4

    # column names differ in your file
    python runoff_animation.py --csv out.csv --date-col Date \
        --cols "Snow=snowmelt,Glacier=glacier_melt,Rain=rainfall,Base=baseflow" \
        --observed-col Qobs --out runoff.gif

    # try it without data (synthetic, clearly labelled as such)
    python runoff_animation.py --demo --out demo.gif
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

# Component order = stacking order (bottom -> top) = legend order.
COMPONENTS = ["snowmelt", "glacier_melt", "rainfall", "baseflow"]
LABELS = {
    "snowmelt": "Snowmelt",
    "glacier_melt": "Glacier melt",
    "rainfall": "Rainfall runoff",
    "baseflow": "Baseflow",
}
# Colour-blind-checked categorical palette (fixed order). Change here if you
# prefer domain colours; keep adjacent bands clearly distinct.
COLORS = {
    "snowmelt": "#2a78d6",
    "glacier_melt": "#eb6834",
    "rainfall": "#1baf7a",
    "baseflow": "#eda100",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="input CSV with date + component columns")
    p.add_argument("--demo", action="store_true",
                   help="use synthetic demo data instead of a CSV")
    p.add_argument("--date-col", default="date")
    p.add_argument("--cols", default="",
                   help="map CSV columns to components: "
                        "'CsvName=snowmelt,CsvName2=glacier_melt,...'")
    p.add_argument("--observed-col", default="observed",
                   help="optional observed discharge column (ignored if absent)")
    p.add_argument("--basin", default="Himalayan basin")
    p.add_argument("--units", default="m³/s")
    p.add_argument("--start", help="first date to plot, e.g. 2015-01-01")
    p.add_argument("--end", help="last date to plot")
    p.add_argument("--resample", default="",
                   help="pandas rule to aggregate first, e.g. 'W' or 'MS' (mean)")
    p.add_argument("--window", type=int, default=0,
                   help="show only the last N time steps (scrolling view); "
                        "0 = whole period, axis fixed")
    p.add_argument("--step", type=int, default=3,
                   help="time steps advanced per frame")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--out", default="runoff_animation.mp4",
                   help=".mp4 (needs ffmpeg) or .gif")
    return p.parse_args()


def demo_data():
    """Synthetic daily series with a rough Himalayan seasonality.
    NOT real data - for testing the animation only."""
    rng = np.random.default_rng(42)
    t = pd.date_range("2016-01-01", "2018-12-31", freq="D")
    doy = t.dayofyear.values

    def bump(center, width, amp):
        return amp * np.exp(-0.5 * ((doy - center) / width) ** 2)

    snow = bump(150, 30, 120) * rng.uniform(0.85, 1.15, len(t))      # pre-monsoon
    glacier = bump(215, 25, 90) * rng.uniform(0.9, 1.1, len(t))      # peak summer
    rain = bump(205, 28, 110) * rng.gamma(2.0, 0.5, len(t))          # monsoon, spiky
    base = 40 + bump(240, 60, 45) + rng.normal(0, 1.5, len(t))
    df = pd.DataFrame({"snowmelt": snow, "glacier_melt": glacier,
                       "rainfall": rain, "baseflow": base}, index=t)
    df["observed"] = df.sum(axis=1) * rng.normal(1.0, 0.07, len(t))
    return df.clip(lower=0)


def load_data(a):
    if a.demo:
        return demo_data()
    if not a.csv:
        sys.exit("Give --csv <file> or --demo")
    df = pd.read_csv(a.csv)
    if a.cols:
        mapping = dict(kv.split("=") for kv in a.cols.split(","))
        bad = set(mapping.values()) - set(COMPONENTS)
        if bad:
            sys.exit(f"--cols targets must be among {COMPONENTS}, got {sorted(bad)}")
        df = df.rename(columns=mapping)
    if a.date_col not in df.columns:
        sys.exit(f"Date column '{a.date_col}' not found. Columns: {list(df.columns)}")
    missing = [c for c in COMPONENTS if c not in df.columns]
    if missing:
        sys.exit(f"Missing component columns {missing}. Columns: {list(df.columns)}. "
                 "Use --cols to map them.")
    df[a.date_col] = pd.to_datetime(df[a.date_col])
    keep = COMPONENTS + ([a.observed_col] if a.observed_col in df.columns else [])
    df = df.set_index(a.date_col)[keep].sort_index()
    if a.observed_col in df.columns:
        df = df.rename(columns={a.observed_col: "observed"})
    return df


def main():
    a = parse_args()
    df = load_data(a)
    if a.start or a.end:
        df = df.loc[a.start:a.end]
    if a.resample:
        df = df.resample(a.resample).mean()
    n_nan = int(df[COMPONENTS].isna().sum().sum())
    if n_nan:
        print(f"Warning: {n_nan} missing component values filled with 0 for plotting.")
    df[COMPONENTS] = df[COMPONENTS].fillna(0)
    if len(df) < 2:
        sys.exit("Need at least 2 time steps.")

    has_obs = "observed" in df.columns
    dates = df.index
    vals = df[COMPONENTS].to_numpy().T            # (4, n)
    cum = np.cumsum(vals, axis=0)                 # stacked tops
    total = cum[-1]
    ymax = max(total.max(), df["observed"].max() if has_obs else 0) * 1.08

    # ---- figure layout -------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK})
    fig = plt.figure(figsize=(12, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.32,
                          left=0.07, right=0.97, top=0.84, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])
    for x in (ax, axb):
        x.set_facecolor(SURFACE)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)

    title = f"Runoff components — {a.basin}"
    if a.demo:
        title += "  (SYNTHETIC DEMO DATA)"
    fig.text(0.07, 0.94, title, fontsize=14, weight="bold", color=INK)
    date_txt = fig.text(0.07, 0.885, "", fontsize=11, color=INK_2)

    ax.set_ylim(0, ymax)
    ax.set_ylabel(f"Discharge ({a.units})")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if not a.window:
        ax.set_xlim(dates[0], dates[-1])

    # side panel: current value per component, horizontal bars
    # (bottom-to-top, same order as the stacked bands)
    ypos = np.arange(len(COMPONENTS))
    bars = axb.barh(ypos, np.zeros(len(COMPONENTS)), height=0.62,
                    color=[COLORS[c] for c in COMPONENTS])
    axb.set_yticks(ypos, [LABELS[c] for c in COMPONENTS])
    axb.set_xlim(0, vals.max() * 1.45)
    axb.tick_params(axis="y", length=0)
    axb.spines["left"].set_visible(False)
    axb.set_xlabel(f"Current ({a.units})")
    bar_txt = [axb.text(0, y, "", va="center", ha="left", fontsize=9, color=INK)
               for y in ypos]

    # legend (identity never colour-only: legend + labelled side panel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in COMPONENTS]
    names = [LABELS[c] for c in COMPONENTS]
    if has_obs:
        handles.append(plt.Line2D([], [], color=INK, lw=1.6))
        names.append("Observed")
    ax.legend(handles, names, loc="upper left", ncol=len(names), frameon=False,
              bbox_to_anchor=(0, 1.02), fontsize=9, handlelength=1.2)

    frames = list(range(1, len(dates), a.step))
    if frames[-1] != len(dates) - 1:
        frames.append(len(dates) - 1)
    hold = a.fps * 2                          # pause on the final frame
    frames += [len(dates) - 1] * hold

    dyn = []                                  # artists redrawn each frame

    def update(i):
        for art in dyn:
            art.remove()
        dyn.clear()
        lo = max(0, i - a.window) if a.window else 0
        sl = slice(lo, i + 1)
        x = dates[sl]
        prev = np.zeros(i + 1 - lo)
        for k, c in enumerate(COMPONENTS):
            top = cum[k, sl]
            # thin surface-coloured edge keeps adjacent bands separated
            dyn.append(ax.fill_between(x, prev, top, color=COLORS[c],
                                       lw=0.6, edgecolor=SURFACE))
            prev = top
        if has_obs:
            dyn.append(ax.plot(x, df["observed"].to_numpy()[sl], color=INK,
                               lw=1.4)[0])
        dyn.append(ax.axvline(dates[i], color=INK_2, lw=0.8, ls="--"))
        if a.window:
            ax.set_xlim(dates[lo], dates[max(i, lo + 1)])

        cur = vals[:, i]
        tot = cur.sum()
        for b, t, v, y in zip(bars, bar_txt, cur, ypos):
            b.set_width(v)
            share = 100 * v / tot if tot > 0 else 0
            t.set_position((v + axb.get_xlim()[1] * 0.02, y))
            t.set_text(f"{v:,.1f}  ({share:.0f}%)")
        obs = (f"   ·   observed {df['observed'].iloc[i]:,.1f} {a.units}"
               if has_obs and pd.notna(df["observed"].iloc[i]) else "")
        date_txt.set_text(f"{dates[i]:%d %b %Y}   ·   total simulated "
                          f"{tot:,.1f} {a.units}{obs}")
        return dyn

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / a.fps,
                         blit=False)
    if a.out.lower().endswith(".gif"):
        writer = PillowWriter(fps=a.fps)
    else:
        writer = FFMpegWriter(fps=a.fps, bitrate=2400)
    print(f"Rendering {len(frames)} frames -> {a.out}")
    anim.save(a.out, writer=writer, dpi=a.dpi, savefig_kwargs={"facecolor": SURFACE})
    print("Done.")


if __name__ == "__main__":
    main()
