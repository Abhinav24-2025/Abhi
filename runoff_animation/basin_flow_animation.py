#!/usr/bin/env python3
"""
Map animation of runoff flowing through a basin's river network.

Coloured particles (snowmelt, glacier melt, rainfall runoff, baseflow) are
released along the rivers and travel downstream to the outlet. The number of
particles released for each component follows your monthly runoff table, and
the rivers widen as total runoff rises.

Inputs
------
  --boundary  basin outline (shapefile / GeoJSON / GeoPackage)
  --rivers    river network lines (any digitising direction; flow direction is
              worked out from the network and the outlet)
  --dem       elevation raster (GeoTIFF, PCRaster .map, ...) - hillshade
              background, outlet detection, and where particles start
  --data      runoff table: SPHY-style Years/Months/QALLDTS/STotDTS/RTotDTS/
              GTotDTS/BTotDTS (.xlsx/.csv/.txt), or date + component columns
  --glaciers  (optional) glacier outlines, e.g. RGI; glacier-melt particles
              then start inside glaciers instead of the highest terrain

IMPORTANT - what is and is not data-driven
  * Amount of each component through time: from your table (outlet values).
  * WHERE particles start is illustrative: the table holds outlet totals, not
    per-cell values, so snowmelt starts on high ground, glacier melt on the
    highest ground (or in --glaciers polygons), rain and baseflow anywhere.
  * Particle speed is not to scale (real travel times are hours to days).
  The figure carries a footnote saying this.

Example
-------
    python basin_flow_animation.py --boundary basin.shp --rivers rivers.shp \\
        --dem dem.tif --data sphy_output.xlsx --basin "My basin" \\
        --start 2000-01 --end 2002-12 --out basin_flow.mp4
"""
import argparse
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from runoff_animation import COMPONENTS, LABELS, load_data

# Particle colours, checked for colour-blind separation when all four are mixed
# together on the dark map background (white/cyan/blue/amber).
COLORS = {
    "snowmelt": "#e8e6df",
    "glacier_melt": "#3fc1d6",
    "rainfall": "#3987e5",
    "baseflow": "#c98500",
}
BG = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
INK_3 = "#8a8980"
RIVER = "#5b6f86"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boundary", required=True)
    p.add_argument("--rivers", required=True)
    p.add_argument("--dem", required=True)
    p.add_argument("--data", required=True, help="runoff table (see above)")
    p.add_argument("--sheet", default=0, help="Excel sheet name/index")
    p.add_argument("--glaciers", help="optional glacier polygons (e.g. RGI)")
    p.add_argument("--outlet", help="outlet as 'lon,lat' (WGS84). "
                                    "Default: lowest river end point on the DEM")
    p.add_argument("--basin", default="Himalayan basin")
    p.add_argument("--units", default="m³/s")
    p.add_argument("--start", help="first month, e.g. 2000-01")
    p.add_argument("--end", help="last month, e.g. 2002-12")
    p.add_argument("--frames-per-step", type=int, default=12,
                   help="animation frames per time step (month)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--travel-frames", type=int, default=60,
                   help="frames for a particle to travel the longest river path")
    p.add_argument("--max-spawn", type=float, default=45,
                   help="particles released per frame at the highest total runoff")
    p.add_argument("--snap", type=float, default=0,
                   help="snap distance (m) to join river lines that do not share "
                        "vertices exactly. Default: 2x the vertex spacing")
    p.add_argument("--spacing", type=float, default=0,
                   help="river vertex spacing in m (default: auto)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--out", default="basin_flow.mp4", help=".mp4 (ffmpeg) or .gif")
    return p.parse_args()


# ---------------------------------------------------------------- GIS input
def load_dem(path, crs, max_px=1400):
    """Read the DEM reprojected to `crs`, at most `max_px` pixels wide."""
    with rasterio.open(path) as src:
        src_crs = src.crs
        if src_crs is None:
            sys.exit(f"{path} has no CRS. Assign one first (e.g. in QGIS).")
        transform, w, h = calculate_default_transform(
            src_crs, crs, src.width, src.height, *src.bounds)
        scale = max(1.0, w / max_px)
        w, h = int(w / scale), int(h / scale)
        transform = transform * transform.scale(scale, scale)
        dem = np.full((h, w), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), dem, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return dem, transform


def sample(dem, transform, xy):
    col, row = ~transform * (xy[:, 0], xy[:, 1])
    row = np.clip(row.astype(int), 0, dem.shape[0] - 1)
    col = np.clip(col.astype(int), 0, dem.shape[1] - 1)
    return dem[row, col]


def hillshade(dem, transform, az=315, alt=45):
    z = np.where(np.isnan(dem), np.nanmin(dem), dem)
    dy, dx = np.gradient(z, abs(transform.e), transform.a)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(az), np.radians(alt)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def build_network(rivers, spacing, snap):
    """River lines -> graph of evenly spaced vertices.
    Returns node coords, edge list (i, j, length)."""
    key_to_id, coords, node_line = {}, [], []
    edges = []
    q = spacing / 20  # vertices closer than this are merged

    def node(x, y, line_id):
        k = (round(x / q), round(y / q))
        if k not in key_to_id:
            key_to_id[k] = len(coords)
            coords.append((x, y))
            node_line.append(line_id)
        return key_to_id[k]

    ends = []
    for li, geom in enumerate(rivers.geometry):
        g = geom.segmentize(spacing)
        xy = np.asarray(g.coords)[:, :2]
        ids = [node(x, y, li) for x, y in xy]
        ends += [ids[0], ids[-1]]
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                edges.append((a, b, float(np.hypot(*np.subtract(coords[a], coords[b])))))
    coords = np.array(coords)
    node_line = np.array(node_line)

    # join line ends that touch another line without sharing a vertex
    tree = cKDTree(coords)
    deg = np.bincount(np.array([e[:2] for e in edges]).ravel(), minlength=len(coords))
    n_snapped = 0
    for e in set(ends):
        if deg[e] != 1:
            continue
        d, idx = tree.query(coords[e], k=min(12, len(coords)), distance_upper_bound=snap)
        for dist, j in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            if np.isfinite(dist) and j != e and node_line[j] != node_line[e]:
                edges.append((e, int(j), max(float(dist), 1e-6)))
                n_snapped += 1
                break
    print(f"River network: {len(coords)} vertices, {len(edges)} edges, "
          f"{n_snapped} line ends snapped")
    return coords, np.array(edges)


def orient_to_outlet(coords, edges, outlet):
    """Shortest path along the network to the outlet -> downstream neighbour
    of every vertex (`down`) and distance to outlet (`dist`)."""
    n = len(coords)
    i, j, w = edges[:, 0].astype(int), edges[:, 1].astype(int), edges[:, 2]
    g = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    dist, pred = dijkstra(g, indices=outlet, return_predecessors=True)
    down = pred.astype(int)          # predecessor seen from the outlet = downstream
    down[outlet] = outlet
    return dist, down


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)

    # runoff table (reuses the loader from runoff_animation.py)
    df = load_data(SimpleNamespace(demo=False, csv=a.data, sheet=a.sheet, cols="",
                                   date_col="date", observed_col="observed"))
    if a.start or a.end:
        df = df.loc[a.start:a.end]
    df[COMPONENTS] = df[COMPONENTS].fillna(0).clip(lower=0)
    if len(df) < 2:
        sys.exit("Need at least 2 time steps in the selected period.")
    monthly = bool((df.index.day == 1).all())
    q = df[COMPONENTS].to_numpy()                      # (T, 4)
    q_ref = q.sum(axis=1).max()

    # vector data in a metric CRS
    boundary = gpd.read_file(a.boundary)
    crs = boundary.estimate_utm_crs() if boundary.crs.is_geographic else boundary.crs
    boundary = boundary.to_crs(crs)
    basin_geom = boundary.union_all()
    rivers = gpd.read_file(a.rivers).to_crs(crs).explode(index_parts=False)
    rivers = rivers[rivers.geometry.notna() & (rivers.geom_type == "LineString")]
    if rivers.empty:
        sys.exit("No line features found in --rivers.")
    total_len = rivers.length.sum()
    spacing = a.spacing or max(total_len / 6000, 20)
    snap = a.snap or 2 * spacing

    dem, dtrans = load_dem(a.dem, crs)
    coords, edges = build_network(rivers, spacing, snap)
    z = sample(dem, dtrans, coords)

    # outlet
    n_comp, label = connected_components(coo_matrix(
        (np.ones(len(edges)), (edges[:, 0].astype(int), edges[:, 1].astype(int))),
        shape=(len(coords),) * 2), directed=False)
    if a.outlet:
        lon, lat = map(float, a.outlet.split(","))
        pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(crs)[0]
        outlet = int(cKDTree(coords).query([pt.x, pt.y])[1])
    else:
        main_comp = np.bincount(label).argmax()
        cand = np.where(label == main_comp)[0]
        outlet = int(cand[np.nanargmin(np.where(np.isnan(z[cand]), np.inf, z[cand]))])
    print(f"Outlet at x={coords[outlet,0]:.0f}, y={coords[outlet,1]:.0f} ({crs}), "
          f"elevation {z[outlet]:.0f}")

    dist, down = orient_to_outlet(coords, edges, outlet)
    ok = np.isfinite(dist)
    if (~ok).any():
        print(f"Note: {(~ok).sum()} river vertices are not connected to the outlet "
              f"(drawn grey, no particles). Increase --snap if this looks wrong.")
    dmax = dist[ok].max()

    # upstream river length -> river width
    order = np.argsort(-np.where(ok, dist, -1))
    acc = np.zeros(len(coords))
    for v in order:
        if not ok[v] or v == outlet:
            continue
        acc[v] += dist[v] - dist[down[v]]
        acc[down[v]] += acc[v]
    seg_nodes = np.where(ok & (np.arange(len(coords)) != outlet))[0]
    segs = np.stack([coords[seg_nodes], coords[down[seg_nodes]]], axis=1)
    acc_frac = np.sqrt(acc[seg_nodes] / acc.max())

    # where each component's particles start (illustrative, see docstring)
    zr = pd.Series(z[ok]).rank(pct=True).to_numpy()   # elevation percentile
    nodes_ok = np.where(ok)[0]
    w_glacier = (zr >= 0.85).astype(float)
    if a.glaciers:
        gl = gpd.read_file(a.glaciers).to_crs(crs).union_all().buffer(spacing * 2)
        inside = gpd.GeoSeries(gpd.points_from_xy(*coords[nodes_ok].T), crs=crs).within(gl)
        if inside.any():
            w_glacier = inside.to_numpy().astype(float)
        else:
            print("Note: no river vertices inside --glaciers; using highest terrain.")
    weights = {
        "snowmelt": np.where(zr >= 0.4, zr ** 2, 0.0),
        "glacier_melt": w_glacier,
        "rainfall": np.ones_like(zr),
        "baseflow": np.ones_like(zr),
    }
    weights = {k: w / w.sum() for k, w in weights.items()}

    # ---- particle simulation (vectorised) -----------------------------------
    speed = dmax / a.travel_frames
    P = SimpleNamespace(node=np.empty(0, int), d=np.empty(0), c=np.empty(0, int))

    def rates_at(f):
        """Interpolated component runoff at animation frame f."""
        t = f / a.frames_per_step
        k = min(int(t), len(q) - 2)
        return q[k] + (q[k + 1] - q[k]) * min(t - k, 1.0)

    def step(rates):
        # release
        new_n, new_c = [], []
        for ci, comp in enumerate(COMPONENTS):
            n = rng.poisson(a.max_spawn * rates[ci] / q_ref)
            if n:
                new_n.append(rng.choice(nodes_ok, n, p=weights[comp]))
                new_c.append(np.full(n, ci))
        if new_n:
            nn = np.concatenate(new_n)
            P.node = np.r_[P.node, nn]
            P.d = np.r_[P.d, dist[nn]]
            P.c = np.r_[P.c, np.concatenate(new_c)]
        # move downstream
        P.d = P.d - speed * rng.uniform(0.85, 1.15, len(P.d))
        keep = P.d > 0
        P.node, P.d, P.c = P.node[keep], P.d[keep], P.c[keep]
        while True:
            m = dist[down[P.node]] > P.d
            if not m.any():
                break
            P.node[m] = down[P.node[m]]

    def positions():
        n0, n1 = P.node, down[P.node]
        span = np.maximum(dist[n0] - dist[n1], 1e-9)
        t = np.clip((dist[n0] - P.d) / span, 0, 1)[:, None]
        return coords[n0] * (1 - t) + coords[n1] * t

    # spin-up so the rivers are already flowing on the first frame
    for _ in range(a.travel_frames):
        step(q[0])

    # ---- figure --------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_3, "ytick.color": INK_3,
                         "axes.edgecolor": "#3a3a37"})
    fig = plt.figure(figsize=(12, 8.4), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[4.2, 1],
                          left=0.03, right=0.97, top=0.9, bottom=0.1,
                          wspace=0.04, hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])
    tl = fig.add_subplot(gs[1, :])
    for x in (ax, side):
        x.set_facecolor(BG)
        x.set_axis_off()
    tl.set_facecolor(BG)

    # hillshade, masked to the basin
    hs = hillshade(dem, dtrans)
    mask = features.geometry_mask([basin_geom], dem.shape, dtrans)
    hs = np.ma.array(hs, mask=mask | np.isnan(dem))
    h, w = dem.shape
    ext = (dtrans.c, dtrans.c + w * dtrans.a, dtrans.f + h * dtrans.e, dtrans.f)
    ax.imshow(hs, cmap="gray", vmin=-0.4, vmax=2.6, extent=ext, interpolation="bilinear")
    boundary.boundary.plot(ax=ax, color=INK_2, lw=1.0)
    minx, miny, maxx, maxy = basin_geom.bounds
    pad = 0.03 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")

    if (~ok).any():
        bad = [(coords[int(i)], coords[int(j)]) for i, j, _ in edges
               if not (ok[int(i)] and ok[int(j)])]
        ax.add_collection(LineCollection(bad, colors="#44443f", linewidths=0.4))
    rivers_lc = LineCollection(segs, colors=RIVER, linewidths=0.5 + 2.5 * acc_frac,
                               capstyle="round", zorder=2)
    ax.add_collection(rivers_lc)
    trail = [ax.scatter([], [], s=s_, linewidths=0, alpha=al, zorder=3)
             for s_, al in ((5, 0.18), (7, 0.35))]   # previous frames = motion trail
    sc = ax.scatter([], [], s=10, linewidths=0, zorder=3)
    hist = []
    ax.plot(*coords[outlet], marker="v", ms=9, color=INK, mec=BG, zorder=4)
    ax.annotate("Outlet", coords[outlet], xytext=(8, -12), textcoords="offset points",
                color=INK_2, fontsize=9)
    pcolors = np.array([matplotlib.colors.to_rgba(COLORS[c]) for c in COMPONENTS])

    fig.text(0.03, 0.955, f"Runoff flowing through {a.basin}", fontsize=15,
             weight="bold")
    date_txt = fig.text(0.03, 0.918, "", fontsize=12, color=INK_2)
    fig.text(0.03, 0.02,
             "Amounts: monthly runoff components at the basin outlet (model output). "
             "Particle start locations are illustrative (by elevation"
             + (" / glacier outlines" if a.glaciers else "") +
             "), not modelled per cell.\nValues are interpolated between months for "
             "smooth motion. Particle speed is not to scale. "
             "One particle ≈ a fixed share of flow; river width scales with total "
             "runoff and upstream length.", fontsize=8, color=INK_3)

    # side panel: legend with live values (identity never colour-only)
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    rows = []
    for k, comp in enumerate(COMPONENTS):
        y = 0.86 - k * 0.17
        side.scatter([0.06], [y], s=90, color=COLORS[comp])
        side.text(0.14, y + 0.025, LABELS[comp], fontsize=11, va="center")
        rows.append(side.text(0.14, y - 0.035, "", fontsize=10, color=INK_2,
                              va="center"))
    total_txt = side.text(0.06, 0.12, "", fontsize=11, color=INK)

    # timeline: stacked area of the whole period + cursor
    cum = np.cumsum(q, axis=1)
    prev = np.zeros(len(q))
    for ci, comp in enumerate(COMPONENTS):
        tl.fill_between(df.index, prev, cum[:, ci], color=COLORS[comp], alpha=0.85,
                        lw=0.4, edgecolor=BG)
        prev = cum[:, ci]
    tl.set_xlim(df.index[0], df.index[-1])
    tl.set_ylim(0, cum[:, -1].max() * 1.1)
    tl.set_ylabel(a.units, fontsize=9)
    for s in ("top", "right"):
        tl.spines[s].set_visible(False)
    tl.tick_params(labelsize=8)
    cursor = tl.axvline(df.index[0], color=INK, lw=1.2)

    n_frames = (len(q) - 1) * a.frames_per_step + 1
    date_fmt = "%B %Y" if monthly else "%d %b %Y"
    span = df.index[-1] - df.index[0]

    def update(f):
        r = rates_at(f)
        step(r)
        xy = positions()
        cols = pcolors[P.c] if len(P.c) else np.empty((0, 4))
        xy = xy if len(xy) else np.empty((0, 2))
        hist.append((xy, cols))
        del hist[:-3]
        for art, (hxy, hc) in zip(trail, hist[:-1][-2:]):
            art.set_offsets(hxy)
            art.set_color(hc)
        sc.set_offsets(xy)
        sc.set_color(cols)
        tot = r.sum()
        rivers_lc.set_linewidths((0.4 + 3.2 * acc_frac) * (0.35 + 0.65 * np.sqrt(tot / q_ref)))
        k = min(int(round(f / a.frames_per_step)), len(q) - 1)
        date_txt.set_text(df.index[k].strftime(date_fmt))
        for txt, v in zip(rows, r):
            share = 100 * v / tot if tot > 0 else 0
            txt.set_text(f"{v:,.1f} {a.units}  ·  {share:.0f}%")
        total_txt.set_text(f"Total  {tot:,.1f} {a.units}")
        cursor.set_xdata([df.index[0] + span * f / (n_frames - 1)] * 2)
        return sc, rivers_lc

    hold = a.fps
    frames = list(range(n_frames)) + [n_frames - 1] * hold
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / a.fps, blit=False)
    writer = (PillowWriter(fps=a.fps) if a.out.lower().endswith(".gif")
              else FFMpegWriter(fps=a.fps, bitrate=4000))
    print(f"Rendering {len(frames)} frames -> {a.out}")
    anim.save(a.out, writer=writer, dpi=a.dpi, savefig_kwargs={"facecolor": BG})
    print("Done.")


if __name__ == "__main__":
    main()
