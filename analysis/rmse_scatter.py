"""
Storm-total continuous verification for HAFS QPF (parent domain + 2-km
nest) against MRMS QPE and NCEP Stage IV QPE over the TC rainfall swath.

Computes RMSE / MAE / bias / Pearson r on the event-total rainfall over
the same valid-point footprint the ETS uses, and renders one figure of
forecast-vs-observed hexbin scatter panels (rows = parent/nest, cols =
MRMS/Stage IV) with a 1:1 line and the scores annotated per panel.

Usage (on Hercules):
    python analysis/run.py storms/<case>.yaml rmse
"""

import sys
import csv
from pathlib import Path

# Make sibling analysis modules importable no matter the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ets_full import build_verification_fields, field_pairs, stage4_caveat
from skill_metrics import continuous_scores
from hafs_case import from_yaml
from plot_units import inches, miles


def valid_points(fcst_grid, obs_grid, swath):
    """1-D (fcst, obs) arrays over the swath's valid points, zero-filled.

    Mirrors ets_full.score_pair's selection (swath & finite obs & finite
    fcst) so the continuous scores describe the identical footprint as
    the categorical ones.
    """
    valid = swath & np.isfinite(obs_grid) & np.isfinite(fcst_grid)
    fcst = np.nan_to_num(fcst_grid[valid], nan=0.0)
    obs = np.nan_to_num(obs_grid[valid], nan=0.0)
    return fcst, obs


def plot_scatter(case, results, max_fhour, out_path, caveat=""):
    """results: list of dicts {forecast, observation, fcst, obs, scores}.

    One hexbin panel per pair; panel grid is forecasts x observations in
    the order the results were computed.
    """
    fcst_names = list(dict.fromkeys(r["forecast"] for r in results))
    obs_names = list(dict.fromkeys(r["observation"] for r in results))
    nrows, ncols = len(fcst_names), len(obs_names)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols, 4.8 * nrows),
                             squeeze=False)

    # One shared, equal axis range (inches) across all panels.
    lim = 1.0
    for r in results:
        if r["fcst"].size:
            lim = max(lim, float(inches(r["fcst"].max())),
                      float(inches(r["obs"].max())))
    lim = float(np.ceil(lim))

    by_pair = {(r["forecast"], r["observation"]): r for r in results}
    for i, fname in enumerate(fcst_names):
        for j, oname in enumerate(obs_names):
            ax = axes[i][j]
            res = by_pair[(fname, oname)]
            s = res["scores"]
            if s["n"] == 0:
                ax.text(0.5, 0.5, "no valid points", ha="center",
                        va="center", transform=ax.transAxes, color="#777")
            else:
                hb = ax.hexbin(inches(res["obs"]), inches(res["fcst"]), gridsize=60,
                               extent=(0, lim, 0, lim), norm=LogNorm(),
                               cmap="viridis", mincnt=1)
                fig.colorbar(hb, ax=ax, label="grid points")
                box = (f"RMSE {inches(s['rmse']):.2f} in\n"
                       f"MAE  {inches(s['mae']):.2f} in\n"
                       f"bias {inches(s['bias']):+.2f} in\n"
                       f"r    {s['r']:.2f}\n"
                       f"n    {s['n']:,}")
                ax.text(0.03, 0.97, box, transform=ax.transAxes, va="top",
                        fontsize=8.5, family="monospace",
                        bbox=dict(boxstyle="round", fc="white",
                                  ec="#999", alpha=0.85))
            ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{oname} observed (inches)")
            ax.set_ylabel(f"{fname} forecast (inches)")
            ax.set_title(f"{fname} vs {oname}", fontsize=10)

    fig.suptitle(
        f"{case.storm_name} — {case.model_label} storm-total QPF vs observed\n"
        f"0–{max_fhour}h | init {case.init_dt:%Y-%m-%d %HZ} | "
        f"TC swath ≤{miles(case.mask_radius_km):.0f} miles", fontsize=12)
    if caveat:
        fig.text(0.5, -0.01, caveat, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compute_rmse(case, fields=None):
    if fields is None:
        fields = build_verification_fields(case)
    max_fhour = fields["max_fhour"]
    swath = fields["swath"]
    forecasts, observations = field_pairs(fields)

    results = []
    print("\n" + "=" * 64)
    print(f"{'forecast':>8} {'obs':>9} {'n':>9} {'RMSE':>8} {'MAE':>8} "
          f"{'bias':>8} {'r':>6}")
    for fname, fgrid in forecasts:
        for oname, ogrid in observations:
            fcst, obs = valid_points(fgrid, ogrid, swath)
            s = continuous_scores(fcst, obs)
            results.append(dict(forecast=fname, observation=oname,
                                fcst=fcst, obs=obs, scores=s))
            print(f"{fname:>8} {oname:>9} {s['n']:>9,} {s['rmse']:>8.2f} "
                  f"{s['mae']:>8.2f} {s['bias']:>+8.2f} {s['r']:>6.2f}")
    print("=" * 64)

    case.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = case.out_dir / f"rmse_{case.output_slug}.csv"
    out_png = case.out_dir / f"rmse_scatter_{case.output_slug}.png"

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["forecast", "observation", "n",
                                           "rmse", "mae", "bias", "r"])
        w.writeheader()
        for res in results:
            w.writerow({"forecast": res["forecast"],
                        "observation": res["observation"], **res["scores"]})
    print(f"\nSaved table: {out_csv}")

    caveat = stage4_caveat(fields)
    print(caveat)
    plot_scatter(case, results, max_fhour, out_png, caveat=caveat)
    print(f"Saved plot : {out_png}")


if __name__ == "__main__":
    compute_rmse(from_yaml(sys.argv[1]))
