# ETS for HAFS QPF vs MRMS & Stage IV — Design

**Date:** 2026-06-18
**Status:** Approved (design), pending implementation plan
**Script:** `analysis/ets_full.py` (new)

## Goal

Compute the Equitable Threat Score (ETS / Gilbert Skill Score), plus frequency
bias, POD, FAR, and CSI, for **two HAFS-A forecast fields** verified against
**two observed QPE references** over the Hurricane Helene rainfall swath:

| Forecast | Observation |
|---|---|
| HAFS-A parent domain (fixed 6-km, geographically honest cumulative APCP) | MRMS MultiSensor QPE (Pass2) |
| HAFS-A moving 2-km nest (running-max storm total) | NCEP Stage IV QPE (CONUS, 24h) |

Both forecasts are scored against **both** observations → 4 ETS-vs-threshold
curves on one combined figure + one combined CSV.

The existing `analysis/ets_score.py` (nest vs MRMS only) is **left untouched**.
All GRIB2 / MRMS / Stage IV plumbing is imported from `qpf_full_run.py` and
`parent_qpf.py` — no logic is duplicated.

## Architecture & data flow

1. **Fixed verification grid** — reuse `FIXED_DOMAIN` + `GRID_RES` from
   `qpf_full_run` (the same mesh `ets_score.py` builds). All fields are brought
   onto this grid before scoring.
2. **HAFS nest total** — `hafs_event_total()` (from `qpf_full_run`), already
   produced on the fixed grid.
3. **HAFS parent total** — `default_parent_path()` + `read_hafs_tp_records()` +
   `pick_cumulative_record()` (from `parent_qpf`). The parent record is on the
   fixed 6-km parent grid; regrid it onto the verification grid.
4. **MRMS total** — accumulate MRMS 1H QPE over the forecast window (same as
   `ets_score.build_mrms_total`); MRMS is a regular 1-D lat/lon grid, regridded
   with `regrid_mrms_to_fixed` (bilinear `RegularGridInterpolator`).
5. **Stage IV total** — reuse `stage4_total()` from `parent_qpf` (sum every
   daily 24h CONUS file the forecast window touches; files valid 12Z→12Z).
   Stage IV is a **2-D curvilinear** grid, so add a `scipy.interpolate.griddata`
   regridder (method="linear", NaN fill) onto the fixed grid — distinct from the
   1-D MRMS interpolator. Regrid the **unmasked** Stage IV total so genuine
   zero-rain obs are preserved and only out-of-domain points become NaN.

## Masking & valid points

- **TC swath**: union of `TC_MASK_RADIUS_KM` (500 km) circles along the Helene
  best track for hours 0..max_fhour — the verification radius, **not**
  `parent_qpf.MASK_RADIUS_KM` (750 km display radius), so the score is fair and
  consistent with `ets_score.py`.
- **Per-observation validity**: verification points = `swath & isfinite(obs)`.
  For Stage IV (CONUS-only), points over the Gulf / open ocean fall outside its
  grid → NaN → automatically excluded, so missing coverage doesn't pollute the
  contingency table.
- Each forecast/obs pair is scored only where **both** that forecast and that
  observation are valid (finite within the swath).

## Output

- **Figure** `analysis/output/../ets_full_helene.png` (sibling of existing
  `ets_helene_hfsa.png`, i.e. `OUT_DIR.parent`): 4 ETS-vs-threshold curves
  (parent/nest × MRMS/StageIV). Log-scaled threshold x-axis, zero-skill line at
  ETS=0. Color encodes observation (MRMS vs Stage IV), linestyle encodes
  forecast (parent vs nest). Frequency bias omitted from the main plot to avoid
  clutter (retained in the CSV).
- **CSV** `ets_full_helene.csv`: one row per (forecast, observation, threshold)
  with columns `forecast, observation, threshold, a, b, c, d, ets, bias, pod,
  far, csi`.
- Thresholds: reuse `THRESHOLDS_MM` from `ets_score`.
- A Stage IV temporal-alignment caveat (24h 12Z→12Z files summed over touched
  days vs the 00Z-init 0→Nh window) is printed to stdout and added as a figure
  footnote.

## Reused 2x2 contingency math

```
ETS  = (a - a_ref) / (a + b + c - a_ref),   a_ref = (a+b)(a+c)/n
bias = (a+b)/(a+c)   POD = a/(a+c)   FAR = b/(a+b)   CSI = a/(a+b+c)
```
where a=hits, b=false alarms, c=misses, d=correct negatives. Reuse
`ets_score.contingency_scores` (import it) rather than reimplementing.

## Testing / verification (on Hercules)

Run `python analysis/ets_full.py` and confirm:
1. **Regression**: the nest-vs-MRMS curve matches the existing `ets_score.py`
   output (same plumbing, same grid → same numbers).
2. **CONUS clipping**: Stage IV verification-point count is sensibly smaller
   than MRMS (Gulf/ocean dropped).
3. **Sanity**: all ETS values fall in [-1/3, 1]; frequency bias is positive.

## Caveats

- Stage IV windows (24h, 12Z→12Z) do not align exactly with the 00Z-init
  0→Nh forecast window; the touched-days sum is an approximation (matches
  `parent_qpf.py`). Documented in output, not corrected.
- Stage IV is CONUS-only — no verification over Helene's Gulf/Caribbean rain.
