"""Per-cycle machine-learning features for precipitation verification."""

import csv
import os
import re
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = [
    "storm", "model", "init", "lead_hours_to_landfall",
    "vmax0_kt", "bdeck_vmax_kt", "bdeck_mslp_hpa",
    "translation_speed_kt", "mean_track_err_km",
    "pwat_mean_500km", "shear_850_200_kt", "rh700_mean_500km",
    "mslp_min_hpa",
]

TARGET_COLUMNS = [
    "ets_headline", "fss_headline", "rmse", "bias_mm",
    "obj_centroid_err_km", "ets_shifted",
]

_WANTED = [
    ("pwat", None), ("u", 850), ("v", 850), ("u", 200),
    ("v", 200), ("r", 700), ("mslet", None),
]


def _haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    dlat = np.radians(np.asarray(lat2) - lat1)
    dlon = np.radians(np.asarray(lon2) - lon1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2.0) ** 2)
    return radius_km * 2.0 * np.arcsin(np.sqrt(a))


def disc_mean(lat2d, lon2d, data, clat, clon, radius_km):
    """NaN-aware mean inside a great-circle disc."""
    distances = _haversine_km(clat, clon, lat2d, lon2d)
    values = np.asarray(data, dtype=float)
    mask = (distances <= radius_km) & np.isfinite(values)
    return float(np.nanmean(values[mask])) if np.any(mask) else np.nan


def translation_speed_kt(bdeck_full, init_dt):
    """Centered 12-hour best-track translation speed in knots."""
    if not bdeck_full:
        return np.nan
    from track_skill import bdeck_state

    before = bdeck_state(bdeck_full, init_dt - timedelta(hours=6))
    after = bdeck_state(bdeck_full, init_dt + timedelta(hours=6))
    if any(before[key] is None or after[key] is None
           for key in ("lat", "lon")):
        return np.nan
    distance_km = _haversine_km(
        before["lat"], before["lon"], after["lat"], after["lon"])
    return float(distance_km / 12.0 / 1.852)


def _slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _cycle_value(cycle, key, default=None):
    if isinstance(cycle, dict):
        return cycle.get(key, default)
    return getattr(cycle, key, default)


def _feature_error(name, exc):
    print(f"  feature {name}: {exc}")


def _crop_record(record, clat, clon):
    from hafs_common import crop_to_domain

    lats = np.asarray(record["lats"])
    lons = np.asarray(record["lons"])
    data = np.asarray(record["data"])
    if lats.ndim == 2 and lons.ndim == 2:
        lat1d, lon1d = lats[:, 0], lons[0, :]
        clats, clons, cdata = crop_to_domain(
            lat1d, lon1d, data, clat - 6.0, clat + 6.0,
            clon - 6.0, clon + 6.0)
        clon2d, clat2d = np.meshgrid(clons, clats)
        return clat2d, clon2d, cdata
    return crop_to_domain(lats, lons, data, clat - 6.0, clat + 6.0,
                          clon - 6.0, clon + 6.0)


def _disc_record_mean(record, center):
    lats, lons, data = _crop_record(record, *center)
    return disc_mean(lats, lons, data, *center, 500.0)


def _storm_center(case, init_dt, bdeck_full):
    if bdeck_full:
        from track_skill import bdeck_state

        state = bdeck_state(bdeck_full, init_dt)
        if state["lat"] is not None and state["lon"] is not None:
            return state["lat"], state["lon"]
    fixes = getattr(case, "track_fixes", None) or []
    for valid, lat, lon, *_ in fixes:
        if valid == init_dt:
            return lat, lon
    raise ValueError("storm center unavailable at initialization")


def _env_records(case):
    from hafs_common import read_hafs_env_records
    from parent_qpf import parent_path_at_fhour

    path = parent_path_at_fhour(case, 0)
    if path is None:
        path = parent_path_at_fhour(case, 6)
    if path is None:
        raise FileNotFoundError("no parent file at f000 or f006")
    return read_hafs_env_records(path, _WANTED)


def extract_cycle_features(ccase, case, cycle, summary_row, bdeck_full):
    """Build one robust feature/target row for a surviving cycle."""
    nan = np.nan
    init_dt = getattr(case, "init_dt", _cycle_value(cycle, "init_dt"))
    init = getattr(case, "init_str", _cycle_value(cycle, "init"))
    if init is None and init_dt is not None:
        init = init_dt.strftime("%Y%m%d%H")
    row = {key: nan for key in FEATURE_COLUMNS + TARGET_COLUMNS}
    row.update({
        "storm": _slugify(getattr(ccase, "storm_name", "")),
        "model": getattr(ccase, "model_label", ""),
        "init": init,
        "lead_hours_to_landfall": summary_row.get(
            "lead_hours_to_landfall", nan),
    })
    for key in TARGET_COLUMNS:
        row[key] = summary_row.get(key, nan)

    try:
        fixes = getattr(case, "track_fixes", None) or []
        fix0 = next(fix for fix in fixes if fix[0] == init_dt)
        if len(fix0) < 4 or fix0[3] is None:
            raise ValueError("TAU 0 vmax unavailable")
        row["vmax0_kt"] = float(fix0[3])
    except Exception as exc:
        _feature_error("vmax0_kt", exc)

    for output_key, state_key in (("bdeck_vmax_kt", "vmax"),
                                  ("bdeck_mslp_hpa", "mslp")):
        try:
            if not bdeck_full:
                raise ValueError("best track unavailable")
            from track_skill import bdeck_state

            value = bdeck_state(bdeck_full, init_dt)[state_key]
            if value is None:
                raise ValueError("best-track value unavailable")
            row[output_key] = float(value)
        except Exception as exc:
            _feature_error(output_key, exc)

    try:
        speed = translation_speed_kt(bdeck_full, init_dt)
        if not np.isfinite(speed):
            raise ValueError("best-track motion unavailable")
        row["translation_speed_kt"] = speed
    except Exception as exc:
        _feature_error("translation_speed_kt", exc)

    try:
        value = summary_row.get("mean_track_err_km", nan)
        if value is None or not np.isfinite(float(value)):
            raise ValueError("cycle track summary unavailable")
        row["mean_track_err_km"] = float(value)
    except Exception as exc:
        _feature_error("mean_track_err_km", exc)

    try:
        records, env_error = _env_records(case), None
    except Exception as exc:
        records, env_error = {}, exc

    def record(key):
        if env_error is not None:
            raise env_error
        if key not in records:
            raise KeyError(f"GRIB field {key} unavailable")
        return records[key]

    try:
        center = _storm_center(case, init_dt, bdeck_full)
        row["pwat_mean_500km"] = _disc_record_mean(
            record(("pwat", None)), center)
    except Exception as exc:
        _feature_error("pwat_mean_500km", exc)

    try:
        center = _storm_center(case, init_dt, bdeck_full)
        u850 = _disc_record_mean(record(("u", 850)), center)
        v850 = _disc_record_mean(record(("v", 850)), center)
        u200 = _disc_record_mean(record(("u", 200)), center)
        v200 = _disc_record_mean(record(("v", 200)), center)
        row["shear_850_200_kt"] = float(
            np.hypot(u200 - u850, v200 - v850) * 1.94384449)
    except Exception as exc:
        _feature_error("shear_850_200_kt", exc)

    try:
        center = _storm_center(case, init_dt, bdeck_full)
        row["rh700_mean_500km"] = _disc_record_mean(
            record(("r", 700)), center)
    except Exception as exc:
        _feature_error("rh700_mean_500km", exc)

    try:
        center = _storm_center(case, init_dt, bdeck_full)
        mslp = record(("mslet", None))
        lats, lons, data = _crop_record(mslp, *center)
        mask = _haversine_km(*center, lats, lons) <= 500.0
        values = np.asarray(data, dtype=float)[mask]
        if not np.any(np.isfinite(values)):
            raise ValueError("no finite MSLP points within 500 km")
        minimum = float(np.nanmin(values))
        units = str(mslp.get("units") or "").lower()
        if units in ("pa", "pascal", "pascals") or minimum > 2000.0:
            minimum /= 100.0
        row["mslp_min_hpa"] = minimum
    except Exception as exc:
        _feature_error("mslp_min_hpa", exc)
    return row


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def append_features(rows, csv_path):
    """Upsert feature rows by identity and atomically rewrite the CSV."""
    rows = list(rows)
    csv_path = Path(csv_path)
    existing = []
    header = []
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            existing = list(reader)
    for row in rows:
        for key in row:
            if key not in header:
                header.append(key)
    if not header:
        header = list(FEATURE_COLUMNS + TARGET_COLUMNS)
    incoming_keys = {
        (str(row.get("storm", "")), str(row.get("model", "")),
         str(row.get("init", ""))) for row in rows
    }
    existing = [row for row in existing if
                (row.get("storm", ""), row.get("model", ""),
                 row.get("init", "")) not in incoming_keys]
    output = existing + rows
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", newline="", dir=csv_path.parent,
                prefix=f".{csv_path.name}.", delete=False) as fh:
            temp_name = fh.name
            writer = csv.DictWriter(fh, fieldnames=header,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in output:
                writer.writerow({key: _csv_value(row.get(key))
                                 for key in header})
        os.replace(temp_name, csv_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
