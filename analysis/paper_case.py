"""Configuration objects for Newman et al. (2024)-style QPF verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


PAPER_THRESHOLDS_MM = [2.54, 12.7, 25.4, 38.1, 63.5, 88.9, 127.0]


@dataclass
class PaperModel:
    name: str
    run_root: Path


@dataclass
class PaperStormCase:
    source_path: Path
    storm_name: str
    models: list[PaperModel]
    best_track: Path
    domain: tuple
    inits: list[str] | None
    lead_hours: list[int]
    accumulation_hours: int
    thresholds_mm: list[float]
    headline_thresholds_mm: list[float]
    forecast_domain: str
    grid_res: float
    mask_radius_km: float
    out_dir: Path
    mrms_cache_dir: Path
    composite_lead_hour: int
    storm_relative_radius_rmw: float
    storm_relative_res_rmw: float
    radial_bin_rmw: float
    rmw_fallback_km: float
    object_threshold_mm: float
    object_smooth_cells: float
    object_min_area_km2: float
    object_init: str | None
    object_lead_hour: int
    bootstrap_replicates: int
    random_seed: int

    @property
    def case_slug(self):
        return self.source_path.stem


@dataclass
class PaperSuiteCase:
    source_path: Path
    label: str
    storm_paths: list[Path]
    out_dir: Path
    bootstrap_replicates: int
    random_seed: int

    @property
    def case_slug(self):
        return self.source_path.stem


def _models(value, path):
    if isinstance(value, dict):
        models = [PaperModel(str(name), Path(root)) for name, root in value.items()]
    elif isinstance(value, list):
        models = []
        for item in value:
            if not isinstance(item, dict) or "name" not in item or "run_root" not in item:
                raise ValueError(f"Each model in {path} needs name and run_root")
            models.append(PaperModel(str(item["name"]), Path(item["run_root"])))
    else:
        raise ValueError(f"'models' in {path} must be a mapping or list")
    if not models:
        raise ValueError(f"At least one model is required in {path}")
    if len({m.name for m in models}) != len(models):
        raise ValueError(f"Model names must be unique in {path}")
    return models


def load_paper_storm(path) -> PaperStormCase:
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in ("models", "best_track", "domain"):
        if key not in cfg:
            raise KeyError(f"'{key}' is required in paper storm YAML {path}")
    accumulation = int(cfg.get("accumulation_hours", 6))
    if accumulation <= 0:
        raise ValueError("accumulation_hours must be positive")
    lead_hours = [int(v) for v in cfg.get("lead_hours", range(accumulation, 127, 6))]
    if not lead_hours or any(v < accumulation for v in lead_hours):
        raise ValueError("lead_hours must contain endpoints >= accumulation_hours")
    forecast_domain = str(cfg.get("forecast_domain", "parent")).lower()
    if forecast_domain not in {"parent", "nest"}:
        raise ValueError("forecast_domain must be 'parent' or 'nest'")
    thresholds = [float(v) for v in cfg.get("thresholds_mm", PAPER_THRESHOLDS_MM)]
    headlines = [float(v) for v in cfg.get(
        "headline_thresholds_mm", [thresholds[0], thresholds[4]])]
    composite_lead = int(cfg.get("composite_lead_hour", 12))
    if composite_lead < accumulation:
        raise ValueError("composite_lead_hour must cover the accumulation window")
    return PaperStormCase(
        source_path=path,
        storm_name=str(cfg.get("storm_name", path.stem)),
        models=_models(cfg["models"], path),
        best_track=Path(cfg["best_track"]),
        domain=tuple(float(v) for v in cfg["domain"]),
        inits=[str(v) for v in cfg["inits"]] if cfg.get("inits") else None,
        lead_hours=sorted(set(lead_hours)),
        accumulation_hours=accumulation,
        thresholds_mm=thresholds,
        headline_thresholds_mm=headlines,
        forecast_domain=forecast_domain,
        grid_res=float(cfg.get("grid_res", 0.1)),
        mask_radius_km=float(cfg.get("mask_radius_km", 600.0)),
        out_dir=Path(cfg.get("out_dir", f"analysis/output/{path.stem}")),
        mrms_cache_dir=Path(cfg.get("mrms_cache_dir", "/tmp/mrms_cache")),
        composite_lead_hour=composite_lead,
        storm_relative_radius_rmw=float(cfg.get("storm_relative_radius_rmw", 6.0)),
        storm_relative_res_rmw=float(cfg.get("storm_relative_res_rmw", 0.2)),
        radial_bin_rmw=float(cfg.get("radial_bin_rmw", 0.4)),
        rmw_fallback_km=float(cfg.get("rmw_fallback_km", 50.0)),
        object_threshold_mm=float(cfg.get("object_threshold_mm", 10.0)),
        object_smooth_cells=float(cfg.get("object_smooth_cells", 1.0)),
        object_min_area_km2=float(cfg.get("object_min_area_km2", 500.0)),
        object_init=str(cfg["object_init"]) if cfg.get("object_init") else None,
        object_lead_hour=int(cfg.get("object_lead_hour", composite_lead)),
        bootstrap_replicates=int(cfg.get("bootstrap_replicates", 500)),
        random_seed=int(cfg.get("random_seed", 42)),
    )


def load_paper_config(path):
    """Return PaperStormCase or PaperSuiteCase based on top-level keys."""
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    if "storms" not in cfg:
        return load_paper_storm(path)
    storms = cfg["storms"]
    if not isinstance(storms, list) or not storms:
        raise ValueError(f"'storms' must be a non-empty list in {path}")
    # Paths follow the same repo-root convention as the rest of the framework.
    return PaperSuiteCase(
        source_path=path,
        label=str(cfg.get("label", path.stem)),
        storm_paths=[Path(p) for p in storms],
        out_dir=Path(cfg.get("out_dir", f"analysis/output/{path.stem}")),
        bootstrap_replicates=int(cfg.get("bootstrap_replicates", 1000)),
        random_seed=int(cfg.get("random_seed", 42)),
    )
