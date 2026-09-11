"""Data and experiment helpers for ``ground_experiments.ipynb``.

The walkthrough imports this file so that its cells stay short. All processing is
restricted to an unsampled spatial crop of the existing development sample. No
function reads a raw survey LAS. Saved labels are joined using complete LAS point
records, not nearest neighbours; no labels are inferred for unmatched points.

Distances below are in unchanged *coordinate units*. The LAS headers say US
survey feet while export sidecars say meters. This module does not resolve that
disagreement or convert coordinates. LAS classification fields remain unchanged;
ground membership is represented by masks and separate output files.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import laspy
import numpy as np


CHUNK_POINTS = 200_000
MAX_PROCESSING_POINTS = 250_000
MAX_CONTEXT_AREA = 250_000.0
MAX_CSF_NODES = 500_000
UNITS = "unresolved coordinate units (LAS: US survey foot; sidecar: meter)"
HISTORICAL_RUNTIME_SCOPE = (
    "historical pipeline total for BOTH sides, including input/output work; "
    "not this ROI's filter time"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (Path, datetime)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _header_info(path: Path) -> dict:
    with laspy.open(path) as reader:
        header = reader.header
        try:
            crs = header.parse_crs()
            axis_units = [axis.unit_name for axis in crs.axis_info] if crs else []
        except Exception as error:
            axis_units = [f"CRS could not be parsed: {error}"]
        return {
            "point_count": int(header.point_count), "min": header.mins.tolist(),
            "max": header.maxs.tolist(), "scales": header.scales.tolist(),
            "offsets": header.offsets.tolist(), "point_format": header.point_format.id,
            "axis_units": axis_units,
        }


def _parameters(manifest: dict) -> dict:
    params = manifest["parameters"]
    if manifest["method"] == "baseline":
        return {"cell_size": params["ground_cell_size"], "tolerance": params["ground_tolerance"]}
    csf = params["csf"]
    return {
        "cloth_resolution": csf["cloth_resolution_m"],
        "class_threshold": csf["class_threshold_m"],
        **{key: csf[key] for key in ("rigidness", "time_step", "iterations", "slope_smooth")},
    }


def load_catalog(project_dir: str | Path) -> dict:
    """Read existing manifests and LAS headers, without processing any points.

    ``rows`` is ready for a DataFrame. ``errors`` reports absent/broken saved
    outputs; loading that result later gives an explicit error. Historical sample
    links are resolved against their manifest folder. The archived CSF link is
    stale, so its regional labels must be verified against the cached sample.
    """
    root = Path(project_dir).resolve()
    input_path = root / "outputs/run1_inputs/run1_inputs_manifest.json"
    if not input_path.is_file():
        raise FileNotFoundError(f"Required development input manifest is missing: {input_path}")
    input_manifest = _read_json(input_path)
    catalog = {"root": root, "input_manifest": input_manifest,
               "input_manifest_path": input_path, "samples": {}, "runs": {},
               "rows": [], "errors": [], "coordinate_units": UNITS}
    for side in ("left", "right"):
        path = root / f"outputs/run1_inputs/run1_{side}_sample.las"
        sample = {"path": path, "source_info": input_manifest["sources"][side]}
        try:
            sample["header"] = _header_info(path)
        except Exception as error:
            sample["error"] = str(error)
            catalog["errors"].append(f"{path}: {error}")
        catalog["samples"][side] = sample
    specs = {
        "baseline": ("Baseline (saved)", "outputs/experiments/baseline/baseline_manifest.json"),
        "csf_previous": ("CSF previous (saved)", "outputs/experiments/csf/previous_2m_rigidness2/csf_manifest.json"),
        "csf_current": ("CSF current (saved)", "outputs/experiments/csf/csf_manifest.json"),
    }
    for key, (name, relative) in specs.items():
        path = root / relative
        run = {"name": name, "path": path, "manifest_path": path, "files": {}}
        catalog["runs"][key] = run
        try:
            manifest = _read_json(path)
            run["manifest"] = manifest
            run["parameters"] = _parameters(manifest)
        except Exception as error:
            run["error"] = str(error)
            catalog["errors"].append(f"{path}: {error}")
            continue
        for side in ("left", "right"):
            source = manifest["sources"][side]
            files = {label: (path.parent / rel).resolve()
                     for label, rel in source["output_files"].items()}
            run["files"][side] = files
            count_by_label, errors = {}, []
            for label in ("ground", "non_ground"):
                try:
                    count_by_label[label] = _header_info(files[label])["point_count"]
                except Exception as error:
                    errors.append(f"{label}: {error}")
                    catalog["errors"].append(f"{files.get(label)}: {error}")
            cached = catalog["samples"][side]
            fields = ("source_path", "source_point_count", "sample_point_count", "selection_probability", "seed")
            same_lineage = all(source.get(f) == cached["source_info"].get(f) for f in fields)
            same_roi = all(manifest["shared_roi"][f] == input_manifest["shared_roi"][f] for f in ("min", "max"))
            run.setdefault("lineage_matches", {})[side] = same_lineage and same_roi
            catalog["rows"].append({
                "saved_run": name, "side": side,
                "sample_points_manifest": source["sample_point_count"],
                "cached_sample_points_header": cached.get("header", {}).get("point_count"),
                "ground_points_header": count_by_label.get("ground"),
                "non_ground_points_header": count_by_label.get("non_ground"),
                "parameters_coordinate_units": run["parameters"].copy(),
                "runtime_seconds": manifest.get("runtime_seconds"),
                "runtime_scope": HISTORICAL_RUNTIME_SCOPE,
                "sample_lineage": "same manifest settings; regional record check still required" if same_lineage and same_roi else "DIFFERENT manifest source/settings",
                "sample_link": "present" if files.get("sample", Path()).is_file() else "stale/missing; cached sample checked when loading region",
                "status": "; ".join(errors) or "saved partitions available",
                "coordinate_units": UNITS,
            })
    return catalog


def _sample(catalog: dict, side: str) -> dict:
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'; raw paths are not supported")
    sample = catalog["samples"][side]
    if "error" in sample or not sample["path"].is_file():
        raise FileNotFoundError(f"Required cached processing sample is missing/unreadable: {sample['path']}")
    return sample


def _xyz(records: np.ndarray, header: laspy.LasHeader) -> np.ndarray:
    return np.column_stack([records[key].astype(np.float64) * header.scales[i] + header.offsets[i]
                            for i, key in enumerate(("X", "Y", "Z"))])


def _xy_mask(xyz: np.ndarray, bounds: list | np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax))


def load_overview(catalog: dict, side: str, max_display_points: int = 40_000) -> dict:
    """Read a deterministic, evenly spaced display subset of the cached sample.

    This is a navigation aid only; ``load_region`` independently rereads every
    processing point inside the selected context. XYZ coordinates are absolute;
    subtract ``origin`` for convenient consistent local plotting coordinates.
    """
    sample = _sample(catalog, side)
    if not 1 <= max_display_points <= 100_000:
        raise ValueError("max_display_points must be between 1 and 100,000")
    chunks = []
    with laspy.open(sample["path"]) as reader:
        count = int(reader.header.point_count)
        indices = np.linspace(0, count - 1, min(max_display_points, count), dtype=np.int64)
        offset = 0
        for points in reader.chunk_iterator(CHUNK_POINTS):
            left, right = np.searchsorted(indices, [offset, offset + len(points)])
            if right > left:
                chunks.append(_xyz(points.array[indices[left:right] - offset], reader.header))
            offset += len(points)
    return {"xyz": np.concatenate(chunks) if chunks else np.empty((0, 3)),
            "origin": np.asarray(catalog["input_manifest"]["shared_roi"]["min"], dtype=float),
            "source_path": sample["path"], "source_count": count, "side": side}


def _record_fingerprint(records: np.ndarray, header: laspy.LasHeader,
                        roi_xy: list, context_roi_xy: list, side: str) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"scales": header.scales.tolist(), "offsets": header.offsets.tolist(),
                              "dtype": records.dtype.descr, "roi_xy": roi_xy,
                              "context_roi_xy": context_roi_xy, "side": side}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(records).tobytes())
    return digest.hexdigest()


def load_region(catalog: dict, side: str, roi_xy: list, padding: float = 10.0,
                max_processing_points: int = MAX_PROCESSING_POINTS) -> dict:
    """Load every cached sample record in an absolute XY rectangle plus halo.

    ``roi_xy=[xmin,xmax,ymin,ymax]`` is the evaluation rectangle. Filtering uses
    the larger ``context_roi_xy``; counts and plots use ``evaluation_mask``.
    No additional processing sampling is performed. Limits prevent accidentally
    loading the whole corridor; reduce the region when a limit is reached.
    """
    start = time.perf_counter()
    sample = _sample(catalog, side)
    roi = np.asarray(roi_xy, dtype=float)
    if roi.shape != (4,) or not np.all(np.isfinite(roi)) or roi[1] <= roi[0] or roi[3] <= roi[2]:
        raise ValueError("roi_xy must contain finite [xmin, xmax, ymin, ymax] with positive spans")
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    if not 1 <= max_processing_points <= MAX_PROCESSING_POINTS:
        raise ValueError(f"max_processing_points must be between 1 and {MAX_PROCESSING_POINTS:,}")
    context = roi + np.array([-padding, padding, -padding, padding])
    if (context[1] - context[0]) * (context[3] - context[2]) > MAX_CONTEXT_AREA:
        raise ValueError(f"Context area exceeds {MAX_CONTEXT_AREA:,.0f} square coordinate units; choose a smaller region")
    parts, count = [], 0
    with laspy.open(sample["path"]) as reader:
        header = reader.header.copy()
        for points in reader.chunk_iterator(CHUNK_POINTS):
            selected = points.array[_xy_mask(_xyz(points.array, header), context)]
            count += len(selected)
            if count > max_processing_points:
                raise ValueError(f"Region plus padding exceeds {max_processing_points:,} processing points; shrink it (no points were dropped)")
            if len(selected):
                parts.append(selected.copy())
    records = np.concatenate(parts) if parts else np.empty(0, dtype=header.point_format.dtype())
    xyz = _xyz(records, header)
    evaluation_mask = _xy_mask(xyz, roi)
    if not np.any(evaluation_mask):
        raise ValueError("The evaluation region contains no cached sample points. Select a populated region from the overview.")
    stat = sample["path"].stat()
    return {"records": records, "header": header, "xyz": xyz, "evaluation_mask": evaluation_mask,
            "origin": np.asarray(catalog["input_manifest"]["shared_roi"]["min"], dtype=float),
            "roi_xy": roi.tolist(), "context_roi_xy": context.tolist(), "padding": float(padding),
            "side": side, "source_path": sample["path"], "source_info": sample["source_info"],
            "source_file_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "input_manifest": catalog["input_manifest"],
            "fingerprint": _record_fingerprint(records, header, roi.tolist(), context.tolist(), side),
            "load_runtime_seconds": time.perf_counter() - start}


def _counts(region: dict, ground: np.ndarray) -> dict:
    core = region["evaluation_mask"]
    return {"input": int(core.sum()), "ground": int((ground & core).sum()),
            "non_ground": int((~ground & core).sum()), "context_input": len(ground),
            "context_ground": int(ground.sum()), "context_non_ground": int((~ground).sum())}


def load_saved_result(catalog: dict, region: dict, run_key: str) -> dict:
    """Join a saved partition to the selected sample using exact record multisets.

    Both saved ground and non-ground files are cropped to the same context. Each
    selected input record must occur exactly once in their union. Duplicate point
    records are counted individually; contradictory labels, missing records,
    additional records, or different coordinate encodings raise an error. This
    verifies the *regional* sample identity, not the entire historical survey.
    """
    start = time.perf_counter()
    run = catalog["runs"][run_key]
    if "error" in run:
        raise FileNotFoundError(f"Cannot load {run_key}: {run['error']}")
    records, header = region["records"], region["header"]
    pending = defaultdict(list)
    for i, record in enumerate(records):
        pending[record.tobytes()].append(i)
    seen_labels, coverage = {}, np.zeros(len(records), dtype=bool)
    ground = np.zeros(len(records), dtype=bool)
    files = run["files"][region["side"]]
    for label, is_ground in (("ground", True), ("non_ground", False)):
        path = files[label]
        if not path.is_file():
            raise FileNotFoundError(f"Saved {label} LAS is missing: {path}")
        with laspy.open(path) as reader:
            if (reader.header.point_format.dtype() != records.dtype or
                    not np.array_equal(reader.header.scales, header.scales) or
                    not np.array_equal(reader.header.offsets, header.offsets)):
                raise ValueError(f"{path}: coordinate encoding or point schema differs from the selected source")
            for points in reader.chunk_iterator(CHUNK_POINTS):
                cropped = points.array[_xy_mask(_xyz(points.array, reader.header), region["context_roi_xy"])]
                for record in cropped:
                    key = record.tobytes()
                    if key in seen_labels and seen_labels[key] != is_ground:
                        raise ValueError(f"{run_key}: identical point record appears in both ground and non-ground partitions")
                    seen_labels[key] = is_ground
                    if key not in pending or not pending[key]:
                        raise ValueError(f"{run_key}: saved context includes a record/multiplicity absent from the chosen sample; sources differ")
                    index = pending[key].pop()
                    coverage[index] = True
                    ground[index] = is_ground
    if not coverage.all():
        raise ValueError(f"{run_key}: {int((~coverage).sum()):,} selected context records are missing from saved partitions; cannot compare as identical inputs")
    sample_link = files.get("sample")
    note = "Exact full-record multiset match for every point in this region plus padding."
    if sample_link is None or not sample_link.is_file():
        note += " Historical sample link is stale; verified against the present cached sample."
    if not run["lineage_matches"][region["side"]]:
        note += " Whole-sample manifest source/settings differ; regional identity alone is verified."
    return {"name": run["name"], "key": run_key, "method": run["manifest"]["method"],
            "mode": "saved", "ground_mask": ground, "coverage_mask": coverage,
            "parameters": run["parameters"].copy(), "runtime_seconds": run["manifest"].get("runtime_seconds"),
            "runtime_scope": HISTORICAL_RUNTIME_SCOPE, "load_runtime_seconds": time.perf_counter() - start,
            "counts": _counts(region, ground), "source_note": note,
            "processing_context": "entire historical development sample, then cropped for evaluation",
            "input_fingerprint": region["fingerprint"], "manifest_path": run["path"],
            "source_manifest_sample_points": run["manifest"]["sources"][region["side"]]["sample_point_count"]}


def _write_las(path: Path, header: laspy.LasHeader, records: np.ndarray) -> None:
    """Write a new file without allocating a point array sized by the raw header."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with laspy.open(path, mode="w", header=header.copy()) as writer:
        for offset in range(0, len(records), CHUNK_POINTS):
            points = laspy.ScaleAwarePointRecord(records[offset:offset + CHUNK_POINTS],
                                                header.point_format, header.scales, header.offsets)
            writer.write_points(points)


def _unique_folder(catalog: dict, kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    parent = catalog["root"] / "outputs/experiments/notebook"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{stamp}_{kind}_{uuid.uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    return path


def _region_settings(region: dict) -> dict:
    return {key: region[key] for key in ("side", "source_path", "source_info", "source_file_stat",
            "roi_xy", "context_roi_xy", "padding", "origin", "fingerprint", "load_runtime_seconds")}


def _versions(catalog: dict) -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "laspy", "cloth_simulation_filter", "pyproj"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    for filename in ("sample_and_ground.py", "ground_experiment_tools.py"):
        path = catalog["root"] / filename
        versions[f"{filename}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "unavailable"
    return versions


def _check_parameters(region: dict, baseline: dict, csf: dict) -> None:
    expected_baseline = {"cell_size", "tolerance"}
    expected_csf = {"cloth_resolution", "class_threshold", "rigidness", "time_step", "iterations", "slope_smooth"}
    if set(baseline) != expected_baseline or set(csf) != expected_csf:
        raise ValueError(f"Parameter keys must be baseline={sorted(expected_baseline)}, csf={sorted(expected_csf)}")
    for label, value, positive in (("cell_size", baseline["cell_size"], True),
                                   ("tolerance", baseline["tolerance"], False),
                                   ("cloth_resolution", csf["cloth_resolution"], True),
                                   ("class_threshold", csf["class_threshold"], False),
                                   ("time_step", csf["time_step"], True)):
        if not np.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    if (isinstance(csf["rigidness"], (bool, np.bool_)) or
            not isinstance(csf["rigidness"], (int, np.integer)) or csf["rigidness"] not in (1, 2, 3)):
        raise ValueError("rigidness must be 1, 2, or 3")
    if (isinstance(csf["iterations"], (bool, np.bool_)) or
            not isinstance(csf["iterations"], (int, np.integer)) or not 1 <= csf["iterations"] <= 2000):
        raise ValueError("iterations must be an integer between 1 and 2,000")
    if not isinstance(csf["slope_smooth"], (bool, np.bool_)):
        raise ValueError("slope_smooth must be True or False")
    if not 1 <= len(region["records"]) <= MAX_PROCESSING_POINTS:
        raise ValueError(f"This notebook processes between 1 and {MAX_PROCESSING_POINTS:,} context points")
    spans = np.ptp(region["xyz"][:, :2], axis=0)
    nodes = math.prod(math.ceil(float(span) / csf["cloth_resolution"]) + 6 for span in spans)
    if nodes > MAX_CSF_NODES:
        raise ValueError(f"Estimated CSF cloth has {nodes:,} nodes (limit {MAX_CSF_NODES:,}); increase cloth_resolution or shrink the context")


def run_experiment(catalog: dict, region: dict, baseline_params: dict, csf_params: dict,
                   observations: str | dict = "", display_settings: dict | None = None) -> dict:
    """Compute both existing project algorithms on exactly the same context.

    Uses ``sample_and_ground.ground_mask`` and ``csf_ground_mask`` directly. A new
    immutable run folder stores the context LAS, per-method masks and partitions,
    source/input manifests, settings, core/context counts, code versions, timings,
    and observations. Masks index the saved context input in its original order.
    Filter timings exclude input loading and output writing. New regional runs
    have less terrain context than historical whole-sample CSF runs.
    """
    _check_parameters(region, baseline_params, csf_params)
    from sample_and_ground import csf_ground_mask, ground_mask

    run_dir = _unique_folder(catalog, "run")
    with (run_dir / "settings.json").open("x", encoding="utf-8") as stream:
        json.dump(_jsonable({"region": _region_settings(region), "coordinate_units": UNITS,
                             "baseline_parameters": baseline_params, "csf_parameters": csf_params,
                             "observations": observations, "display_settings": display_settings or {}}),
                  stream, indent=2, allow_nan=False)
    started = time.perf_counter()
    io_started = time.perf_counter()
    _write_las(run_dir / "context_input.las", region["header"], region["records"])
    np.save(run_dir / "evaluation_mask.npy", region["evaluation_mask"], allow_pickle=False)
    io_seconds = time.perf_counter() - io_started
    results = []
    for method, function, parameters in (("baseline", ground_mask, baseline_params),
                                          ("csf", csf_ground_mask, csf_params)):
        tick = time.perf_counter()
        mask, _ = function(region["records"], region["header"], **parameters)
        elapsed = time.perf_counter() - tick
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(region["records"]),):
            raise ValueError(f"{method} returned a mask with the wrong shape")
        result = {"name": f"{method.upper() if method == 'csf' else 'Baseline'} local {run_dir.name[-8:]}",
                  "key": f"{run_dir.name}/{method}", "method": method, "mode": "computed",
                  "ground_mask": mask, "coverage_mask": np.ones(len(mask), dtype=bool),
                  "parameters": parameters.copy(), "runtime_seconds": elapsed,
                  "runtime_scope": "this method's filter call on this region plus padding; excludes input/output",
                  "counts": _counts(region, mask), "source_note": "Same region record array and fingerprint passed to both filters; no display sampling in processing.",
                  "processing_context": "selected region plus padding only",
                  "input_fingerprint": region["fingerprint"], "manifest_path": run_dir / "manifest.json",
                  "source_manifest_sample_points": region["source_info"]["sample_point_count"]}
        result["observations"] = (observations.get(result["key"], observations.get(result["name"], ""))
                                  if isinstance(observations, dict) else observations)
        results.append(result)
        tick = time.perf_counter()
        np.save(run_dir / f"{method}_ground_mask.npy", mask, allow_pickle=False)
        _write_las(run_dir / f"{method}_ground.las", region["header"], region["records"][mask])
        _write_las(run_dir / f"{method}_non_ground.las", region["header"], region["records"][~mask])
        io_seconds += time.perf_counter() - tick
    rows = comparison_rows(region, results, observations)
    manifest = {"kind": "regional_ground_experiment", "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_dir.name, "coordinate_units": UNITS, "region": _region_settings(region),
                "input_manifest_snapshot": region["input_manifest"], "versions": _versions(catalog),
                "display_settings": display_settings or {}, "observations": observations,
                "counts_scope": "core evaluation rectangle; context counts include halo",
                "results": [{k: v for k, v in r.items() if k not in ("ground_mask", "coverage_mask")} for r in results],
                "input_and_partition_write_seconds": io_seconds,
                "run_seconds_excluding_region_load_and_manifest_write": time.perf_counter() - started,
                "classification_note": "LAS classification attributes preserved; ground labels live in filenames and masks",
                "comparison_rows": rows}
    _save_metadata(run_dir, manifest, rows)
    return {"results": results, "run_dir": run_dir, "manifest": manifest}


def comparison_rows(region: dict, results: list[dict], observations: str | dict | None = None) -> list[dict]:
    """Build comparison records with all counts restricted to the same core ROI.

    Observation dictionaries may use a result's ``key`` or ``name``. Blank notes
    remain blank, rather than implying a visual finding or validated road label.
    Historical total runtimes and regional filter times are explicitly separated.
    """
    rows = []
    for result in results:
        if result["input_fingerprint"] != region["fingerprint"]:
            raise ValueError(f"{result['name']} belongs to a different selected input; reload results after changing the region")
        if not np.asarray(result["coverage_mask"]).all():
            raise ValueError(f"{result['name']} has incomplete input coverage")
        note = result.get("observations", "") if observations is None else observations
        if isinstance(note, dict):
            note = note.get(result["key"], note.get(result["name"], result.get("observations", "")))
        counts = result["counts"]
        rows.append({"name": result["name"], "experiment": result["name"], "result_key": result["key"], "mode": result["mode"],
                     "method": result["method"], "side": region["side"],
                     "input_sample": str(region["source_path"]), "input_sha256": region["fingerprint"],
                     "roi_xy": list(region["roi_xy"]), "padding": region["padding"],
                     "input_points": counts["input"], "ground_points": counts["ground"],
                     "non_ground_points": counts["non_ground"], "ground_fraction": counts["ground"] / counts["input"],
                     "loaded_context_points": counts["context_input"],
                     "historical_source_sample_points": result["source_manifest_sample_points"],
                     "parameters": result["parameters"].copy(),
                     "runtime_seconds": result["runtime_seconds"], "runtime_scope": result["runtime_scope"],
                     "processing_context": result["processing_context"], "same_input_evidence": result["source_note"],
                     "coordinate_units": UNITS, "observations": note,
                     "manifest_path": str(result["manifest_path"])})
    return rows


def _save_metadata(folder: Path, manifest: dict, rows: list[dict]) -> None:
    with (folder / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(manifest), stream, indent=2, allow_nan=False)
    with (folder / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in _jsonable(rows))


def save_review(catalog: dict, region: dict, results: list[dict], observations: str | dict,
                display_settings: dict | None = None) -> dict:
    """Save a unique JSON/CSV observation snapshot without recomputing filters."""
    rows = comparison_rows(region, results, observations)
    folder = _unique_folder(catalog, "review")
    manifest = {"kind": "ground_experiment_review", "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": folder.name, "coordinate_units": UNITS, "region": _region_settings(region),
                "input_manifest_snapshot": region["input_manifest"], "display_settings": display_settings or {},
                "observations": observations, "comparison_rows": rows,
                "source_manifest_snapshots": {str(result["manifest_path"]): _read_json(Path(result["manifest_path"])) for result in results}}
    _save_metadata(folder, manifest, rows)
    return {"run_dir": folder, "manifest": manifest, "rows": rows}


def history_rows(catalog: dict) -> list[dict]:
    """Read the comparison rows from every completed notebook run/review.

    Reviews are separate snapshots and can repeat a run with later observations.
    Unfinished run folders lacking a manifest are ignored, not deleted.
    """
    rows = []
    parent = catalog["root"] / "outputs/experiments/notebook"
    for path in sorted(parent.glob("*/manifest.json")):
        manifest = _read_json(path)
        for row in manifest.get("comparison_rows", []):
            rows.append({"snapshot_id": manifest["run_id"], "snapshot_kind": manifest["kind"], **row})
    return rows


def load_notebook_run(catalog: dict, region: dict, run_dir: str | Path) -> dict:
    """Reload a completed notebook run for visual comparison with this region.

    Only folders below ``outputs/experiments/notebook`` are accepted. The saved
    input records, coordinate encoding, ROI, halo, side, and evaluation mask must
    agree with the current region. Mismatches require selecting that earlier
    run's original region instead of aligning or silently dropping points.
    """
    folder = Path(run_dir)
    if not folder.is_absolute():
        folder = catalog["root"] / folder
    folder = folder.resolve()
    allowed = (catalog["root"] / "outputs/experiments/notebook").resolve()
    if folder == allowed or not folder.is_relative_to(allowed):
        raise ValueError(f"Choose a completed run folder inside {allowed}")
    manifest = _read_json(folder / "manifest.json")
    if manifest.get("kind") != "regional_ground_experiment":
        raise ValueError("This folder is a review snapshot, not a computed notebook run")
    settings = manifest["region"]
    for key in ("side", "roi_xy", "context_roi_xy", "padding", "fingerprint"):
        if settings[key] != region[key]:
            raise ValueError(f"Saved run {key} differs from the current region; select the run's original input and ROI before loading it")
    with laspy.open(folder / "context_input.las") as reader:
        if reader.header.point_count != len(region["records"]):
            raise ValueError("Saved context point count differs from selected input")
        points = reader.read_points(MAX_PROCESSING_POINTS + 1)
        actual = _record_fingerprint(points.array, reader.header, settings["roi_xy"],
                                     settings["context_roi_xy"], settings["side"])
    if actual != region["fingerprint"]:
        raise ValueError("Saved context records/coordinate encoding fail the exact input fingerprint check")
    core = np.load(folder / "evaluation_mask.npy", allow_pickle=False)
    if core.dtype != np.bool_ or not np.array_equal(core, region["evaluation_mask"]):
        raise ValueError("Saved evaluation mask differs from selected ROI")
    results = []
    for metadata in manifest["results"]:
        method = metadata["method"]
        if method not in ("baseline", "csf"):
            raise ValueError(f"Unrecognized saved method: {method}")
        mask = np.load(folder / f"{method}_ground_mask.npy", allow_pickle=False)
        if mask.dtype != np.bool_ or mask.shape != (len(region["records"]),):
            raise ValueError(f"Invalid saved {method} ground mask")
        if _counts(region, mask) != metadata["counts"]:
            raise ValueError(f"Saved {method} counts no longer match its mask")
        result = {**metadata, "mode": "reloaded", "ground_mask": mask,
                  "coverage_mask": np.ones(len(mask), dtype=bool), "manifest_path": folder / "manifest.json",
                  "source_note": metadata["source_note"] + " Reloaded context fingerprint and evaluation mask verified."}
        results.append(result)
    if {r["method"] for r in results} != {"baseline", "csf"} or len(results) != 2:
        raise ValueError("Saved run must contain one baseline and one CSF result")
    return {"results": results, "run_dir": folder, "manifest": manifest}
