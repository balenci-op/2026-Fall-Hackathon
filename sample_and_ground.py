"""Reproducible Run 1 spatial sampling and geometry-based ground separation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import laspy
import numpy as np
import CSF
from pyproj import CRS


DEFAULT_CHUNK_POINTS = 1_000_000
DEFAULT_TARGET_POINTS = 2_000_000
DEFAULT_GROUND_CELL_SIZE = 1.0
DEFAULT_GROUND_TOLERANCE = 0.35
EPSG_CODE = 6553
OPERATIONAL_UNITS = "meter"


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def scaled_xyz(points: laspy.ScaleAwarePointRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(points.x, dtype=np.float64),
        np.asarray(points.y, dtype=np.float64),
        np.asarray(points.z, dtype=np.float64),
    )


def header_bounds(header: laspy.LasHeader) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(header.mins, dtype=np.float64), np.asarray(header.maxs, dtype=np.float64)


def verify_epsg(header: laspy.LasHeader, sidecar_path: Path) -> dict[str, Any]:
    expected = CRS.from_epsg(EPSG_CODE)
    result: dict[str, Any] = {
        "expected": expected.to_string(),
        "expected_axis_units": [axis.unit_name for axis in expected.axis_info],
    }
    try:
        actual = header.parse_crs()
    except Exception as error:
        result["las_vlr"] = {"available": False, "error": str(error)}
    else:
        result["las_vlr"] = {
            "available": actual is not None,
            "matches_epsg_6553": bool(actual and actual.equals(expected)),
            "authority": actual.to_authority() if actual else None,
            "name": actual.name if actual else None,
            "axis_units": [axis.unit_name for axis in actual.axis_info] if actual else [],
        }

    sidecar_result: dict[str, Any] = {"available": False, "path": str(sidecar_path)}
    if sidecar_path.is_file():
        try:
            sidecar_text = sidecar_path.read_text(encoding="utf-8")
            wkt_line = next(line for line in sidecar_text.splitlines() if line.startswith("  WKT (GDAL): "))
            sidecar_crs = CRS.from_wkt(wkt_line.split("WKT (GDAL): ", 1)[1])
            sidecar_result.update({
                "available": True,
                "matches_epsg_6553": bool(sidecar_crs.equals(expected)),
                "authority": sidecar_crs.to_authority(),
                "name": sidecar_crs.name,
                "axis_units": [axis.unit_name for axis in sidecar_crs.axis_info],
                "wkt": sidecar_crs.to_wkt(),
                "note": "The supplied WKT contains EPSG authority 6553 but uses meter axes; official EPSG:6553 uses US survey feet." if not sidecar_crs.equals(expected) else None,
            })
        except Exception as error:
            sidecar_result["error"] = str(error)
    result["sidecar_wkt"] = sidecar_result
    result["matches_epsg_6553"] = bool(sidecar_result.get("matches_epsg_6553", False))
    return result


def stable_scores(raw_points: np.ndarray, seed: int) -> np.ndarray:
    raw = np.asarray(raw_points)
    x = raw["X"].astype(np.uint64, copy=False)
    y = raw["Y"].astype(np.uint64, copy=False)
    z = raw["Z"].astype(np.uint64, copy=False)
    value = x * np.uint64(0x9E3779B185EBCA87)
    value ^= y * np.uint64(0xC2B2AE3D27D4EB4F)
    value ^= z * np.uint64(0x165667B19E3779F9)
    value ^= np.uint64(seed)
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


def in_roi(points: laspy.ScaleAwarePointRecord, roi_min: np.ndarray, roi_max: np.ndarray) -> np.ndarray:
    x, y, z = scaled_xyz(points)
    return (
        (x >= roi_min[0]) & (x <= roi_max[0]) &
        (y >= roi_min[1]) & (y <= roi_max[1]) &
        (z >= roi_min[2]) & (z <= roi_max[2])
    )


def collect_sample(
    path: Path,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    target_points: int,
    chunk_points: int,
    seed: int,
) -> tuple[laspy.LasHeader, np.ndarray, dict[str, Any]]:
    records: list[np.ndarray] = []
    roi_count = 0
    with laspy.open(path) as reader:
        header = reader.header
        selection_probability = min(1.0, target_points / max(int(header.point_count), 1))
        for points in reader.chunk_iterator(chunk_points):
            roi_mask = in_roi(points, roi_min, roi_max)
            roi_points = points.array[roi_mask]
            roi_count += int(roi_points.shape[0])
            if roi_points.size == 0:
                continue
            scores = stable_scores(roi_points, seed)
            threshold = np.uint64(selection_probability * float(np.iinfo(np.uint64).max))
            selected = roi_points[scores <= threshold]
            if selected.size:
                records.append(selected.copy())
        sampled = np.concatenate(records) if records else np.empty(0, dtype=header.point_format.dtype())
    return header, sampled, {
        "source_path": str(path),
        "source_point_count": int(header.point_count),
        "roi_point_count": roi_count,
        "sample_point_count": int(sampled.shape[0]),
        "selection_probability": selection_probability,
        "seed": seed,
    }


def voxel_downsample(records: np.ndarray, header: laspy.LasHeader, voxel_size: float) -> np.ndarray:
    if records.size == 0:
        return records
    x = records["X"].astype(np.float64) * header.x_scale + header.x_offset
    y = records["Y"].astype(np.float64) * header.y_scale + header.y_offset
    z = records["Z"].astype(np.float64) * header.z_scale + header.z_offset
    keys = np.column_stack((
        np.floor(x / voxel_size).astype(np.int64),
        np.floor(y / voxel_size).astype(np.int64),
        np.floor(z / voxel_size).astype(np.int64),
    ))
    _, first_indices = np.unique(keys, axis=0, return_index=True)
    return records[np.sort(first_indices)]


def ground_mask(records: np.ndarray, header: laspy.LasHeader, cell_size: float, tolerance: float) -> tuple[np.ndarray, dict[str, Any]]:
    if records.size == 0:
        return np.zeros(0, dtype=bool), {"cell_count": 0, "ground_point_count": 0}
    x = records["X"].astype(np.float64) * header.x_scale + header.x_offset
    y = records["Y"].astype(np.float64) * header.y_scale + header.y_offset
    z = records["Z"].astype(np.float64) * header.z_scale + header.z_offset
    cell_x = np.floor(x / cell_size).astype(np.int64)
    cell_y = np.floor(y / cell_size).astype(np.int64)
    keys = np.rec.fromarrays((cell_x, cell_y), names="x,y")
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    cell_min = np.full(unique_keys.shape[0], np.inf, dtype=np.float64)
    np.minimum.at(cell_min, inverse, z)
    residual = z - cell_min[inverse]
    mask = residual <= tolerance
    return mask, {
        "cell_count": int(unique_keys.shape[0]),
        "ground_point_count": int(np.count_nonzero(mask)),
        "non_ground_point_count": int(np.count_nonzero(~mask)),
        "ground_fraction": float(np.mean(mask)),
        "cell_size": cell_size,
        "height_tolerance": tolerance,
        "method": "lowest point per XY cell plus residual tolerance",
        "evaluation_note": "Unlabelled geometry-based proxy; review in CloudCompare before using for detection.",
    }


def csf_ground_mask(
    records: np.ndarray,
    header: laspy.LasHeader,
    cloth_resolution: float,
    class_threshold: float,
    rigidness: int,
    time_step: float,
    iterations: int,
    slope_smooth: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if records.size == 0:
        return np.zeros(0, dtype=bool), {"ground_point_count": 0, "non_ground_point_count": 0}
    x = records["X"].astype(np.float64) * header.x_scale + header.x_offset
    y = records["Y"].astype(np.float64) * header.y_scale + header.y_offset
    z = records["Z"].astype(np.float64) * header.z_scale + header.z_offset
    points = np.column_stack((x, y, z)).tolist()
    params = CSF.Params()
    params.cloth_resolution = cloth_resolution
    params.class_threshold = class_threshold
    params.rigidness = rigidness
    params.time_step = time_step
    params.interations = iterations
    params.bSloopSmooth = slope_smooth
    filter_instance = CSF.CSF()
    filter_instance.params = params
    filter_instance.setPointCloud(points)
    ground_indices = CSF.VecInt()
    off_ground_indices = CSF.VecInt()
    filter_instance.do_filtering(ground_indices, off_ground_indices, False)
    mask = np.zeros(records.shape[0], dtype=bool)
    mask[np.asarray(list(ground_indices), dtype=np.int64)] = True
    return mask, {
        "ground_point_count": int(np.count_nonzero(mask)),
        "non_ground_point_count": int(np.count_nonzero(~mask)),
        "ground_fraction": float(np.mean(mask)),
        "method": "Cloth Simulation Filter (CSF)",
        "parameters": {
            "cloth_resolution_m": cloth_resolution,
            "class_threshold_m": class_threshold,
            "rigidness": rigidness,
            "time_step": time_step,
            "iterations": iterations,
            "slope_smooth": slope_smooth,
        },
        "evaluation_note": "Recognized geometry-based terrain filter; review against baseline in CloudCompare.",
    }


def write_las(path: Path, header: laspy.LasHeader, records: np.ndarray) -> None:
    output_header = header.copy()
    output = laspy.LasData(output_header)
    output.points = laspy.PackedPointRecord(records, output_header.point_format)
    output.write(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample the shared Run 1 ROI and separate geometry-based ground/non-ground points")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--inputs-dir", type=Path, default=Path("outputs/run1_inputs"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("outputs/experiments/baseline"))
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS)
    parser.add_argument("--chunk-points", type=int, default=DEFAULT_CHUNK_POINTS)
    parser.add_argument("--ground-cell-size", type=float, default=DEFAULT_GROUND_CELL_SIZE)
    parser.add_argument("--ground-tolerance", type=float, default=DEFAULT_GROUND_TOLERANCE)
    parser.add_argument("--csf-cloth-resolution", type=float, default=0.5)
    parser.add_argument("--csf-class-threshold", type=float, default=0.5)
    parser.add_argument("--csf-rigidness", type=int, default=2)
    parser.add_argument("--csf-time-step", type=float, default=0.65)
    parser.add_argument("--csf-iterations", type=int, default=500)
    parser.add_argument("--csf-slope-smooth", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional preview voxel size in coordinate units; 0 disables it")
    args = parser.parse_args()
    if args.method not in {"baseline", "csf"}:
        parser.error("--method must be baseline or csf")
    if args.target_points <= 0 or args.chunk_points <= 0 or args.ground_cell_size <= 0 or args.ground_tolerance < 0 or args.voxel_size < 0 or args.csf_cloth_resolution <= 0 or args.csf_class_threshold < 0 or args.csf_time_step <= 0 or args.csf_iterations <= 0:
        parser.error("sampling and ground parameters must be positive; voxel size may be zero")
    if not args.left.is_file() or not args.right.is_file():
        parser.error("both LAS input files must exist")

    with laspy.open(args.left) as left_reader, laspy.open(args.right) as right_reader:
        left_min, left_max = header_bounds(left_reader.header)
        right_min, right_max = header_bounds(right_reader.header)
        roi_min = np.maximum(left_min, right_min)
        roi_max = np.minimum(left_max, right_max)
        if np.any(roi_min >= roi_max):
            parser.error("the input files do not have a positive-volume common bounding region")
        crs_results = {
            "left": verify_epsg(left_reader.header, args.left.with_suffix(args.left.suffix + ".txt")),
            "right": verify_epsg(right_reader.header, args.right.with_suffix(args.right.suffix + ".txt")),
        }

    inputs_dir = args.inputs_dir
    experiment_dir = args.experiment_dir
    inputs_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    sources = [("left", args.left, 101), ("right", args.right, 202)]
    input_manifest_path = inputs_dir / "run1_inputs_manifest.json"
    existing_manifest = None
    if input_manifest_path.is_file():
        existing_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    settings_match = bool(existing_manifest and existing_manifest.get("shared_roi", {}).get("min") == roi_min.tolist() and
        existing_manifest.get("shared_roi", {}).get("max") == roi_max.tolist() and
        existing_manifest.get("parameters", {}).get("target_points_per_source") == args.target_points and
        existing_manifest.get("parameters", {}).get("voxel_size") == (args.voxel_size if args.voxel_size > 0 else None))
    manifest: dict[str, Any] = {
        "read_only_inputs": True,
        "method": args.method,
        "runtime_seconds": None,
        "crs_verification": crs_results,
        "shared_roi": {"min": roi_min.tolist(), "max": roi_max.tolist(), "coordinate_units": OPERATIONAL_UNITS},
        "unit_resolution": {
            "status": "resolved_for_processing",
            "operational_units": OPERATIONAL_UNITS,
            "basis": ["Export unit: Meter", "sidecar WKT horizontal axes: Meter"],
            "epsg_authority": EPSG_CODE,
            "epsg_assignment": "not applied",
            "warning": "The sidecar cites EPSG:6553 but official EPSG:6553 uses US survey feet; processing preserves source coordinates and uses meter-valued parameters from the export WKT.",
        },
        "parameters": {
            "target_points_per_source": args.target_points,
            "chunk_points": args.chunk_points,
            "ground_cell_size": args.ground_cell_size,
            "ground_tolerance": args.ground_tolerance,
            "voxel_size": args.voxel_size if args.voxel_size > 0 else None,
            "sampling": "deterministic spatial hash over points in the shared 3D ROI",
            "unit": OPERATIONAL_UNITS,
            "csf": {
                "cloth_resolution_m": args.csf_cloth_resolution,
                "class_threshold_m": args.csf_class_threshold,
                "rigidness": args.csf_rigidness,
                "time_step": args.csf_time_step,
                "iterations": args.csf_iterations,
                "slope_smooth": args.csf_slope_smooth,
            },
        },
        "sources": {},
    }

    input_manifest: dict[str, Any] = {
        "read_only_inputs": True,
        "kind": "reusable_run1_development_inputs",
        "crs_verification": crs_results,
        "shared_roi": manifest["shared_roi"],
        "parameters": manifest["parameters"],
        "sources": {},
    }
    pipeline_start = time.perf_counter()
    for name, path, seed in sources:
        sample_path = inputs_dir / f"run1_{name}_sample.las"
        preview_path = inputs_dir / f"run1_{name}_preview_voxel.las"
        reusable = settings_match and sample_path.is_file() and (args.voxel_size <= 0 or preview_path.is_file())
        if reusable:
            print(f"Reusing {name} development input: {sample_path}", flush=True)
            with laspy.open(sample_path) as sample_reader:
                header = sample_reader.header
                sampled = sample_reader.read().points.array.copy()
            sample_info = existing_manifest["sources"][name]
        else:
            print(f"Sampling {name}: {path}", flush=True)
            header, sampled, sample_info = collect_sample(path, roi_min, roi_max, args.target_points, args.chunk_points, seed)
            write_las(sample_path, header, sampled)
            if args.voxel_size > 0:
                preview = voxel_downsample(sampled, header, args.voxel_size)
                write_las(preview_path, header, preview)
                sample_info["voxel_preview_point_count"] = int(preview.shape[0])
        input_manifest["sources"][name] = {
            key: value for key, value in sample_info.items()
            if key in {"source_path", "source_point_count", "roi_point_count", "sample_point_count", "selection_probability", "seed", "voxel_preview_point_count"}
        }
        if args.method == "csf":
            ground, ground_info = csf_ground_mask(
                sampled, header, args.csf_cloth_resolution, args.csf_class_threshold,
                args.csf_rigidness, args.csf_time_step, args.csf_iterations,
                args.csf_slope_smooth,
            )
        else:
            ground, ground_info = ground_mask(sampled, header, args.ground_cell_size, args.ground_tolerance)
        write_las(experiment_dir / f"run1_{name}_ground.las", header, sampled[ground])
        write_las(experiment_dir / f"run1_{name}_non_ground.las", header, sampled[~ground])
        sample_info.update({"ground_separation": ground_info, "output_files": {
            "sample": f"../../run1_inputs/run1_{name}_sample.las",
            "ground": f"run1_{name}_ground.las",
            "non_ground": f"run1_{name}_non_ground.las",
        }})
        manifest["sources"][name] = sample_info

    manifest["runtime_seconds"] = time.perf_counter() - pipeline_start
    input_manifest["runtime_seconds"] = manifest["runtime_seconds"]
    input_manifest_path.write_text(json.dumps(json_value(input_manifest), indent=2), encoding="utf-8")
    (experiment_dir / f"{args.method}_manifest.json").write_text(json.dumps(json_value(manifest), indent=2), encoding="utf-8")
    print(f"Wrote experiment outputs to {experiment_dir}")


if __name__ == "__main__":
    main()