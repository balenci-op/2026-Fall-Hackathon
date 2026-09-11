"""Read-only, chunked profiler for LAS files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import laspy
import numpy as np


DEFAULT_CHUNK_POINTS = 1_000_000
DEFAULT_RESERVOIR_SIZE = 200_000


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


class StreamingStats:
    def __init__(self, capacity: int, seed: int) -> None:
        self.count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.mean = 0.0
        self.m2 = 0.0
        self.reservoir = np.empty(capacity, dtype=np.float64)
        self.reservoir_size = 0
        self.capacity = capacity
        self.random = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        chunk_count = int(values.size)
        chunk_mean = float(np.mean(values))
        chunk_delta = values - chunk_mean
        chunk_m2 = float(np.sum(chunk_delta * chunk_delta))
        if self.count == 0:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
        else:
            total = self.count + chunk_count
            delta = chunk_mean - self.mean
            self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total
            self.mean += delta * chunk_count / total
            self.count = total
        if self.reservoir_size < self.capacity:
            remaining = self.capacity - self.reservoir_size
            if values.size <= remaining:
                self.reservoir[self.reservoir_size:self.reservoir_size + values.size] = values
                self.reservoir_size += values.size
            else:
                selected = self.random.choice(values, size=remaining, replace=False)
                self.reservoir[self.reservoir_size:] = selected
                self.reservoir_size = self.capacity
            return
        ranks = np.arange(self.count - chunk_count + 1, self.count + 1, dtype=np.float64)
        candidates = (self.random.random(chunk_count) * ranks).astype(np.int64)
        replace_mask = candidates < self.capacity
        if np.any(replace_mask):
            self.reservoir[candidates[replace_mask]] = values[replace_mask]

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"available": False, "count": 0}
        sample = self.reservoir[:self.reservoir_size]
        percentile_values = np.percentile(sample, [1, 5, 50, 95, 99])
        return {
            "available": True,
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "standard_deviation": math.sqrt(self.m2 / self.count),
            "median_estimate": float(percentile_values[2]),
            "p01_estimate": float(percentile_values[0]),
            "p05_estimate": float(percentile_values[1]),
            "p95_estimate": float(percentile_values[3]),
            "p99_estimate": float(percentile_values[4]),
            "percentile_method": "reservoir_sample",
            "percentile_sample_count": int(sample.size),
        }


def safe_header_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def vlr_summary(records: Any) -> list[dict[str, Any]]:
    result = []
    for record in records:
        result.append({
            "type": type(record).__name__,
            "user_id": safe_header_value(getattr(record, "user_id", None)),
            "record_id": safe_header_value(getattr(record, "record_id", None)),
            "description": safe_header_value(getattr(record, "description", None)),
            "record_length": len(getattr(record, "record_data", b"")),
        })
    return result


def dimension_array(points: Any, name: str) -> np.ndarray | None:
    if name not in points.point_format.dimension_names:
        return None
    if name == "X":
        return np.asarray(points.x, dtype=np.float64)
    if name == "Y":
        return np.asarray(points.y, dtype=np.float64)
    if name == "Z":
        return np.asarray(points.z, dtype=np.float64)
    return np.asarray(points[name], dtype=np.float64)


def add_counts(counter: Counter[str], values: np.ndarray) -> None:
    unique_values, counts = np.unique(values, return_counts=True)
    counter.update({str(value): int(count) for value, count in zip(unique_values, counts)})


def metadata(path: Path, header: Any) -> dict[str, Any]:
    dimensions = [str(name) for name in header.point_format.dimension_names]
    standard_names = {
        name
        for dimensions in laspy.point.dims.POINT_FORMAT_DIMENSIONS.values()
        for name in dimensions
    }
    standard_names.update({
        "return_number", "number_of_returns", "synthetic", "key_point", "withheld", "overlap",
        "scanner_channel", "scan_direction_flag", "edge_of_flight_line",
    })
    try:
        crs = header.parse_crs()
    except Exception as error:
        crs = f"unavailable: {error}"
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "las_version": str(header.version),
        "point_format": header.point_format.id,
        "point_record_length": header.point_format.size,
        "point_count": int(header.point_count),
        "generating_software": str(header.generating_software),
        "system_identifier": str(header.system_identifier),
        "creation_date": str(header.creation_date) if header.creation_date else None,
        "crs": str(crs) if crs else None,
        "scales": [float(value) for value in header.scales],
        "offsets": [float(value) for value in header.offsets],
        "header_xyz_bounds": {
            "min": [float(value) for value in header.mins],
            "max": [float(value) for value in header.maxs],
        },
        "point_dimensions": dimensions,
        "extra_or_custom_dimensions": [name for name in dimensions if name not in standard_names],
        "vlrs": vlr_summary(header.vlrs),
        "evlrs": vlr_summary(header.evlrs) if header.evlrs else [],
    }


def profile_file(path: Path, chunk_points: int, reservoir_size: int) -> dict[str, Any]:
    with laspy.open(path) as reader:
        header = reader.header
        dimensions = [str(name) for name in header.point_format.dimension_names]
        stats = {name: StreamingStats(reservoir_size, index) for index, name in enumerate(dimensions)}
        value_counts: dict[str, Counter[str]] = defaultdict(Counter)
        return_cross_tab: Counter[str] = Counter()
        processed = 0
        for points in reader.chunk_iterator(chunk_points):
            processed += len(points)
            for name in dimensions:
                values = dimension_array(points, name)
                if values is not None and np.issubdtype(values.dtype, np.number):
                    stats[name].update(values)
            for name in ("classification", "return_number", "number_of_returns", "synthetic_flag", "key_point_flag", "withheld_flag", "overlap_flag"):
                values = dimension_array(points, name)
                if values is not None:
                    add_counts(value_counts[name], values)
            return_numbers = dimension_array(points, "return_number")
            total_returns = dimension_array(points, "number_of_returns")
            if return_numbers is not None and total_returns is not None:
                pairs = np.rec.fromarrays([return_numbers, total_returns], names="return_number,total_returns")
                unique_pairs, counts = np.unique(pairs, return_counts=True)
                return_cross_tab.update({
                    f"{int(pair.return_number)} x {int(pair.total_returns)}": int(count)
                    for pair, count in zip(unique_pairs, counts)
                })
        return {
            "metadata": metadata(path, header),
            "processed_point_count": processed,
            "numeric_statistics": {name: item.summary() for name, item in stats.items()},
            "categorical_counts": {
                name: {"available": True, "counts": dict(counter), "total": sum(counter.values())}
                for name, counter in value_counts.items()
            },
            "return_number_by_number_of_returns": dict(return_cross_tab),
            "read_only": True,
            "operations_performed": ["header_read", "chunked_point_statistics"],
        }


def comparison(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    left = next((item for item in profiles if "left" in Path(item["metadata"]["path"]).stem.lower()), None)
    right = next((item for item in profiles if "right" in Path(item["metadata"]["path"]).stem.lower()), None)
    if not left or not right:
        return None
    left_meta, right_meta = left["metadata"], right["metadata"]
    left_min, left_max = left_meta["header_xyz_bounds"]["min"], left_meta["header_xyz_bounds"]["max"]
    right_min, right_max = right_meta["header_xyz_bounds"]["min"], right_meta["header_xyz_bounds"]["max"]
    common_min = [max(left_min[index], right_min[index]) for index in range(3)]
    common_max = [min(left_max[index], right_max[index]) for index in range(3)]
    common_extent = [max(0.0, common_max[index] - common_min[index]) for index in range(3)]
    left_extent = [left_max[index] - left_min[index] for index in range(3)]
    right_extent = [right_max[index] - right_min[index] for index in range(3)]
    common_volume = math.prod(common_extent)
    return {
        "left_point_count": left_meta["point_count"],
        "right_point_count": right_meta["point_count"],
        "left_xyz_bounds": left_meta["header_xyz_bounds"],
        "right_xyz_bounds": right_meta["header_xyz_bounds"],
        "common_bounding_region": {"min": common_min, "max": common_max, "volume": common_volume},
        "approximate_density_points_per_cubic_unit": {
            "left": left_meta["point_count"] / math.prod(left_extent) if math.prod(left_extent) > 0 else None,
            "right": right_meta["point_count"] / math.prod(right_extent) if math.prod(right_extent) > 0 else None,
        },
        "gps_time_ranges": {
            "left": left["numeric_statistics"].get("gps_time"),
            "right": right["numeric_statistics"].get("gps_time"),
        },
        "available_dimensions": {"left": left_meta["point_dimensions"], "right": right_meta["point_dimensions"]},
        "schemas_match": left_meta["point_dimensions"] == right_meta["point_dimensions"],
    }


def human_report(summary: dict[str, Any]) -> str:
    lines = ["LAS PROFILING REPORT", "=" * 20, "", "Read-only: yes", ""]
    for profile in summary["files"]:
        meta = profile["metadata"]
        lines.extend([
            f"FILE: {meta['path']}", f"  Size: {meta['file_size_bytes']:,} bytes",
            f"  Points: {meta['point_count']:,}",
            f"  LAS version / point format: {meta['las_version']} / {meta['point_format']}",
            f"  Record length: {meta['point_record_length']} bytes",
            f"  System / software: {meta['system_identifier']} / {meta['generating_software']}",
            f"  Creation date: {meta['creation_date']}", f"  CRS: {meta['crs']}",
            f"  Scales: {meta['scales']}", f"  Offsets: {meta['offsets']}",
            f"  Header XYZ bounds: {meta['header_xyz_bounds']}",
            f"  Dimensions: {', '.join(meta['point_dimensions'])}",
            f"  Extra/custom dimensions: {meta['extra_or_custom_dimensions']}",
            f"  VLRs / EVLRs: {len(meta['vlrs'])} / {len(meta['evlrs'])}",
            "  Numeric statistics:",
        ])
        for name, values in profile["numeric_statistics"].items():
            if values.get("available"):
                lines.append(
                    f"    {name}: min={values['min']:.6g}, max={values['max']:.6g}, mean={values['mean']:.6g}, "
                    f"sd={values['standard_deviation']:.6g}, median~={values['median_estimate']:.6g}, "
                    f"p01~={values['p01_estimate']:.6g}, p05~={values['p05_estimate']:.6g}, "
                    f"p95~={values['p95_estimate']:.6g}, p99~={values['p99_estimate']:.6g}"
                )
        lines.append("  Categorical counts:")
        for name, values in profile["categorical_counts"].items():
            lines.append(f"    {name}: {values['counts']}")
        lines.extend([f"  Return cross-tab: {profile['return_number_by_number_of_returns']}", ""])
    if summary.get("comparison"):
        lines.extend(["LEFT/RIGHT COMPARISON", "=" * 22, json.dumps(summary["comparison"], indent=2), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only, chunked LAS profiler")
    parser.add_argument("paths", nargs="+", type=Path, help="LAS files to profile")
    parser.add_argument("--output", required=True, type=Path, help="Output prefix, e.g. reports/run1")
    parser.add_argument("--chunk-points", type=int, default=DEFAULT_CHUNK_POINTS)
    parser.add_argument("--reservoir-size", type=int, default=DEFAULT_RESERVOIR_SIZE)
    args = parser.parse_args()
    if args.chunk_points <= 0 or args.reservoir_size <= 0:
        parser.error("--chunk-points and --reservoir-size must be positive")
    for path in args.paths:
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiles = []
    for path in args.paths:
        print(f"Profiling {path} ...", flush=True)
        profile = profile_file(path, args.chunk_points, args.reservoir_size)
        profiles.append(profile)
        per_file = args.output.parent / f"{args.output.name}_{path.stem.lower().replace(' ', '_')}_summary.json"
        per_file.write_text(json.dumps(json_value(profile), indent=2), encoding="utf-8")
    summary = {
        "read_only": True,
        "configuration": {"chunk_points": args.chunk_points, "reservoir_size": args.reservoir_size, "percentiles": "approximate from bounded reservoir sample"},
        "files": profiles,
        "comparison": comparison(profiles),
    }
    args.output.with_name(f"{args.output.name}_summary.json").write_text(json.dumps(json_value(summary), indent=2), encoding="utf-8")
    args.output.with_name(f"{args.output.name}_report.txt").write_text(human_report(json_value(summary)), encoding="utf-8")
    print(f"Wrote {args.output.with_name(f'{args.output.name}_report.txt')}")
    print(f"Wrote {args.output.with_name(f'{args.output.name}_summary.json')}")


if __name__ == "__main__":
    main()