"""One bounded dense-input quality pass; raw scans are explicit and never notebook defaults.

The original sparse membership and all reviews remain untouched. Crops are regions,
not objects. A single local CSF + spatial grouping configuration proposes one dense
component per original seed. Every rejected/raw point remains in the crop cache.
Distances use unresolved source coordinate units, without conversion.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree

import object_candidates as objects
import ground_experiment_tools as ground
from sample_and_ground import csf_ground_mask


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    objects._atomic_json(Path(path), value)


def original_candidate(source, batch, candidate_id):
    """Recover an existing retained group without reclustering or changing its ID."""
    row = next(row for row in batch["cluster_table"] if row["candidate_id"] == candidate_id)
    if row["status"] == "below_min_cluster_points":
        raise ValueError("This pass only inspects existing retained candidates")
    indices = batch["eligible_indices"][batch["labels"] == row["cluster_label"]]
    return {**row, "id": candidate_id, "indices": indices,
            "xyz": source["xyz"][indices], "rgb": source["rgb"][indices],
            "sample_record_indices": source["sample_record_indices"][indices],
            "source_identity": source["identity"], "batch_cache_key": batch["cache_key"]}


def _stat(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def extract_raw_crops(project_dir, source, candidates, run_dir, margin_xy=5.0,
                      allow_raw_scan=False, max_crop_points=1_000_000):
    """Collect <=3 XY boxes in ONE chunked read of the matching raw LAS.

    Keep the original sample's shared Z interval, including local ground, so CSF
    has terrain context. The original candidate's XYZ bounds remain in provenance.
    Cache full records, exact XYZ/RGB, and zero-based raw record rows. The SHA256
    covers the point-record stream, not the entire LAS file. Reload uses source
    stat/header and cached array hashes without another whole-file scan.
    """
    root, run_dir = Path(project_dir).resolve(), Path(run_dir).resolve()
    if not 1 <= len(candidates) <= 3 or margin_xy <= 0:
        raise ValueError("Select 1..3 candidates with a positive context margin")
    side = source["region"]["side"]
    input_manifest = source["catalog"]["input_manifest"]
    info = input_manifest["sources"][side]
    raw_path = root / info["source_path"]
    before = _stat(raw_path)
    bounds = []
    for candidate in candidates:
        if candidate["source_identity"] != source["identity"]:
            raise ValueError("All candidates must belong to the same source")
        low, high = np.array(candidate["bounds"]["min"]), np.array(candidate["bounds"]["max"])
        bounds.append({"min": [low[0]-margin_xy, low[1]-margin_xy, input_manifest["shared_roi"]["min"][2]],
                       "max": [high[0]+margin_xy, high[1]+margin_xy, input_manifest["shared_roi"]["max"][2]]})
    settings = {"source_identity": source["identity"], "raw_path": str(raw_path),
                "raw_stat": before, "candidate_ids": [c["id"] for c in candidates],
                "bounds": bounds, "margin_xy": margin_xy, "max_crop_points": max_crop_points}
    manifest_path = run_dir / "raw_crops/manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest["settings"] != objects._plain(settings):
            raise ValueError("Crop settings/input changed; preserve this run and use a new folder")
        for crop in manifest["crops"]:
            if objects._sha_file(crop["path"]) != crop["sha256"]:
                raise ValueError(f"Crop cache is damaged: {crop['path']}")
        return {**manifest, "cache_hit": True}
    if not allow_raw_scan:
        raise FileNotFoundError("Dense crops are not cached. Raw scanning is off; use the explicit bounded extraction function.")
    if (run_dir / "raw_crops/scan_started.json").exists():
        raise RuntimeError("A raw pass was already attempted here; do not silently repeat it")
    write_json(run_dir / "raw_crops/scan_started.json", {"settings": settings, "started_utc": datetime.now(timezone.utc).isoformat()})
    records_parts, rows_parts = [[] for _ in candidates], [[] for _ in candidates]
    counts, scanned, chunks = [0]*len(candidates), 0, 0
    digest = hashlib.sha256()
    started = time.perf_counter()
    with laspy.open(raw_path) as reader:
        header = reader.header.copy()
        sample_header = source["region"]["header"]
        if reader.header.point_count != info["source_point_count"]:
            raise ValueError("Raw source count no longer matches input manifest")
        if not np.array_equal(header.scales, sample_header.scales) or not np.array_equal(header.offsets, sample_header.offsets):
            raise ValueError("Raw/sample coordinate encoding differs")
        for points in reader.chunk_iterator(1_000_000):
            records = points.array
            digest.update(memoryview(records).cast("B"))
            xyz = ground._xyz(records, header)
            for index, box in enumerate(bounds):
                inside = np.all((xyz >= box["min"]) & (xyz <= box["max"]), axis=1)
                rows = np.flatnonzero(inside)
                counts[index] += len(rows)
                if counts[index] > max_crop_points:
                    raise RuntimeError("Explicit crop point cap exceeded; raw pass stopped, no truncation or automatic retry")
                if len(rows):
                    records_parts[index].append(records[rows].copy())
                    rows_parts[index].append(rows.astype(np.int64) + scanned)
            scanned += len(records)
            chunks += 1
            if chunks % 20 == 0:
                print(f"Single raw pass: {scanned:,}/{header.point_count:,} points; crops {counts}", flush=True)
    if before != _stat(raw_path) or scanned != header.point_count:
        raise RuntimeError("Raw file changed or read was incomplete; no completed manifest saved")
    crop_rows = []
    for index, candidate in enumerate(candidates):
        records = np.concatenate(records_parts[index]) if records_parts[index] else np.empty(0, dtype=header.point_format.dtype())
        raw_rows = np.concatenate(rows_parts[index]) if rows_parts[index] else np.empty(0, dtype=np.int64)
        path = run_dir / "raw_crops" / f"{candidate['id']}.npz"
        xyz = ground._xyz(records, header)
        rgb_raw = np.column_stack([records[name] for name in ("red", "green", "blue")])
        np.savez_compressed(path, records=records, xyz=xyz, rgb_raw=rgb_raw, raw_record_indices=raw_rows)
        # Every sparse point should be recovered in its wider raw crop.
        raw_keys = {record.tobytes() for record in records}
        matched = sum(source["region"]["records"][row].tobytes() in raw_keys for row in candidate["indices"])
        if matched != candidate["point_count"]:
            raise ValueError("Not all original sparse seed records were recovered from matching raw source")
        crop_rows.append({"parent_candidate_id": candidate["id"], "path": str(path),
                          "sha256": objects._sha_file(path), "point_count": len(records),
                          "bounds": bounds[index], "original_candidate_bounds": candidate["bounds"],
                          "original_points_recovered_exactly": matched})
    manifest = {"kind": "one_pass_raw_region_crops", "settings": settings,
                "created_utc": datetime.now(timezone.utc).isoformat(), "raw_passes": 1,
                "raw_points_scanned": scanned, "chunks": chunks,
                "raw_point_record_sha256": digest.hexdigest(), "raw_stat_after": _stat(raw_path),
                "scales": header.scales.tolist(), "offsets": header.offsets.tolist(),
                "point_format": header.point_format.id, "las_version": str(header.version),
                "coordinate_units": ground.UNITS, "crops": crop_rows,
                "runtime_seconds": time.perf_counter()-started,
                "note": "Crops contain ground and possibly multiple objects. Shared source-sample Z interval is retained; no crop inherits original object membership."}
    write_json(manifest_path, manifest)
    return {**manifest, "cache_hit": False}


def load_crop(crop):
    with np.load(crop["path"], allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    data["rgb"] = data["rgb_raw"].astype(float)/65535.0
    return data


def nn_spacing(xyz, max_queries=6000):
    """Query exact neighbors in all supplied points; only statistics queries are sampled."""
    if len(xyz) < 2:
        return {"points": len(xyz), "median": None, "p90": None}
    query_indices = np.linspace(0, len(xyz)-1, min(max_queries, len(xyz)), dtype=int)
    distances = cKDTree(xyz).query(xyz[query_indices], k=2, workers=1)[0][:, 1]
    return {"points": len(xyz), "queried_points": len(query_indices),
            "median": float(np.median(distances)), "p90": float(np.quantile(distances, .9)),
            "duplicate_query_fraction": float(np.mean(distances == 0))}


def group_dense_crop(source, sparse, crop, raw_manifest, run_dir):
    """One fixed local CSF + voxel DBSCAN attempt, selecting by original seed support.

    Existing CSF parameters are reused, not tuned. Dense points need new labels:
    the saved sparse mask is never copied onto newly recovered raw points. CSF
    sees a local crop, so its terrain context differs from the prior whole sample.
    Voxel representatives are real points and are used for grouping only; every
    raw point in the selected voxels is restored to the dense candidate.
    """
    folder = Path(run_dir) / "grouping" / sparse["id"]
    manifest_path = folder / "manifest.json"
    params = ground._parameters(source["catalog"]["runs"]["csf_current"]["manifest"])
    provenance = {"crop_sha256": crop["sha256"], "original_candidate_id": sparse["id"],
                  "csf_parameters": params, "helper_sha256": objects._sha_file(__file__),
                  "sample_and_ground_sha256": objects._sha_file(Path(csf_ground_mask.__code__.co_filename)),
                  "voxel_rule": "clip(2 * median raw nearest-neighbor spacing, 0.06, 0.18)",
                  "eps_rule": "3 * voxel size", "min_voxel_neighbors": 3,
                  "component_rule": "most original seed points within original eps 0.8; tie by dense point count"}
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing["provenance"] != provenance:
            raise ValueError("Grouping implementation/settings changed; this bounded run is immutable")
        return {**existing, "cache_hit": True}
    if (folder / "attempt_started.json").exists():
        raise RuntimeError("One grouping attempt was already started here; no automatic search/retry")
    write_json(folder / "attempt_started.json", provenance)
    started = time.perf_counter()
    data = load_crop(crop)
    spacing = nn_spacing(data["xyz"])
    voxel = float(np.clip(2*spacing["median"], .06, .18))
    eps = 3*voxel
    header = laspy.LasHeader(point_format=raw_manifest["point_format"], version=raw_manifest["las_version"])
    header.scales, header.offsets = np.array(raw_manifest["scales"]), np.array(raw_manifest["offsets"])
    mask, csf_metadata = csf_ground_mask(data["records"], header, **params)
    eligible = np.flatnonzero(~mask)
    if len(eligible) < 3:
        raise RuntimeError("Local CSF left fewer than 3 non-ground points; no dense candidate proposed")
    xyz = data["xyz"][eligible]
    cells = np.floor((xyz-xyz.min(axis=0))/voxel).astype(np.int64)
    _, representatives, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    if len(representatives) > 80_000:
        raise RuntimeError("Voxel grouping exceeds the bounded 80,000-node cap; no extra search")
    voxel_labels = objects._dbscan(xyz[representatives], eps, 3)
    labels = voxel_labels[inverse]
    distance, nearest = cKDTree(xyz).query(sparse["xyz"], k=1, workers=1)
    seed_labels = labels[nearest]
    rows = []
    for label in np.unique(labels[labels >= 0]):
        rows.append({"component": int(label), "points": int(np.sum(labels == label)),
                     "original_seed_support": int(np.sum((seed_labels == label) & (distance <= .8)))})
    rows.sort(key=lambda row: (-row["original_seed_support"], -row["points"], row["component"]))
    if not rows or rows[0]["original_seed_support"] == 0:
        raise RuntimeError("No component overlaps the original seed; one attempt ended without a candidate")
    chosen = rows[0]
    selected = eligible[labels == chosen["component"]]
    low, high = data["xyz"][selected].min(axis=0), data["xyz"][selected].max(axis=0)
    box_low, box_high = np.array(crop["bounds"]["min"]), np.array(crop["bounds"]["max"])
    touches = bool(np.any(low[:2]-box_low[:2] <= eps) or np.any(box_high[:2]-high[:2] <= eps))
    membership_path = folder / "membership.npz"
    np.savez_compressed(membership_path, ground_mask=mask, eligible_crop_indices=eligible,
                        component_labels=labels, selected_crop_indices=selected,
                        selected_raw_record_indices=data["raw_record_indices"][selected])
    manifest = {"provenance": provenance, "candidate_id": sparse["id"]+"_dense",
                "parent_candidate_id": sparse["id"], "crop_path": crop["path"],
                "membership_path": str(membership_path), "crop_points": len(data["xyz"]),
                "sparse_points": len(sparse["xyz"]), "dense_candidate_points": len(selected),
                "local_csf_ground_points": int(mask.sum()), "local_csf_non_ground_points": len(eligible),
                "voxel_size": voxel, "neighborhood_distance": eps, "voxel_representatives": len(representatives),
                "original_seed_support": chosen["original_seed_support"],
                "seed_support_fraction": chosen["original_seed_support"]/len(sparse["xyz"]),
                "components": rows, "component_count": len(rows),
                "dbscan_noise_points": int(np.sum(labels < 0)), "touches_crop_boundary": touches,
                "dense_bounds": {"min": low.tolist(), "max": high.tolist()},
                "sparse_spacing": nn_spacing(sparse["xyz"]), "crop_spacing": spacing,
                "dense_spacing": nn_spacing(data["xyz"][selected]),
                "csf_metadata": csf_metadata, "runtime_seconds": time.perf_counter()-started,
                "attempts": 1, "membership_sha256": objects._sha_file(membership_path),
                "note": "Density component, not a verified object. Local CSF context differs from the saved sample; merged objects, changed ground decisions, and crop-edge truncation remain possible."}
    write_json(manifest_path, manifest)
    return {**manifest, "cache_hit": False}


def comparison_inputs(sparse, grouping):
    """Return real RGB sparse/component/context arrays for the separate renderer."""
    data = load_crop({"path": grouping["crop_path"]})
    with np.load(grouping["membership_path"], allow_pickle=False) as arrays:
        indices = arrays["selected_crop_indices"]
    dense = {"id": grouping["candidate_id"], "xyz": data["xyz"][indices], "rgb": data["rgb"][indices]}
    context = {"id": sparse["id"]+"_context", "xyz": data["xyz"], "rgb": data["rgb"]}
    return sparse, dense, context


def load_report(run_dir):
    """Lightweight notebook entry: saved artifacts only; no raw scans or inference."""
    run_dir = Path(run_dir)
    report = read_json(run_dir / "report.json")
    for item in report["comparisons"]:
        if not Path(item["figure_path"]).is_file():
            raise FileNotFoundError(item["figure_path"])
    return report


def show_comparison(report, index=0):
    """Display saved comparison, point counts and candidate/context scores separately."""
    from IPython.display import Image, display
    from ground_experiment_views import show_table
    item = report["comparisons"][index]
    print(item["parent_candidate_id"], "|", item["selection_reason"])
    display(Image(filename=item["figure_path"]))
    show_table([item["counts_and_source"]])
    for role in ("sparse", "dense", "context"):
        prediction = read_json(item["predictions"][role])
        print(role.upper(), "— context is a whole region, not an object prediction" if role == "context" else "— candidate-only similarity")
        show_table(prediction["top5"])
        show_table([{"view": view["view_label"], "top1": view["top1"], "top3": view["top3"]} for view in prediction["per_view"]])
        print("View disagreement:", prediction["top1_disagreement_fraction"], "| similarities, not probabilities")
