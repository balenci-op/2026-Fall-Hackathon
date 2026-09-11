"""Cached candidate extraction and durable reviews for ``object_review.ipynb``.

The only processing input is a bounded region of an existing development sample,
joined to saved CSF partitions with ``ground_experiment_tools``' exact LAS-record
check. No raw survey file is opened. Coordinates and RGB attributes are preserved;
all distances are in unresolved source coordinate units. A cluster is a candidate,
not an assertion that its points form one complete object.

Caches are content addressed. Full SHA256s of the development sample, both saved
partitions and their manifests detect changed inputs even when filenames persist.
Membership stores zero-based region and original development-sample record rows.
Human labels live in separate, atomically replaced review documents with history.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import laspy
import numpy as np

import ground_experiment_tools as ground


SOURCE_VERSION = "1"
CLUSTER_VERSION = "dbscan-kdtree-1"
MAX_CLUSTER_POINTS = 50_000
MAX_NEIGHBOR_VISITS = 20_000_000
DEFAULT_ROI_XY = [2_454_000.0, 2_454_100.0, 415_300.0, 415_400.0]
CROP_QUALITIES = (
    "complete object", "incomplete object", "multiple objects",
    "ground contamination", "unclear",
)


def _plain(value):
    """Convert metadata to portable JSON; point arrays are saved separately."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    """Flush a complete JSON document before replacing the previous version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(_plain(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_rows(sample_path, context_roi_xy):
    """Recover original sample row numbers using the identical ordered XY crop."""
    xmin, xmax, ymin, ymax = context_roi_xy
    pieces, offset = [], 0
    with laspy.open(sample_path) as reader:
        for points in reader.chunk_iterator(ground.CHUNK_POINTS):
            x, y = np.asarray(points.x), np.asarray(points.y)
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            pieces.append(np.flatnonzero(mask).astype(np.int64) + offset)
            offset += len(points)
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def load_source(project_dir, side="left", roi_xy=None, padding=8.0,
                csf_run="csf_current"):
    """Load a manageable saved CSF non-ground region and original RGB context.

    ``roi_xy`` uses absolute [xmin, xmax, ymin, ymax]. ``padding`` adds context
    from the colored development sample; clustering only uses the core rectangle.
    Context includes ground and non-ground points. The first load verifies exact
    record membership and saves a source cache; later loads reuse that cache after
    rehashing the actual input files. Never pass a raw LAS path here.
    """
    started = time.perf_counter()
    root = Path(project_dir).resolve()
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if csf_run not in ("csf_current", "csf_previous"):
        raise ValueError("csf_run must be 'csf_current' or 'csf_previous'")
    roi = [float(value) for value in (DEFAULT_ROI_XY if roi_xy is None else roi_xy)]
    catalog = ground.load_catalog(root)
    sample = catalog["samples"][side]
    run = catalog["runs"][csf_run]
    if "error" in run:
        raise FileNotFoundError(run["error"])
    paths = {
        "development_sample": sample["path"],
        "input_manifest": catalog["input_manifest_path"],
        "csf_manifest": run["manifest_path"],
        "saved_ground": run["files"][side]["ground"],
        "saved_non_ground": run["files"][side]["non_ground"],
    }
    fingerprints = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} is missing: {path}")
        fingerprints[name] = {"path": str(path), "sha256": _sha_file(path),
                              "bytes": path.stat().st_size}
    source_identity = {
        "source_version": SOURCE_VERSION, "side": side, "roi_xy": roi,
        "padding": float(padding), "csf_run": csf_run,
        "coordinate_units": ground.UNITS, "files": fingerprints,
        "ground_loader_sha256": _sha_file(Path(ground.__file__)),
    }
    identity = _key(source_identity)
    cache_dir = root / "outputs/objects/sources" / identity
    manifest_path = cache_dir / "manifest.json"
    cache_hit = manifest_path.is_file() and (cache_dir / "source.npz").is_file()
    if cache_hit:
        metadata = _read_json(manifest_path)
        if metadata["identity"] != identity:
            raise ValueError(f"Source cache identity mismatch: {cache_dir}")
        with np.load(cache_dir / "source.npz", allow_pickle=False) as arrays:
            records = arrays["records"]
            ground_mask = arrays["ground_mask"]
            sample_rows = arrays["sample_record_indices"]
        with laspy.open(sample["path"]) as reader:
            header = reader.header.copy()
        region = dict(metadata["region"])
        region.update({"records": records, "header": header,
                       "source_path": sample["path"], "source_info": sample["source_info"],
                       "input_manifest": catalog["input_manifest"],
                       "origin": np.asarray(region["origin"], dtype=float)})
        region["xyz"] = ground._xyz(records, header)
        region["evaluation_mask"] = ground._xy_mask(region["xyz"], region["roi_xy"])
        saved_result = dict(metadata["saved_result"])
        saved_result.update({"ground_mask": ground_mask,
                             "coverage_mask": np.ones(len(records), dtype=bool)})
    else:
        region = ground.load_region(catalog, side, roi, padding)
        saved_result = ground.load_saved_result(catalog, region, csf_run)
        records = region["records"]
        ground_mask = saved_result["ground_mask"]
        sample_rows = _record_rows(sample["path"], region["context_roi_xy"])
        if len(sample_rows) != len(records):
            raise ValueError("Source sample changed while loading; reload before clustering")
        cache_dir.mkdir(parents=True, exist_ok=True)
        region_metadata = {key: value for key, value in region.items()
                           if key not in {"records", "xyz", "header", "evaluation_mask",
                                          "input_manifest", "source_info"}}
        metadata = {"identity": identity, "source_identity": source_identity,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "region": region_metadata,
                    "saved_result": {key: value for key, value in saved_result.items()
                                     if key not in {"ground_mask", "coverage_mask"}},
                    "membership": "source.npz records retain full LAS attributes; sample_record_indices are zero-based rows in the hashed development sample",
                    "load_runtime_seconds": time.perf_counter() - started}
        np.savez_compressed(cache_dir / "source.npz", records=records,
                            ground_mask=ground_mask, sample_record_indices=sample_rows)
        _atomic_json(manifest_path, metadata)
    rgb_fields = {"red", "green", "blue"}
    if not rgb_fields.issubset(records.dtype.names or ()):
        raise ValueError("The selected saved sample has no original RGB dimensions")
    rgb_raw = np.column_stack([records[name] for name in ("red", "green", "blue")])
    rgb = rgb_raw.astype(np.float64) / 65535.0
    eligible_mask = region["evaluation_mask"] & ~ground_mask
    if int(eligible_mask.sum()) > MAX_CLUSTER_POINTS:
        raise ValueError(f"Region contains {eligible_mask.sum():,} non-ground processing points; shrink the region below {MAX_CLUSTER_POINTS:,}. No display sampling is used for clustering.")
    return {"project_dir": root, "catalog": catalog, "region": region,
            "saved_result": saved_result, "xyz": region["xyz"], "rgb": rgb,
            "rgb_raw": rgb_raw, "rgb_scale": 65535,
            "eligible_mask": eligible_mask, "sample_record_indices": sample_rows,
            "identity": identity, "source_identity": source_identity,
            "cache_dir": cache_dir, "cache_hit": cache_hit,
            "load_runtime_seconds": time.perf_counter() - started,
            "coordinate_units": ground.UNITS}


def spacing_report(source):
    """Summarize exact 3D nearest-neighbor spacing before choosing DBSCAN eps.

    Includes self-excluded nearest-neighbor zeros (coincident XYZ records) and
    positive nearest-neighbor quantiles. The suggested eps is only an exploratory
    scale: twice the median positive spacing, not an estimate of object size.
    """
    from scipy.spatial import cKDTree

    xyz = source["xyz"][source["eligible_mask"]]
    report = {"non_ground_processing_points": len(xyz),
              "original_context_points": len(source["xyz"]),
              "original_core_points": int(source["region"]["evaluation_mask"].sum()),
              "coordinate_units": ground.UNITS,
              "header_axis_units": source["catalog"]["samples"][source["region"]["side"]]["header"]["axis_units"],
              "coordinate_scales": source["region"]["header"].scales.tolist(),
              "duplicate_xyz_extra_records": int(len(xyz) - len(np.unique(xyz, axis=0)))}
    if len(xyz) < 2:
        report.update({"nearest_neighbor_quantiles": {}, "positive_nn_quantiles": {},
                       "zero_nn_records": 0, "suggested_eps": None,
                       "note": "Choose a populated region: at least two non-ground points are required to inspect spacing."})
        return report
    nn = cKDTree(xyz).query(xyz, k=2, workers=1)[0][:, 1]
    positive = nn[nn > 0]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    report.update({
        "nearest_neighbor_quantiles": {f"p{int(q * 100)}": float(np.quantile(nn, q)) for q in quantiles},
        "positive_nn_quantiles": {f"p{int(q * 100)}": float(np.quantile(positive, q)) for q in quantiles} if len(positive) else {},
        "zero_nn_records": int((nn == 0).sum()),
        "suggested_eps": float(2 * np.median(positive)) if len(positive) else None,
        "note": "3D Euclidean, same unsampled CSF non-ground region used for clustering. Two times median positive spacing is a starting scale only; inspect merges, fragments and noise."})
    return report


def _dbscan(xyz, eps, min_samples):
    """Deterministic DBSCAN with one neighborhood at a time, no dense matrix.

    Neighborhood counts include each point itself. Each core point is expanded
    once; border points are assigned to the first reachable cluster in source
    record order. A work guard stops impractically dense eps choices explicitly.
    """
    from scipy.spatial import cKDTree

    n = len(xyz)
    labels = np.full(n, -1, dtype=np.int32)
    if n < min_samples:
        return labels
    tree = cKDTree(xyz)
    kth = tree.query(xyz, k=[min_samples], workers=1)[0][:, 0]
    core = kth <= eps
    expanded = np.zeros(n, dtype=bool)
    cluster, visits = 0, 0
    for seed in range(n):
        if not core[seed] or labels[seed] >= 0:
            continue
        labels[seed] = cluster
        pending = deque([seed])
        while pending:
            current = pending.popleft()
            if expanded[current]:
                continue
            expanded[current] = True
            neighbors = np.asarray(tree.query_ball_point(xyz[current], eps), dtype=np.int64)
            visits += len(neighbors)
            if visits > MAX_NEIGHBOR_VISITS:
                raise ValueError("DBSCAN exceeded the local neighborhood work limit. Reduce neighborhood distance or region size; no partial candidate cache was saved.")
            unassigned = neighbors[labels[neighbors] == -1]
            labels[unassigned] = cluster
            pending.extend(unassigned[core[unassigned]].tolist())
        cluster += 1
    return labels


def extract_candidates(source, eps, min_cluster_points, min_samples=5,
                       max_candidates=8):
    """Cluster all core non-ground points; cache membership and retain noise.

    ``eps`` is the 3D neighborhood distance in source coordinate units.
    ``min_cluster_points`` rejects tiny DBSCAN groups after density clustering;
    ``min_samples`` defines a dense neighborhood (including the point itself).
    ``max_candidates`` limits the review batch only, never processing or noise.
    Larger retained clusters are presented first; every group remains in the
    membership file and table, including groups outside this review batch.
    """
    started = time.perf_counter()
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite distance in source coordinate units")
    for name, value, maximum in (("min_cluster_points", min_cluster_points, MAX_CLUSTER_POINTS),
                                  ("min_samples", min_samples, MAX_CLUSTER_POINTS),
                                  ("max_candidates", max_candidates, 30)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    eligible_indices = np.flatnonzero(source["eligible_mask"]).astype(np.int64)
    if len(eligible_indices) > MAX_CLUSTER_POINTS:
        raise ValueError(f"Reduce the region below {MAX_CLUSTER_POINTS:,} non-ground points")
    params = {"method": "3D DBSCAN via scipy cKDTree", "eps": float(eps),
              "min_samples": int(min_samples), "min_cluster_points": int(min_cluster_points),
              "max_review_candidates": int(max_candidates), "version": CLUSTER_VERSION,
              "distance_units": ground.UNITS}
    cache_key = _key({"source_identity": source["identity"], "clustering": params})
    cache_dir = source["project_dir"] / "outputs/objects/candidates" / cache_key
    manifest_path = cache_dir / "manifest.json"
    cache_hit = manifest_path.is_file() and (cache_dir / "membership.npz").is_file()
    if cache_hit:
        manifest = _read_json(manifest_path)
        if manifest["cache_key"] != cache_key:
            raise ValueError(f"Candidate cache identity mismatch: {cache_dir}")
        with np.load(cache_dir / "membership.npz", allow_pickle=False) as data:
            labels, raw_labels = data["labels"], data["raw_labels"]
            if not np.array_equal(data["eligible_region_indices"], eligible_indices):
                raise ValueError("Cached candidate membership no longer matches its source")
    else:
        xyz = source["xyz"][eligible_indices]
        raw_labels = _dbscan(xyz, eps, min_samples)
        labels = raw_labels.copy()
        cluster_labels, counts = np.unique(raw_labels[raw_labels >= 0], return_counts=True)
        all_groups = sorted(zip(cluster_labels.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
        rows, review_count = [], 0
        roi = source["region"]["roi_xy"]
        for label, count in all_groups:
            selected = xyz[raw_labels == label]
            lo, hi = selected.min(axis=0), selected.max(axis=0)
            tiny = count < min_cluster_points
            if tiny:
                labels[raw_labels == label] = -1
            in_batch = not tiny and review_count < max_candidates
            candidate_id = f"{cache_key[:16]}_{label:05d}"
            boundary = bool(lo[0] - roi[0] <= eps or roi[1] - hi[0] <= eps or
                            lo[1] - roi[2] <= eps or roi[3] - hi[1] <= eps)
            rows.append({"candidate_id": candidate_id, "cluster_label": int(label),
                         "point_count": int(count), "bounds": {"min": lo.tolist(), "max": hi.tolist()},
                         "extent": (hi - lo).tolist(), "touches_region_boundary": boundary,
                         "status": "below_min_cluster_points" if tiny else "in_review_batch" if in_batch else "outside_review_batch",
                         "candidate_index": review_count if in_batch else None})
            if in_batch:
                review_count += 1
        manifest = {"schema_version": 1, "cache_key": cache_key,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_identity": source["source_identity"],
                    "source_cache": str(source["cache_dir"]),
                    "source_record_fingerprint": source["region"]["fingerprint"],
                    "same_input_evidence": source["saved_result"]["source_note"],
                    "clustering": params, "cluster_table": rows,
                    "counts": {"processing_non_ground": len(eligible_indices),
                               "dbscan_noise": int((raw_labels < 0).sum()),
                               "below_min_cluster_points": int(((labels < 0) & (raw_labels >= 0)).sum()),
                               "rejected_or_noise": int((labels < 0).sum()),
                               "retained_cluster_points": int((labels >= 0).sum()),
                               "raw_clusters": len(rows), "review_batch_candidates": review_count,
                               "retained_clusters": sum(row["status"] != "below_min_cluster_points" for row in rows)},
                    "runtime_seconds": time.perf_counter() - started,
                    "membership_note": "labels/raw_labels align eligible_region_indices. region indices address the exact source cache records; sample_record_indices address original development LAS rows. -1 means noise or rejected tiny groups in labels; raw_labels retains tiny group membership.",
                    "boundary_note": "touches_region_boundary means a bounding-box edge is within eps of the selected region edge; it warns about possible cropping, not measured completeness.",
                    "display_sampling": "none in extraction; render/display subsets are separate",
                    "candidate_note": "Density groups can be object fragments, merged objects, vegetation or residual ground; no ground truth is asserted."}
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_dir / "membership.npz", labels=labels,
                            raw_labels=raw_labels, eligible_region_indices=eligible_indices,
                            sample_record_indices=source["sample_record_indices"][eligible_indices])
        _atomic_json(manifest_path, manifest)
    candidates = []
    for row in manifest["cluster_table"]:
        if row["status"] != "in_review_batch":
            continue
        indices = eligible_indices[labels == row["cluster_label"]]
        candidates.append({**row, "id": row["candidate_id"], "indices": indices,
                           "xyz": source["xyz"][indices], "rgb": source["rgb"][indices],
                           "sample_record_indices": source["sample_record_indices"][indices],
                           "source_identity": source["identity"], "batch_cache_key": cache_key})
    return {"source": source, "candidates": candidates, "labels": labels,
            "raw_labels": raw_labels, "eligible_indices": eligible_indices,
            "noise_indices": eligible_indices[labels < 0],
            "cluster_table": manifest["cluster_table"], "cache_dir": cache_dir,
            "cache_key": cache_key, "manifest": manifest, "cache_hit": cache_hit,
            "runtime_seconds": time.perf_counter() - started}


def _review_path(batch, candidate, review_root=None):
    candidate_id = candidate["id"]
    valid = {row["candidate_id"] for row in batch["cluster_table"]}
    if candidate_id not in valid or not re.fullmatch(r"[0-9a-f]{16}_[0-9]{5,}", candidate_id):
        raise ValueError("Candidate does not belong to this extraction batch")
    root = (Path(review_root) if review_root is not None else
            batch["source"]["project_dir"] / "outputs/objects/reviews")
    return root / batch["cache_key"] / f"{candidate_id}.json"


def load_review(batch, candidate, review_root=None):
    """Return the latest saved human review, or None for an unreviewed candidate."""
    path = _review_path(batch, candidate, review_root)
    if not path.is_file():
        return None
    document = _read_json(path)
    if document["candidate_id"] != candidate["id"] or document["batch_cache_key"] != batch["cache_key"]:
        raise ValueError(f"Review provenance does not match candidate: {path}")
    return document["history"][-1] if document["history"] else None


def save_review(batch, candidate, inference_record, reviewed_label,
                crop_quality, notes="", decision="correct", review_root=None):
    """Save an explicit human review immediately, keeping prediction snapshots.

    ``decision`` is accept, correct, or unlabeled. Accept/correct require an
    explicit nonblank human label; no model guess is adopted automatically.
    A review never runs clustering, rendering, inference, or model training.
    ``review_root`` supports isolated persistence checks without fabricated labels
    in the real review dataset. Each save preserves earlier revisions in history.
    """
    if decision not in {"accept", "correct", "unlabeled"}:
        raise ValueError("decision must be 'accept', 'correct', or 'unlabeled'")
    if crop_quality not in CROP_QUALITIES:
        raise ValueError(f"crop_quality must be one of {CROP_QUALITIES}")
    if decision == "unlabeled":
        if reviewed_label not in (None, ""):
            raise ValueError("An unlabeled review must not contain a human label")
        reviewed_label = None
    elif not isinstance(reviewed_label, str) or not reviewed_label.strip():
        raise ValueError("Accept/correct requires an explicit nonblank reviewed_label")
    else:
        reviewed_label = reviewed_label.strip()
    if not isinstance(notes, str):
        raise ValueError("notes must be text")
    if not isinstance(inference_record, dict):
        raise ValueError("inference_record must be a real prediction record or an explicit unavailable-status record")
    if inference_record.get("candidate_id", candidate["id"]) != candidate["id"]:
        raise ValueError("Prediction record belongs to a different candidate")
    path = _review_path(batch, candidate, review_root)
    if path.is_file():
        document = _read_json(path)
        load_review(batch, candidate, review_root)  # validate before modifying
    else:
        document = {"schema_version": 1, "candidate_id": candidate["id"],
                    "batch_cache_key": batch["cache_key"], "history": []}
    record = {
        "candidate_id": candidate["id"], "batch_cache_key": batch["cache_key"],
        "revision": len(document["history"]) + 1,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "source_and_region": batch["manifest"]["source_identity"],
        "clustering": batch["manifest"]["clustering"],
        "membership_file": str(batch["cache_dir"] / "membership.npz"),
        "source_cache": str(batch["source"]["cache_dir"]),
        "candidate_point_count": int(candidate["point_count"]),
        "candidate_bounds": candidate["bounds"],
        "prediction_snapshot": _plain(inference_record),
        "human_review": {"decision": decision, "reviewed_label": reviewed_label,
                         "crop_quality": crop_quality, "notes": notes},
        "training_note": "Human review only; model weights were not changed.",
    }
    document["history"].append(record)
    _atomic_json(path, document)
    return record


def review_table(batch):
    """Small table for the selected review batch, leaving unreviewed labels null."""
    rows = []
    for candidate in batch["candidates"]:
        review = load_review(batch, candidate)
        human = review["human_review"] if review else {}
        rows.append({"candidate_index": candidate["candidate_index"],
                     "candidate_id": candidate["id"], "points": candidate["point_count"],
                     "reviewed": review is not None,
                     "reviewed_label": human.get("reviewed_label"),
                     "crop_quality": human.get("crop_quality"),
                     "notes": human.get("notes", "")})
    return rows


def review_summary(batch):
    """Alias used by notebook cells when displaying review progress."""
    return review_table(batch)
