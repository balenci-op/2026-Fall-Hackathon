"""Prepare and read a small review batch from ALREADY CACHED dense components.

No function here reads raw LAS, filters ground, clusters points, invokes CLIP, or
computes a geometry feature pipeline. Existing component labels supply membership.
New IDs depend on raw-source identity and sorted raw-record rows, so equal point
memberships retain one ID across crop caches. Existing sparse IDs are provenance,
not aliases for human-label transfer onto different dense memberships.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from object_candidates import _atomic_json, _plain, _sha_file


SCHEMA_VERSION = 1


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _latest_report(root, kind, filename="report.json"):
    paths = sorted((root / "outputs/objects" / kind).glob("*/" + filename))
    if not paths:
        raise FileNotFoundError(f"No saved {kind}/{filename} report exists")
    return paths[-1]


@lru_cache(maxsize=8)
def _crop_arrays(path, size, mtime_ns):
    # Stat arguments invalidate the process-local read cache if a file changes.
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("xyz", "rgb_raw", "raw_record_indices")}


@lru_cache(maxsize=8)
def _verify_crop(path, size, mtime_ns, expected_sha256):
    if _sha_file(path) != expected_sha256:
        raise ValueError(f"Cached crop no longer matches batch provenance: {path}")


def _crop(path, expected_sha256=None):
    path = Path(path).resolve()
    stat = path.stat()
    if expected_sha256:
        _verify_crop(str(path), stat.st_size, stat.st_mtime_ns, expected_sha256)
    return _crop_arrays(str(path), stat.st_size, stat.st_mtime_ns)


def _membership_id(source_identity, raw_rows):
    rows = np.unique(np.asarray(raw_rows, dtype="<i8"))
    digest = hashlib.sha256(source_identity.encode())
    digest.update(rows.tobytes())
    return "candidate_" + digest.hexdigest()[:24]


def _box_overlap(a, b, dimensions=3):
    return bool(np.all(np.minimum(a["max"][:dimensions], b["max"][:dimensions]) >=
                       np.maximum(a["min"][:dimensions], b["min"][:dimensions])))


def _bbox_hint(extent):
    """A selection hint from the required bounding box; never a class label."""
    spans = np.sort(np.asarray(extent))
    if spans[2] >= 2.5 * max(spans[1], 1e-12):
        return "elongated bounding box"
    if spans[1] >= 2.5 * max(spans[0], 1e-12):
        return "thin bounding box"
    return "compact or mixed bounding box"


def _collect(root):
    """Read the two completed comparison passes and their five cached crops."""
    quality_path = _latest_report(root, "quality")
    new_path = _latest_report(root, "new_candidates")
    feature_path = _latest_report(root, "geometry_vs_vision", "features_report.json")
    feature_report = _read(feature_path)
    feature_entries = {entry["id"]: entry for entry in feature_report["entries"]}
    observations = []
    # The two new structures are useful initial anchors; then the original three.
    for report_path, is_new in ((new_path, True), (quality_path, False)):
        report = _read(report_path)
        raw_path = Path(report["raw_manifest"] if is_new else report["raw_extraction_manifest"])
        raw = _read(raw_path)
        original_source_manifest = None
        if not is_new:
            audit = _read(report["selection"]["audit_path"])
            original_source_manifest = _read(Path(audit["source_cache"])/"manifest.json")
        raw_identity_fields = {key: raw[key] for key in (
            "raw_point_record_sha256", "raw_points_scanned", "scales", "offsets",
            "point_format", "las_version")}
        source_identity = _hash(raw_identity_fields)
        raw_source_path = str(Path(raw["settings"]["raw_path"]).resolve())
        crop_entries = {c["parent_candidate_id"]: c for c in raw["crops"]}
        items = list(report["items"] if is_new else report["comparisons"])
        if is_new:
            items.reverse()
        for item in items:
            parent_id = item["candidate_id"] if is_new else item["parent_candidate_id"]
            grouping_path = Path(item["grouping_manifest"])
            grouping = _read(grouping_path)
            crop = crop_entries[parent_id]
            if _sha_file(crop["path"]) != crop["sha256"]:
                raise ValueError(f"Cached crop hash mismatch: {crop['path']}")
            if _sha_file(grouping["membership_path"]) != grouping["membership_sha256"]:
                raise ValueError(f"Cached grouping hash mismatch: {grouping['membership_path']}")
            with np.load(grouping["membership_path"], allow_pickle=False) as data:
                membership = {key: data[key] for key in data.files}
            if is_new:
                roi = item["region_roi"]
                region_name = item["region"]
            else:
                roi = original_source_manifest["region"]["roi_xy"]
                region_name = "original comparison region"
            region_id = "region_" + _hash({"source_identity": source_identity,
                                           "roi_xy": roi})[:16]
            observations.append({
                "parent_id": parent_id, "grouping": grouping,
                "grouping_path": str(grouping_path.resolve()), "membership": membership,
                "crop": crop, "data": _crop(crop["path"]), "item": item,
                "report_path": str(report_path.resolve()), "raw_manifest_path": str(raw_path.resolve()),
                "raw_source_path": raw_source_path, "source_identity": source_identity,
                "raw_identity_fields": raw_identity_fields, "side": "left",
                "region_id": region_id, "region_name": region_name, "roi_xy": roi,
                "feature_reference": {"report_path": str(feature_path.resolve()),
                                      "entry_id": parent_id} if parent_id in feature_entries else None,
            })
    if len(observations) != 5:
        raise ValueError(f"Expected the five already computed parent crops; found {len(observations)}")
    return observations


def _component(observation, label, indices, is_parent):
    data = observation["data"]
    indices = np.asarray(indices, dtype=np.int64)
    points = data["xyz"][indices]
    raw_rows = data["raw_record_indices"][indices]
    lo, hi = points.min(axis=0), points.max(axis=0)
    bounds = observation["crop"]["bounds"]
    clearance = float(min(*(lo[:2] - np.asarray(bounds["min"])[:2]),
                          *(np.asarray(bounds["max"])[:2] - hi[:2])))
    return {"id": _membership_id(observation["source_identity"], raw_rows),
            "observation": observation, "component_label": int(label),
            "indices": indices, "raw_rows": np.sort(raw_rows), "point_count": len(indices),
            "bounds": {"min": lo.tolist(), "max": hi.tolist()},
            "extent": (hi-lo).tolist(), "clearance": clearance,
            "near_crop_boundary": clearance <= observation["grouping"]["neighborhood_distance"],
            "bbox_hint": _bbox_hint(hi-lo), "is_parent": is_parent, "aliases": []}


def _shared(a, b):
    if a["observation"]["source_identity"] != b["observation"]["source_identity"]:
        return 0
    if not _box_overlap(a["bounds"], b["bounds"]):
        return 0
    return int(np.intersect1d(a["raw_rows"], b["raw_rows"], assume_unique=True).size)


def _choose(observations, target_total, minimum_points):
    selected, pools, decisions = [], [], []
    for obs in observations:
        member = obs["membership"]
        chosen = member["selected_crop_indices"]
        eligible = member["eligible_crop_indices"]
        labels = member["component_labels"]
        selected_labels = np.unique(labels[np.isin(eligible, chosen)])
        if len(selected_labels) != 1:
            raise ValueError("Existing parent membership must be one saved component")
        anchor = _component(obs, selected_labels[0], chosen, True)
        selected.append(anchor)
        pool = []
        for row in obs["grouping"]["components"]:
            if row["component"] == selected_labels[0] or row["points"] < minimum_points:
                continue
            component = _component(obs, row["component"], eligible[labels == row["component"]], False)
            if component["near_crop_boundary"]:
                decisions.append({"legacy_crop_id": obs["parent_id"], "component": row["component"],
                                  "reason": "near crop boundary; kept in original grouping cache"})
                continue
            pool.append(component)
        pool.sort(key=lambda c: (-c["point_count"], -c["clearance"], c["component_label"]))
        # Alternate the simple bounding-box appearances before adding more of one.
        hints = list(dict.fromkeys(c["bbox_hint"] for c in pool))
        buckets = [[c for c in pool if c["bbox_hint"] == hint] for hint in hints]
        ordered = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    ordered.append(bucket.pop(0))
        pools.append(ordered)
    # Retain every existing anchor. They have disjoint memberships in these runs.
    if len({c["id"] for c in selected}) != len(selected):
        raise ValueError("Existing parents contain an exact duplicate; explicit alias review is needed")
    while len(selected) < target_total and any(pools):
        progress = False
        for pool in pools:
            if len(selected) >= target_total:
                break
            while pool:
                component = pool.pop(0)
                duplicate = None
                for other in selected:
                    overlap = _shared(component, other)
                    if overlap / min(component["point_count"], other["point_count"]) >= .80:
                        duplicate = (other, overlap)
                        break
                if duplicate:
                    other, overlap = duplicate
                    alias = {"crop_id": component["observation"]["parent_id"],
                             "component_label": component["component_label"],
                             "candidate_id": component["id"], "shared_raw_records": overlap,
                             "exact_membership": component["id"] == other["id"]}
                    other["aliases"].append(alias)
                    decisions.append({**alias, "retained_candidate_id": other["id"],
                                      "reason": "at least 80% of smaller membership overlaps an existing selection"})
                    continue
                selected.append(component)
                progress = True
                break
        if not progress:
            break
    return selected, decisions


def _groups(observations):
    parent = list(range(len(observations)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for i, a in enumerate(observations):
        for j in range(i+1, len(observations)):
            b = observations[j]
            if a["source_identity"] != b["source_identity"]:
                continue
            same_region = a["region_id"] == b["region_id"]
            crop_overlap = _box_overlap(a["crop"]["bounds"], b["crop"]["bounds"])
            if same_region or crop_overlap:
                parent[find(j)] = find(i)
                edges.append({"crop_a": a["parent_id"], "crop_b": b["parent_id"],
                              "same_original_region": same_region,
                              "overlapping_xyz_crop_boxes": crop_overlap,
                              "meaning": "conservative shared-observation grouping; not verified same object"})
    groups = {}
    for i, obs in enumerate(observations):
        component = [other["parent_id"] for j, other in enumerate(observations) if find(i) == find(j)]
        groups[obs["parent_id"]] = "leakage_" + _hash({"source_identity": obs["source_identity"],
                                                      "crop_ids": sorted(component)})[:16]
    return groups, edges


def _existing_views(obs, role):
    path = Path(obs["item"]["render_manifest"])
    manifest = _read(path)
    rows = []
    for i, view in enumerate(manifest["settings"]["views"]):
        relative = f"{role}/view_{i:02d}.png"
        image_path = path.parent / relative
        expected = manifest["file_sha256"][relative]
        if _sha_file(image_path) != expected:
            raise ValueError(f"Existing view hash mismatch: {image_path}")
        rows.append({"name": view["label"], "path": str(image_path.resolve()),
                     "sha256": expected, "reused": True})
    return rows


def _new_views(component, folder):
    import object_views
    from object_quality_views import _shared_projection, _projected_image
    obs = component["observation"]
    points = obs["data"]["xyz"][component["indices"]]
    colors = obs["data"]["rgb_raw"][component["indices"]].astype(float) / 65535
    origin = (points.min(axis=0)+points.max(axis=0))/2
    settings = object_views._render_settings(None)
    rows = []
    destination = folder / "views" / component["id"]
    destination.mkdir(parents=True, exist_ok=False)
    for i, view in enumerate(settings["views"]):
        projection = _shared_projection({"candidate": {"xyz": points}}, origin, view, settings)
        image, metrics = _projected_image(points, colors, origin, projection, settings)
        path = destination / f"view_{i:02d}.png"
        image.save(path, format="PNG")
        rows.append({"name": view["label"], "path": str(path), "sha256": _sha_file(path),
                     "reused": False, "rendered_points": metrics["rendered_input_count"]})
    return rows


def _locator(component, folder):
    """Human-only candidate locator. It is never passed to a vision model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    obs = component["observation"]
    xyz, rgb = obs["data"]["xyz"], obs["data"]["rgb_raw"].astype(float)/65535
    ix = component["indices"]
    origin = np.asarray(component["bounds"]["min"])
    local = xyz-origin
    context = np.linspace(0, len(xyz)-1, min(12000, len(xyz)), dtype=int)
    shown = ix[np.linspace(0, len(ix)-1, min(8000, len(ix)), dtype=int)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    lo, hi = np.zeros(3), np.asarray(component["bounds"]["max"])-origin
    for ax, h, v, title in ((axes[0],0,1,"Top context"),(axes[1],0,2,"Elevation context")):
        ax.scatter(local[context,h],local[context,v],c=rgb[context],s=2,alpha=.25,linewidths=0)
        ax.scatter(local[shown,h],local[shown,v],c=rgb[shown],s=4,linewidths=0)
        ax.add_patch(Rectangle((lo[h],lo[v]),hi[h],hi[v],fill=False,color="#db2777",linewidth=1.3))
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("XYZ"[h]+" relative to candidate min [source units]",fontsize=8)
        ax.set_ylabel("XYZ"[v]+" [source units]",fontsize=8)
        ax.set_aspect("equal",adjustable="datalim")
        ax.tick_params(labelsize=8)
    fig.suptitle("Outlined candidate in its original dense crop; units unresolved",fontsize=11)
    fig.tight_layout()
    path = folder / "locators" / (component["id"]+".png")
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path,dpi=120)
    plt.close(fig)
    return str(path)


def prepare_batch(project_dir, target_total=20, minimum_points=100):
    """Create one immutable batch and point the UI at it, using only saved caches.

    Five prior dense memberships remain anchors. Interior additional groups are
    selected round-robin across the five crop observations, alternating simple
    bounding-box appearances and rejecting high shared-row membership. This is
    review triage, not a semantic classifier or a completeness assessment.
    """
    if not 5 <= target_total <= 30 or minimum_points < 100:
        raise ValueError("Use 5..30 total candidates and at least 100 points per additional group")
    root = Path(project_dir).resolve()
    observations = _collect(root)
    selected, decisions = _choose(observations, target_total, minimum_points)
    group_ids, overlap_edges = _groups(observations)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = root / "outputs/objects/review_batches" / (stamp+"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"memberships").mkdir()
    rows = []
    for index, component in enumerate(selected):
        obs = component["observation"]
        membership_path = folder / "memberships" / (component["id"]+".npz")
        np.savez_compressed(membership_path, crop_indices=component["indices"],
                            raw_record_indices=obs["data"]["raw_record_indices"][component["indices"]])
        static_views = (_existing_views(obs,"dense") if component["is_parent"] else
                        _new_views(component,folder))
        predictions = obs["item"].get("predictions",{}) if component["is_parent"] else {}
        prediction_paths = {"candidate": predictions.get("dense"),
                            "dense": predictions.get("dense"), "context": predictions.get("context")}
        possible, overlap_rows = [], []
        for other in selected:
            if other["id"] == component["id"]:
                continue
            if obs["source_identity"] != other["observation"]["source_identity"]:
                continue
            common = _shared(component,other)
            if common or _box_overlap(component["bounds"],other["bounds"]):
                possible.append(other["id"])
                overlap_rows.append({"candidate_id":other["id"],"shared_raw_records":common,
                    "shared_fraction_of_smaller":common/min(component["point_count"],other["point_count"]),
                    "reason":"shared raw records" if common else "overlapping candidate boxes only; same object unverified"})
        locator_path = _locator(component,folder)
        rows.append({
            "id":component["id"],"candidate_id":component["id"],"index":index,
            "original_parent_id":obs["parent_id"],"original_sparse_candidate_id":obs["parent_id"],
            "legacy_ids":[obs["grouping"]["candidate_id"]] if component["is_parent"] else [],
            "legacy_mapping_note":"Sparse parent ID is provenance only. Dense/sparse memberships differ; no human labels are transferred.",
            "component_label":component["component_label"],"is_existing_parent":component["is_parent"],
            "point_count":component["point_count"],"bounds":component["bounds"],"extent":component["extent"],
            "near_crop_boundary":component["near_crop_boundary"],"crop_edge_clearance_xy":component["clearance"],
            "bbox_selection_hint":component["bbox_hint"],"source_identity":obs["source_identity"],
            "raw_source_path":obs["raw_source_path"],"raw_source_filename":Path(obs["raw_source_path"]).name,
            "source_side":obs["side"],"raw_source_identity":obs["raw_identity_fields"],
            "raw_manifest_path":obs["raw_manifest_path"],"region_id":obs["region_id"],
            "region_name":obs["region_name"],"region_roi_xy":obs["roi_xy"],
            "leakage_group_id":group_ids[obs["parent_id"]],"crop_observation_id":obs["parent_id"],
            "crop_path":obs["crop"]["path"],"crop_sha256":obs["crop"]["sha256"],
            "crop_bounds":obs["crop"]["bounds"],"crop_point_count":obs["crop"]["point_count"],
            "membership_path":str(membership_path),"membership_sha256":_sha_file(membership_path),
            "grouping_manifest_path":obs["grouping_path"],"comparison_report_path":obs["report_path"],
            "static_views":static_views,"context_views":_existing_views(obs,"context"),
            "locator_path":locator_path,"locator_image":locator_path,"predictions":prediction_paths,
            "prediction_path":prediction_paths["candidate"],
            "feature_reference":obs["feature_reference"] if component["is_parent"] else None,
            "possible_duplicate_ids":sorted(possible),"overlap_evidence":overlap_rows,
            "deduplicated_observations":component["aliases"],
            "coordinate_units":"unresolved source coordinate units (LAS: US survey foot; sidecar: meter)",
            "candidate_note":"Cached spatial component; may be incomplete, merged, residual ground or unclear. No human or semantic label assigned.",
            "display_note":"Four original-RGB point views. Existing anchors retain their saved common-context frame; additional groups use a candidate-fit frame. Context and locator views are separate.",
        })
    region_rows = {}
    for obs in observations:
        region_rows[obs["region_id"]] = {"id":obs["region_id"],"name":obs["region_name"],
            "source_identity":obs["source_identity"],"side":obs["side"],"roi_xy":obs["roi_xy"],
            "leakage_group_id":group_ids[obs["parent_id"]]}
    manifest = {"schema_version":SCHEMA_VERSION,"batch_id":folder.name,
        "batch_path":str(folder/"batch.json"),"created_utc":datetime.now(timezone.utc).isoformat(),
        "project_dir":str(root),"candidates":rows,"regions":list(region_rows.values()),
        "counts":{"candidates":len(rows),"existing_dense_parents":sum(r["is_existing_parent"] for r in rows),
                  "additional_cached_components":sum(not r["is_existing_parent"] for r in rows),
                  "distinct_regions":len(region_rows),"leakage_groups":len(set(group_ids.values()))},
        "selection_policy":{"target_total":target_total,"minimum_additional_points":minimum_points,
            "interior_margin":"more than that crop's saved DBSCAN neighborhood distance",
            "diversity":"round-robin crop observations, then simple bbox span appearances and point counts",
            "duplicate_rule":"reject additional membership if at least 80% of its smaller pair membership is shared with an already selected group",
            "no_new_features":True,"no_new_clustering":True,"no_new_ground_filtering":True,
            "raw_las_reads":False,"new_clip_inference":False},
        "dedup_and_rejection_log":decisions,"crop_overlap_edges":overlap_edges,
        "grouping_policy":{"meaning":"Conservative groups for future evaluation splits, not object identity.",
            "union_rule":"same original region or overlapping XYZ crop boxes from the same raw-source observation; shared raw records also imply overlap",
            "same_crop":"all candidate components from one crop share a group even when their memberships are disjoint",
            "cross_scan":"This batch uses LEFT only. Future left/right or overlapping scans must not be treated as independent examples without registration/location matching. Keep potential same-location observations in one split group until checked; coordinate coincidence alone is not a verified object match."},
        "feature_policy":"Only existing feature-cache references for five parents; none computed for additions.",
        "prediction_policy":"Only existing candidate/context predictions for five parents; additions explicitly have no predictions.",
        "review_policy":"No human labels or review decisions are stored by batch preparation.",
        "implementation_sha256":_sha_file(__file__)}
    _atomic_json(folder/"batch.json",manifest)
    _atomic_json(root/"outputs/objects/review_batches/active_batch.json",{"batch_path":str(folder/"batch.json")})
    return manifest


def load_batch(path):
    """Read a batch JSON, batch directory, or active_batch.json pointer."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path/"batch.json"
    data = _read(path)
    if "batch_path" in data and "candidates" not in data:
        path = Path(data["batch_path"])
        data = _read(path)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported review batch schema")
    ids = [row["id"] for row in data["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Batch contains duplicate candidate IDs")
    return data


def candidate_record(batch, candidate_id):
    """Resolve a stable ID or an unambiguous old dense-membership alias."""
    batch = load_batch(batch) if not isinstance(batch,dict) else batch
    matches = [row for row in batch["candidates"]
               if candidate_id == row["id"] or candidate_id in row["legacy_ids"]]
    if len(matches) != 1:
        raise KeyError(f"Unknown or ambiguous candidate ID: {candidate_id}")
    return matches[0]


def load_candidate_arrays(batch, candidate_id, context=True):
    """Read cached membership/XYZ/RGB for a UI, without inference or processing.

    Full candidate membership is returned. A UI may display a deterministic
    subset separately. Context is the original dense crop, including ground and
    unselected components, with preserved original source XYZ and 16-bit RGB.
    """
    row = candidate_record(batch,candidate_id)
    data = _crop(row["crop_path"], row["crop_sha256"])
    if _sha_file(row["membership_path"]) != row["membership_sha256"]:
        raise ValueError("Review batch membership file changed")
    with np.load(row["membership_path"],allow_pickle=False) as member:
        indices = member["crop_indices"]
        raw_rows = member["raw_record_indices"]
    if not np.array_equal(data["raw_record_indices"][indices],raw_rows):
        raise ValueError("Candidate membership does not match its crop cache")
    if _membership_id(row["source_identity"],raw_rows) != row["id"]:
        raise ValueError("Candidate stable ID does not match its source/membership")
    result = {"id":row["id"],"record":row,"xyz":data["xyz"][indices],
        "rgb":data["rgb_raw"][indices].astype(float)/65535,"rgb_raw":data["rgb_raw"][indices],
        "raw_record_indices":raw_rows,"crop_indices":indices,
        "origin":np.asarray(row["bounds"]["min"],dtype=float)}
    if context:
        result.update(context_xyz=data["xyz"],context_rgb=data["rgb_raw"].astype(float)/65535,
                      context_rgb_raw=data["rgb_raw"],context_raw_record_indices=data["raw_record_indices"])
    return result
