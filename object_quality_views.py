"""Controlled sparse/dense/context RGB comparisons for candidate-quality review.

Every column in a given view uses the same origin, camera, image size, and
orthographic projection bounds. The wider context therefore makes candidates
smaller in all columns, rather than silently giving each one a different scale.
All supplied points are rendered with their original RGB (quantized to 8 bits).
No surfaces, textures, interpolated colors, class labels, or additional points
are generated. This module does not read LAS or invoke any model.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

import object_views
from object_views import UNIT_NOTE, _render_settings, _rgb, _sha_file


QUALITY_RENDER_VERSION = "1.0"
VARIANTS = ("sparse", "dense", "context")
DEFAULT_LABELS = {"sparse": "Original sparse candidate", "dense": "Denser regrouped candidate",
                  "context": "Wider dense crop context"}


def _input(group, role):
    points = np.asarray(group["xyz"], dtype=np.float64)
    colors = _rgb(group["rgb"])
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or points.shape != colors.shape:
        raise ValueError(f"{role}: XYZ/RGB must be nonempty matching (N, 3) arrays.")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{role}: XYZ coordinates must be finite.")
    identifier = str(group["id"])
    if not identifier:
        raise ValueError(f"{role}: id must not be empty.")
    return {"xyz": points, "rgb": colors, "id": identifier,
            "label": str(group.get("label", DEFAULT_LABELS[role])), "role": role}


def _array_hash(array):
    canonical = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256(str(canonical.shape).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _camera_basis(view):
    """Use the existing candidate renderer's numeric camera convention."""
    forward = np.asarray(view["camera"], dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.asarray(view["up"], dtype=float), forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.column_stack([right, up, forward])


def _shared_projection(groups, origin, view, settings):
    basis = _camera_basis(view)
    minimum = np.full(2, np.inf)
    maximum = np.full(2, -np.inf)
    for group in groups.values():
        projected = (group["xyz"] - origin) @ basis
        minimum = np.minimum(minimum, projected[:, :2].min(axis=0))
        maximum = np.maximum(maximum, projected[:, :2].max(axis=0))
    span = max(float(np.max(maximum - minimum)), 1e-9)
    size, radius = settings["image_size"], settings["point_radius"]
    scale = (size - 1 - 2 * radius) * (1 - 2 * settings["padding_fraction"]) / span
    return {"label": view["label"], "camera": view["camera"], "up": view["up"],
            "basis_columns_right_up_forward": basis.tolist(),
            "horizontal_min_max": [float(minimum[0]), float(maximum[0])],
            "vertical_min_max": [float(minimum[1]), float(maximum[1])],
            "projected_center_xy": ((minimum + maximum) / 2).tolist(),
            "pixels_per_cloud_unit": scale}


def _projected_image(points, colors, origin, projection, settings):
    """Render all RGB points with deterministic nearest-depth circular splats.

    This uses the same camera, splat, color, and tie-breaking convention as
    object_views._render_image. The frame is supplied externally. Vectorized
    pixel winner selection avoids a Python loop per dense input point.
    """
    basis = np.asarray(projection["basis_columns_right_up_forward"], dtype=float)
    projected = (points - origin) @ basis
    size, radius = settings["image_size"], settings["point_radius"]
    center = np.asarray(projection["projected_center_xy"])
    scale = projection["pixels_per_cloud_unit"]
    x = np.rint((projected[:, 0] - center[0]) * scale + (size - 1) / 2).astype(np.int64)
    y = np.rint(-(projected[:, 1] - center[1]) * scale + (size - 1) / 2).astype(np.int64)
    # Larger rank means closer to the camera. Equal-depth ties prefer later
    # records, matching the stable sorting and >= update in the old renderer.
    order = np.argsort(projected[:, 2], kind="stable")
    x, y = x[order], y[order]
    ranks = np.arange(len(order), dtype=np.int64)
    winners = np.full(size * size, -1, dtype=np.int64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            px, py = x + dx, y + dy
            valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
            np.maximum.at(winners, py[valid] * size + px[valid], ranks[valid])
    pixels = np.empty((size * size, 3), dtype=np.uint8)
    pixels[:] = settings["background_rgb"]
    filled = winners >= 0
    color8 = np.rint(colors * 255).astype(np.uint8)
    pixels[filled] = color8[order[winners[filled]]]
    result = Image.fromarray(pixels.reshape(size, size, 3))
    metrics = {"point_count": len(points), "rendered_input_count": len(points),
               "occupied_pixels": int(filled.sum()), "occupied_pixel_percent": float(filled.mean() * 100),
               "visible_point_count": int(len(np.unique(winners[filled]))),
               "point_sampling": False}
    return result, metrics


def _comparison_figure(folder, groups, settings, origin):
    """Paint human captions only into the separate comparison PNG."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(settings["views"]), 3, figsize=(11.5, 14.6), squeeze=False)
    for row, view in enumerate(settings["views"]):
        for column, role in enumerate(VARIANTS):
            axis = axes[row, column]
            with Image.open(folder / role / f"view_{row:02d}.png") as image:
                axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"{groups[role]['label']}\n{len(groups[role]['xyz']):,} points", fontsize=11)
            if column == 0:
                axis.set_ylabel(view["label"], fontsize=11)
    figure.suptitle("Density comparison · shared frame, camera, and scale within each row", fontsize=14)
    origin_text = ", ".join(f"{value:,.3f}" for value in origin)
    footer = (f"Original RGB · all supplied points rendered · {settings['image_size']} px · radius {settings['point_radius']} px. "
              "Each row uses projection bounds from all three inputs.\n"
              "Wider crop context is a separate scene view; it is not part of the regrouped candidate. No object completeness is implied.\n"
              f"Shared origin (X, Y, Z): ({origin_text}).\n" + UNIT_NOTE)
    figure.text(0.03, 0.014, footer, fontsize=8)
    figure.tight_layout(rect=(0, 0.075, 1, 0.97), h_pad=0.7, w_pad=0.6)
    path = folder / "comparison.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _record(folder, manifest, cache_hit):
    clean_paths = {role: [folder / role / f"view_{i:02d}.png" for i in range(len(manifest["settings"]["views"]))]
                   for role in VARIANTS}
    records = {}
    for role in VARIANTS:
        group_key = hashlib.sha256((manifest["cache_key"] + ":" + role).encode()).hexdigest()
        records[role] = {"cache_key": group_key, "candidate_id": manifest["inputs"][role]["id"],
                         "role": "context" if role == "context" else "candidate",
                         "density_variant": role, "paths": clean_paths[role],
                         "views": manifest["settings"]["views"], "cache_hit": cache_hit,
                         "settings": {**manifest["settings"], "shared_origin": manifest["origin"],
                                      "shared_projection_bounds": manifest["shared_projection_bounds"],
                                      "framing": "union of sparse candidate, dense candidate, wider dense crop context",
                                      "point_sampling": False},
                         "manifest_path": folder / "manifest.json"}
    return {"cache_key": manifest["cache_key"], "cache_hit": cache_hit,
            "comparison_path": folder / "comparison.png", "manifest_path": folder / "manifest.json",
            "views": manifest["settings"]["views"], "settings": manifest["settings"],
            "shared_projection_bounds": manifest["shared_projection_bounds"], "clean_paths": clean_paths,
            "candidate_renders": {role: records[role] for role in ("sparse", "dense")},
            "context_render": records["context"], "render_records": records,
            "metrics": manifest["metrics"]}


def render_quality_comparison(sparse, dense, context, output_root, origin=None, render_settings=None):
    """Save a cached 4-view × 3-variant figure and clean model input PNGs.

    Each input dictionary requires ``id``, ``xyz`` (absolute N × 3), and ``rgb``
    (normalized, uint8 or uint16), with an optional human-only ``label``.
    ``output_root`` is the parent cache directory, such as
    ``outputs/objects/quality/renders``. Explicitly pass the existing survey
    origin to retain its coordinate frame. If omitted, the union bounding-box
    midpoint becomes the shared origin. No input is modified or sampled.

    Defaults (336 pixels, radius 2, neutral background, four camera directions)
    match object_views. Each view's scale includes the wider context in every
    variant, including the clean model inputs. ``candidate_renders`` contains
    CLIP-compatible sparse and dense records; ``context_render`` is separately
    tagged ``role='context'`` so its predictions can be kept separate.

    Keys include XYZ/RGB hashes of all variants, IDs, labels, settings, origin,
    cameras, projection bounds, dependency versions, and both renderer code
    hashes. Every PNG hash is checked on reuse. Invalid prior files are retained
    and fresh outputs are written to a new attempt directory.
    """
    groups = {role: _input(group, role) for role, group in zip(VARIANTS, (sparse, dense, context))}
    settings = _render_settings(render_settings)
    if origin is None:
        minimum = np.min([group["xyz"].min(axis=0) for group in groups.values()], axis=0)
        maximum = np.max([group["xyz"].max(axis=0) for group in groups.values()], axis=0)
        origin = (minimum + maximum) / 2
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain three finite coordinates.")
    projections = [_shared_projection(groups, origin, view, settings) for view in settings["views"]]
    inputs = {role: {"id": group["id"], "label": group["label"], "point_count": len(group["xyz"]),
                     "xyz_sha256": _array_hash(group["xyz"]), "rgb_sha256": _array_hash(group["rgb"])}
              for role, group in groups.items()}
    identity = {"version": QUALITY_RENDER_VERSION, "inputs": inputs, "settings": settings,
                "origin": origin.tolist(), "shared_projection_bounds": projections,
                "renderer_code_sha256": _sha_file(__file__), "base_renderer_code_sha256": _sha_file(object_views.__file__),
                "versions": {"numpy": np.__version__, "pillow": PIL_VERSION}, "comparison_dpi": 120}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    root = Path(output_root).resolve() / key
    manifests = [root / "manifest.json"]
    if root.is_dir():
        manifests.extend(sorted(root.glob("attempt_*/manifest.json")))
    expected_files = ["comparison.png"] + [f"{role}/view_{i:02d}.png" for role in VARIANTS for i in range(len(projections))]
    for path in manifests:
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") != key or manifest.get("identity") != identity:
                continue
            files = manifest["file_sha256"]
            if sorted(files) != sorted(expected_files):
                continue
            if all((path.parent / name).is_file() and _sha_file(path.parent / name) == digest for name, digest in files.items()):
                return _record(path.parent, manifest, True)
        except (OSError, ValueError, TypeError, KeyError):
            continue
    folder = root if not root.exists() else root / ("attempt_" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True, exist_ok=False)
    metrics = {}
    for role, group in groups.items():
        (folder / role).mkdir()
        metrics[role] = []
        for i, projection in enumerate(projections):
            image, image_metrics = _projected_image(group["xyz"], group["rgb"], origin, projection, settings)
            image.save(folder / role / f"view_{i:02d}.png", format="PNG")
            metrics[role].append({"view": projection["label"], **image_metrics})
    _comparison_figure(folder, groups, settings, origin)
    manifest = {"cache_key": key, "identity": identity, "inputs": inputs, "settings": settings,
                "origin": origin.tolist(), "shared_projection_bounds": projections, "metrics": metrics,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "file_sha256": {name: _sha_file(folder / name) for name in expected_files},
                "point_sampling": False, "coordinate_units": UNIT_NOTE,
                "description": "Shared-frame original RGB point projections; no generated surfaces or colors. Human captions occur only in comparison.png.",
                "inference_separation": "Sparse and dense candidate PNGs contain only their supplied candidate points. Context PNGs contain the wider crop and must be scored/reported separately."}
    with (folder / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return _record(folder, manifest, False)


def show_quality_comparison(record):
    """Display the saved comparison PNG; do not pass this labelled figure to a model."""
    from IPython.display import Image as NotebookImage, display

    display(NotebookImage(filename=str(record["comparison_path"])))
