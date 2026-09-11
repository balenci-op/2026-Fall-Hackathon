"""Visual inspection and reproducible RGB point renders for object_review.ipynb.

Human views retain coordinate axes and surrounding sampled survey context. Model
renders are separate, orthographic images of candidate points only: no captions,
axes, class names, filenames, or review labels are painted into those images.
These are colored LiDAR point renders, not photographs. Input arrays are never
changed, and display sampling never changes clustering or processing inputs.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ground_experiment_views import UNIT_NOTE


RENDER_VERSION = "1.0"
DEFAULT_VIEWS = [
    {"label": "Front", "camera": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0]},
    {"label": "Side", "camera": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]},
    {"label": "Top", "camera": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0]},
    {"label": "Oblique", "camera": [1.0, -1.0, 0.8], "up": [0.0, 0.0, 1.0]},
]


def _sample(indices, maximum):
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("max_display_points must be positive.")
    indices = np.asarray(indices, dtype=np.int64)
    return indices if len(indices) <= maximum else indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]


def _rgb(values):
    """Normalize original RGB without auto-brightening or assigning fake colors."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Original RGB must have shape (N, 3).")
    if values.dtype == np.uint16:
        colors = values.astype(np.float64) / 65535.0
    elif values.dtype == np.uint8:
        colors = values.astype(np.float64) / 255.0
    else:
        colors = values.astype(np.float64)
    if not np.all(np.isfinite(colors)) or np.any(colors < 0) or np.any(colors > 1):
        raise ValueError("RGB must be normalized to [0, 1], uint8, or uint16; colors are never guessed.")
    return colors


def _source_arrays(source):
    xyz = np.asarray(source["xyz"], dtype=np.float64)
    rgb = _rgb(source["rgb"])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape != rgb.shape or not np.all(np.isfinite(xyz)):
        raise ValueError("Source XYZ/RGB must be matching finite arrays with shape (N, 3).")
    origin = np.asarray(source["region"]["origin"], dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("region.origin must contain three finite coordinates.")
    return xyz, rgb, origin


def _candidate_arrays(source, candidate):
    xyz, rgb, origin = _source_arrays(source)
    if candidate.get("source_identity") is not None and candidate["source_identity"] != source.get("identity"):
        raise ValueError("Candidate belongs to a different source. Regenerate it after changing the source.")
    indices = np.asarray(candidate["indices"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("Candidate indices must be a nonempty integer vector.")
    if np.any(indices < 0) or np.any(indices >= len(xyz)) or len(np.unique(indices)) != len(indices):
        raise ValueError("Candidate indices are duplicated or outside the selected source.")
    selected_xyz, selected_rgb = xyz[indices], rgb[indices]
    if "xyz" in candidate and not np.array_equal(np.asarray(candidate["xyz"]), selected_xyz):
        raise ValueError("Candidate coordinates do not match this source. Regenerate candidates after changing the source.")
    if "rgb" in candidate and not np.array_equal(_rgb(candidate["rgb"]), selected_rgb):
        raise ValueError("Candidate RGB does not match this source. Regenerate candidates after changing the source.")
    return xyz, rgb, origin, indices.astype(np.int64), selected_xyz, selected_rgb


def _context(source, candidate, context_padding):
    xyz, rgb, origin, indices, points, colors = _candidate_arrays(source, candidate)
    if not np.isfinite(context_padding) or context_padding < 0:
        raise ValueError("context_padding must be finite and nonnegative.")
    low, high = points.min(axis=0), points.max(axis=0)
    requested = [low[0] - context_padding, high[0] + context_padding,
                 low[1] - context_padding, high[1] + context_padding]
    mask = ((xyz[:, 0] >= requested[0]) & (xyz[:, 0] <= requested[1]) &
            (xyz[:, 1] >= requested[2]) & (xyz[:, 1] <= requested[3]))
    loaded = np.asarray(source["region"]["context_roi_xy"], dtype=float)
    truncated = requested[0] < loaded[0] or requested[1] > loaded[1] or requested[2] < loaded[2] or requested[3] > loaded[3]
    notes = []
    if truncated:
        notes.append("Requested context extends beyond the loaded crop; the shown context is truncated.")
    if candidate.get("touches_region_boundary", False):
        notes.append("Candidate touches the clustering region boundary and may be only part of an object.")
    notes.append("A cluster is a candidate group of points; object completeness is not established.")
    return dict(xyz=xyz, rgb=rgb, origin=origin, indices=indices, points=points, colors=colors,
                context_indices=np.flatnonzero(mask), bounds=(low, high), notes=notes,
                context_truncated=truncated, requested=requested)


def _limits(values):
    minimum, maximum = float(np.min(values)), float(np.max(values))
    pad = max(0.05, (maximum - minimum) * 0.07)
    return minimum - pad, maximum + pad


def show_candidate(source, candidate, context_padding=5, max_display_points=12000):
    """Display original RGB from three orthographic angles and a wider top-down context.

    Uses local coordinates relative to the same origin as the ground notebook.
    Context comes from the already loaded original processing sample, including
    ground when present; it is not a denser extraction from raw LAS. Any crop
    truncation and candidate boundary contact are reported below the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = _context(source, candidate, context_padding)
    selected = _sample(data["indices"], max_display_points)
    context = _sample(data["context_indices"], max_display_points)
    local = data["xyz"] - data["origin"]
    points = data["points"] - data["origin"]
    low, high = [bounds - data["origin"] for bounds in data["bounds"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, horizontal, vertical, name in zip(axes.flat[:3], [0, 0, 1], [1, 2, 2], ["Top", "Front", "Side"]):
        ax.scatter(local[selected, horizontal], local[selected, vertical], c=data["rgb"][selected],
                   s=7, linewidths=0, rasterized=True)
        ax.set(xlim=_limits(points[:, horizontal]), ylim=_limits(points[:, vertical]), aspect="equal",
               xlabel=f"Local {'XYZ'[horizontal]} [cloud units]", ylabel=f"Local {'XYZ'[vertical]} [cloud units]")
        ax.set_title(name + " · candidate original RGB")
    ax = axes.flat[3]
    ax.scatter(local[context, 0], local[context, 1], c=data["rgb"][context], s=3, alpha=0.30,
               linewidths=0, rasterized=True)
    ax.scatter(local[selected, 0], local[selected, 1], c=data["rgb"][selected], s=8, linewidths=0, rasterized=True)
    ax.add_patch(Rectangle((low[0], low[1]), high[0] - low[0], high[1] - low[1],
                           fill=False, edgecolor="#db2777", linewidth=1.5, linestyle="--"))
    ax.set(xlim=_limits(local[data["context_indices"], 0]), ylim=_limits(local[data["context_indices"], 1]),
           aspect="equal", xlabel="Local X [cloud units]", ylabel="Local Y [cloud units]")
    ax.set_title("Original sample context · outlined candidate")
    for ax in axes.flat:
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(alpha=0.12)
    fig.suptitle(f"{candidate['id']} · {len(data['indices']):,} candidate points · original RGB", fontsize=13)
    footer = (f"Display only: {len(selected):,} candidate points; {len(context):,} / {len(data['context_indices']):,} context points. "
              f"XY context padding: {context_padding:g} cloud units.\n"
              + " ".join(data["notes"]) + "\n" + UNIT_NOTE)
    fig.text(0.025, 0.02, footer, fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    plt.show()


def candidate_3d(source, candidate, context_padding=5, max_display_points=12000):
    """Return a Plotly RGB candidate view; context can be toggled without widgets."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Interactive 3D unavailable: plotly is missing from the selected kernel. Static views remain available.")
        return None
    data = _context(source, candidate, context_padding)
    selected = _sample(data["indices"], max_display_points)
    context_indices = np.setdiff1d(data["context_indices"], data["indices"], assume_unique=True)
    context = _sample(context_indices, max_display_points)
    local = data["xyz"] - data["origin"]
    fig = go.Figure()
    for indices, name, size, opacity in [(selected, "Candidate · original RGB", 3, 1.0),
                                        (context, "Surrounding original sample", 2, 0.30)]:
        rgb8 = np.rint(data["rgb"][indices] * 255).astype(np.uint8)
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb8]
        fig.add_trace(go.Scatter3d(x=local[indices, 0], y=local[indices, 1], z=local[indices, 2],
                                  mode="markers", name=name, marker=dict(size=size, color=colors, opacity=opacity),
                                  visible=True if name.startswith("Candidate") else "legendonly",
                                  hovertemplate="Local X %{x:.3f}<br>Local Y %{y:.3f}<br>Local Z %{z:.3f}<extra>" + name + "</extra>"))
    # The same bounds remain fixed as context visibility changes.
    limits = [_limits(local[data["context_indices"], dim]) for dim in range(3)]
    fig.update_layout(
        title=dict(text=f"{candidate['id']} · colored point cloud<br><sup>{len(selected):,} display / {len(data['indices']):,} candidate points; click context legend to toggle</sup>"),
        height=650, margin=dict(l=0, r=0, b=105, t=90), scene=dict(aspectmode="data",
            xaxis=dict(title="Local X [cloud units]", range=list(limits[0])),
            yaxis=dict(title="Local Y [cloud units]", range=list(limits[1])),
            zaxis=dict(title="Local Z [cloud units]", range=list(limits[2]))),
        uirevision=str(source.get("identity", "source")) + ":" + str(candidate["id"]),
        annotations=[dict(x=0, y=-0.16, xref="paper", yref="paper", showarrow=False, xanchor="left", align="left",
                          text="<br>".join(data["notes"] + [UNIT_NOTE]), font=dict(size=10))],
    )
    return fig


def show_clusters(source, batch, max_display_points=15000):
    """Show every retained candidate and noise, independent of review decisions.

    Colors identify cluster membership only. Each cluster receives at least one
    point in the display sample when the cap permits. Noise is retained as gray
    context; no semantic label or review status removes a candidate from this view.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xyz, _, origin = _source_arrays(source)
    if batch.get("source") is not None and batch["source"].get("identity") != source.get("identity"):
        raise ValueError("Cluster batch belongs to a different source. Regenerate it after changing the source.")
    eligible = np.asarray(batch["eligible_indices"], dtype=np.int64)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    if labels.shape != eligible.shape or not len(eligible):
        raise ValueError("Batch labels must align with nonempty eligible_indices.")
    label_values, first = np.unique(labels, return_index=True)
    maximum = int(max_display_points)
    if maximum < len(first):
        raise ValueError(f"Use max_display_points >= {len(first)} to represent every cluster and noise group.")
    remaining = np.setdiff1d(np.arange(len(eligible)), first, assume_unique=True)
    chosen = np.sort(np.concatenate([first, _sample(remaining, maximum - len(first))])) if maximum > len(first) else np.sort(first)
    indices, shown_labels = eligible[chosen], labels[chosen]
    local = xyz - origin
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = np.array([plt.get_cmap("tab20")(int(label) % 20) if label >= 0 else (0.45, 0.45, 0.45, 0.45) for label in shown_labels])
    for ax, vertical, name in zip(axes, [1, 2], ["Top-down", "Elevation across local X"]):
        ax.scatter(local[indices, 0], local[indices, vertical], c=colors, s=3, linewidths=0, rasterized=True)
        ax.set(xlim=_limits(local[eligible, 0]), ylim=_limits(local[eligible, vertical]),
               xlabel="Local X [cloud units]", ylabel=f"Local {'XYZ'[vertical]} [cloud units]")
        if vertical == 1:
            ax.set_aspect("equal")
        ax.set_title(name)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(alpha=0.12)
    counts = int(np.sum(label_values >= 0))
    fig.suptitle(f"{counts:,} retained clusters · {len(batch['candidates']):,} selected for review · {int((labels == -1).sum()):,} noise / small-group points")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color="#2563eb", label="Cluster colors identify candidates, not object classes"),
                        Line2D([], [], marker="o", linestyle="", color="#737373", label="Noise / excluded small groups")],
               loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, frameon=False, fontsize=9)
    fig.text(0.025, 0.02, f"Display: {len(indices):,} / {len(eligible):,} eligible points; clustering uses the full eligible input. Clusters outside the review batch remain visible.\n" + UNIT_NOTE, fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    plt.show()


def _render_settings(settings):
    defaults = {"image_size": 336, "point_radius": 2, "background_rgb": [245, 245, 245],
                "padding_fraction": 0.08, "views": DEFAULT_VIEWS}
    supplied = {} if settings is None else dict(settings)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown render settings: {sorted(unknown)}")
    result = json.loads(json.dumps({**defaults, **supplied}))
    if result["image_size"] not in (224, 336):
        raise ValueError("image_size must be 224 or 336 pixels.")
    if not isinstance(result["point_radius"], int) or not 1 <= result["point_radius"] <= 5:
        raise ValueError("point_radius must be an integer from 1 to 5 pixels.")
    bg = result["background_rgb"]
    if len(bg) != 3 or any(not isinstance(value, int) or value < 0 or value > 255 for value in bg):
        raise ValueError("background_rgb must contain three 8-bit integer channel values.")
    if not 0 <= result["padding_fraction"] < 0.4:
        raise ValueError("padding_fraction must be between 0 and 0.4.")
    if len(result["views"]) != 4:
        raise ValueError("Provide four deterministic views (front, side, top, oblique by default).")
    for view in result["views"]:
        if set(view) != {"label", "camera", "up"}:
            raise ValueError("Each view requires label, camera, and up only.")
        camera, up = np.asarray(view["camera"], dtype=float), np.asarray(view["up"], dtype=float)
        if camera.shape != (3,) or up.shape != (3,) or not np.all(np.isfinite([camera, up])):
            raise ValueError("Camera and up vectors must each have three finite coordinates.")
        if np.linalg.norm(camera) == 0 or np.linalg.norm(np.cross(up, camera)) < 1e-10:
            raise ValueError("Camera direction must be nonzero and not parallel to the up vector.")
    return result


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(digest, array):
    array = np.ascontiguousarray(array, dtype="<f8")
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


def _render_image(points, colors, view, settings):
    """Orthographic RGB splats with a true per-pixel nearest-depth buffer."""
    from PIL import Image

    size, radius = settings["image_size"], settings["point_radius"]
    centered = points - (points.min(axis=0) + points.max(axis=0)) / 2
    forward = np.asarray(view["camera"], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(view["up"], dtype=float), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    horizontal, vertical, depth = centered @ right, centered @ up, centered @ forward
    # Uniform scale within a view preserves geometry; centering changes no input.
    span = max(float(np.ptp(horizontal)), float(np.ptp(vertical)), 1e-9)
    scale = (size - 1 - 2 * radius) * (1 - 2 * settings["padding_fraction"]) / span
    pixels_x = np.rint((horizontal - (horizontal.min() + horizontal.max()) / 2) * scale + (size - 1) / 2).astype(np.int64)
    pixels_y = np.rint(-(vertical - (vertical.min() + vertical.max()) / 2) * scale + (size - 1) / 2).astype(np.int64)
    # Each splat writes its nearest point to each pixel. Stable tie-breaking
    # follows original record order; there is no random jitter or hidden shading.
    color8 = np.rint(colors * 255).astype(np.uint8)
    order = np.argsort(depth, kind="stable")
    output = np.empty((size, size, 3), dtype=np.uint8)
    output[:] = settings["background_rgb"]
    zbuffer = np.full((size, size), -np.inf)
    for index in order:
        px, py = int(pixels_x[index]), int(pixels_y[index])
        xmin, xmax = max(0, px - radius), min(size, px + radius + 1)
        ymin, ymax = max(0, py - radius), min(size, py + radius + 1)
        if xmin >= xmax or ymin >= ymax:
            continue
        yy, xx = np.ogrid[ymin:ymax, xmin:xmax]
        disk = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
        target = zbuffer[ymin:ymax, xmin:xmax]
        nearer = disk & (depth[index] >= target)
        target[nearer] = depth[index]
        output[ymin:ymax, xmin:xmax][nearer] = color8[index]
    return Image.fromarray(output)


def render_candidate(source, candidate, render_settings=None):
    """Cache four annotation-free candidate RGB images and a provenance manifest.

    Default images are 336 × 336 pixels, generated from every candidate point.
    Inputs to models are PNGs in ``paths`` only, never the human view, source
    context, manifest, or captions. Original RGB is quantized to 8 bits without
    brightness adjustment, color assignment, texture generation, or class cues.

    The cache key includes candidate XYZ/RGB/ID, all cameras and rendering
    settings, renderer version, and this implementation's SHA256. PNG hashes
    are rechecked on every hit. Corrupted cache files are preserved and a fresh
    attempt is written below the same hash directory; nothing is overwritten.
    """
    _, _, _, _, points, colors = _candidate_arrays(source, candidate)
    settings = _render_settings(render_settings)
    implementation_hash = _sha_file(Path(__file__))
    identity = {"candidate_id": str(candidate["id"]), "settings": settings,
                "renderer_version": RENDER_VERSION, "renderer_code_sha256": implementation_hash}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    _array_digest(digest, points)
    _array_digest(digest, colors)
    key = digest.hexdigest()
    root = Path(source["project_dir"]) / "outputs" / "objects" / "renders" / key
    possible = [root / "manifest.json"]
    if root.is_dir():
        possible.extend(sorted(root.glob("attempt_*/manifest.json")))
    for manifest_path in possible:
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_key") != key or manifest.get("identity") != identity:
                continue
            filenames = [item["file"] for item in manifest["images"]]
            if len(filenames) != len(settings["views"]) or filenames != [f"view_{i:02d}.png" for i in range(len(settings["views"]))]:
                continue
            paths = [manifest_path.parent / filename for filename in filenames]
            if all(path.is_file() and _sha_file(path) == info["sha256"] for path, info in zip(paths, manifest["images"])):
                return {"cache_key": key, "candidate_id": str(candidate["id"]), "paths": paths, "views": settings["views"], "cache_hit": True,
                        "settings": settings, "manifest_path": manifest_path}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    folder = root
    if root.exists():
        folder = root / ("attempt_" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True, exist_ok=False)
    paths, images = [], []
    for i, view in enumerate(settings["views"]):
        path = folder / f"view_{i:02d}.png"
        _render_image(points, colors, view, settings).save(path, format="PNG")
        paths.append(path)
        images.append({"file": path.name, "sha256": _sha_file(path), "view": view})
    manifest = {"cache_key": key, "candidate_id": str(candidate["id"]), "identity": identity, "images": images,
                "candidate_point_count": len(points), "source_identity": source.get("identity"),
                "created_utc": datetime.now(timezone.utc).isoformat(), "display_sampling": False,
                "description": "Original RGB LiDAR points only, orthographic splats, neutral background, no text or axes.",
                "coordinate_units": UNIT_NOTE}
    manifest_path = folder / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return {"cache_key": key, "candidate_id": str(candidate["id"]), "paths": paths, "views": settings["views"], "cache_hit": False,
            "settings": settings, "manifest_path": manifest_path}


def show_renders(record):
    """Show a human-readable collage; captions are outside the model-input PNGs."""
    import matplotlib.pyplot as plt
    from PIL import Image

    paths, views = record["paths"], record["views"]
    fig, axes = plt.subplots(1, len(paths), figsize=(3.5 * len(paths), 4), squeeze=False)
    for ax, path, view in zip(axes.flat, paths, views):
        with Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(view["label"])
        ax.axis("off")
    fig.suptitle("Candidate-only RGB point renders · images contain no captions or axes")
    fig.text(0.02, 0.015, f"{record['settings']['image_size']} px · all candidate points rendered · "
             f"cache {'hit' if record['cache_hit'] else 'created'} · colored point renders, not photographs", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    plt.show()
