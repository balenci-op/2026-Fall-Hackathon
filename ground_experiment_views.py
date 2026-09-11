"""Display helpers for ``ground_experiments.ipynb``.

The notebook keeps processing arrays intact.  These functions select deterministic
display subsets only, translate coordinates by one shared origin, and never write
LAS files or modify input arrays.  Distances are deliberately labelled *cloud
units*: the workspace's LAS headers and export sidecars disagree about units.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np


COLORS = {"original": "#64748b", "ground": "#15803d", "non_ground": "#c2410c", "unassigned": "#9ca3af"}
UNIT_NOTE = "Cloud units unresolved: LAS headers declare US survey feet; export sidecars say metres."


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _format_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        rendered = json.dumps(value, default=_json_default, ensure_ascii=False, indent=2)
        summary = rendered.replace("\n", " ")
        if len(summary) > 100:
            summary = summary[:97] + "…"
        return (f"<details><summary>{html.escape(summary)}</summary>"
                f"<pre style='white-space:pre-wrap'>{html.escape(rendered)}</pre></details>")
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}".rstrip("0").rstrip(".") if np.isfinite(value) else str(value)
    return html.escape(str(value)).replace("\n", "<br>")


def show_table(rows, columns=None):
    """Display mappings as an HTML table; expand structured cells for full settings.

    Call directly in a notebook cell.  This function displays its result and
    returns ``None``. No pandas dependency is required.
    """
    from IPython.display import HTML, display

    rows = list(rows)
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    columns = list(columns)
    if not rows:
        display(HTML("<p>No experiments recorded yet.</p>"))
        return
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{_format_cell(row.get(c))}</td>" for c in columns) + "</tr>" for row in rows)
    display(HTML(
        "<div style='overflow-x:auto'><style>.ground-table{border-collapse:collapse;font-size:13px}"
        ".ground-table th,.ground-table td{border:1px solid #cbd5e1;padding:7px;text-align:left;vertical-align:top;max-width:320px;overflow-wrap:anywhere}"
        ".ground-table th{background:#e2e8f0;color:#0f172a}.ground-table pre{max-width:650px}"
        "</style><table class='ground-table'><thead><tr>" + header + "</tr></thead><tbody>" + body + "</tbody></table></div>"
    ))


def _sample_indices(indices, maximum):
    """Evenly spaced record indices, stable across methods and notebook reruns."""
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("max_display_points must be at least 1.")
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]


def _coordinates(item):
    xyz = np.asarray(item["xyz"], dtype=float)
    origin = np.asarray(item["origin"], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or origin.shape != (3,):
        raise ValueError("Expected xyz with shape (N, 3) and origin with shape (3,).")
    return xyz, origin, xyz - origin


def _origin_note(origin):
    return "Shared origin (X, Y, Z) = (" + ", ".join(f"{n:,.3f}" for n in origin) + ")."


def suggested_center(overview):
    """Return the overview sample's median XY in local coordinates.

    This is an automatic starting region, never an identified missing-road site.
    """
    _, _, local = _coordinates(overview)
    finite = local[np.all(np.isfinite(local), axis=1)]
    if not len(finite):
        raise ValueError("The overview contains no finite points.")
    return np.median(finite[:, :2], axis=0).tolist()


def plot_overview(overview, roi_xy=None):
    """Display a top-down overview colored by elevation and an optional absolute ROI."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xyz, origin, local = _coordinates(overview)
    local = local[np.all(np.isfinite(local), axis=1)]
    if not len(local):
        raise ValueError("The overview contains no finite points.")
    # The survey is a long, narrow corridor. A portrait layout keeps the map
    # legible without the large empty area produced by a landscape equal-aspect axis.
    fig = plt.figure(figsize=(6.5, 10))
    ax = fig.add_axes((0.16, 0.14, 0.50, 0.76))
    color_axis = fig.add_axes((0.76, 0.21, 0.035, 0.60))
    points = ax.scatter(local[:, 0], local[:, 1], c=local[:, 2], cmap="viridis", s=1.5,
                        linewidths=0, rasterized=True)
    fig.colorbar(points, cax=color_axis, label="Local elevation Z [cloud units]")
    if roi_xy is not None:
        xmin, xmax, ymin, ymax = np.asarray(roi_xy, dtype=float)
        ax.add_patch(Rectangle((xmin - origin[0], ymin - origin[1]), xmax - xmin,
                               ymax - ymin, facecolor="none", edgecolor="#ef4444", linewidth=2,
                               label="Selected evaluation region"))
        ax.legend(loc="best")
    source = Path(str(overview.get("source_path", "Selected source"))).name
    count = int(overview.get("source_count", len(xyz)))
    fig.suptitle(f"{source}\n{len(local):,} overview display points / {count:,} source points", fontsize=11, y=0.97)
    ax.set(xlabel="Local X [cloud units]", ylabel="Local Y [cloud units]", aspect="equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(alpha=0.15)
    fig.text(0.04, 0.035, _origin_note(origin) + "\nCloud units unresolved: LAS = US survey feet;\nexport sidecars = meters.", fontsize=8)
    plt.show()


def _limits(values):
    low, high = float(np.min(values)), float(np.max(values))
    padding = max((high - low) * 0.025, 0.05)
    return low - padding, high + padding


def _prepare(region, maximum, section_axis, section_position, section_width):
    xyz, origin, local = _coordinates(region)
    core = np.asarray(region["evaluation_mask"], dtype=bool)
    if core.shape != (len(xyz),):
        raise ValueError("evaluation_mask must have one value per processing point.")
    if section_axis not in ("x", "y"):
        raise ValueError("section_axis must be 'x' or 'y'.")
    if not np.isfinite(section_width) or section_width <= 0:
        raise ValueError("section_width must be positive and finite.")
    valid = np.all(np.isfinite(local), axis=1)
    if np.any(core & ~valid):
        raise ValueError("Evaluation region has non-finite coordinates; clean input before comparing methods.")
    indices = np.flatnonzero(core)
    if not len(indices):
        raise ValueError("No processing points are inside the selected evaluation region. Choose another ROI.")
    core_xyz = local[indices]
    band_dimension = 1 if section_axis == "y" else 0
    horizontal_dimension = 1 - band_dimension
    if section_position is None:
        if "roi_xy" in region:
            roi = np.asarray(region["roi_xy"], dtype=float)
            pair = roi[2:4] if band_dimension == 1 else roi[:2]
            section_position = float(np.mean(pair) - origin[band_dimension])
        else:
            section_position = float(np.mean([core_xyz[:, band_dimension].min(), core_xyz[:, band_dimension].max()]))
    if not np.isfinite(section_position):
        raise ValueError("section_position must be finite.")
    section = core & (np.abs(local[:, band_dimension] - section_position) <= section_width / 2)
    return dict(xyz=xyz, origin=origin, local=local, core=core,
                core_count=len(indices), top_indices=_sample_indices(indices, maximum),
                section_indices=_sample_indices(np.flatnonzero(section), maximum),
                section_count=int(section.sum()), band_dimension=band_dimension,
                horizontal_dimension=horizontal_dimension, section_position=section_position,
                section_width=float(section_width), limits=[_limits(core_xyz[:, axis]) for axis in range(3)])


def _result_masks(result, length):
    ground = np.asarray(result["ground_mask"], dtype=bool)
    coverage = np.asarray(result.get("coverage_mask", np.ones(length, dtype=bool)), dtype=bool)
    if ground.shape != (length,) or coverage.shape != (length,):
        raise ValueError(f"{result.get('name', 'Result')}: masks must have one value per processing point.")
    return ground & coverage, ~ground & coverage, ~coverage


def _validate_result_identity(region, result):
    """Reject out-of-order notebook state when provenance fingerprints exist."""
    actual = result.get("input_fingerprint")
    if actual is not None and actual != region.get("fingerprint"):
        raise ValueError(
            f"{result.get('name', 'Result')}: input fingerprint differs from the current region. "
            "Reload saved results or rerun filtering after changing the region."
        )


def _scatter(ax, local, indices, horizontal, vertical, mask=None, color=None):
    selected = indices if mask is None else indices[mask[indices]]
    ax.scatter(local[selected, horizontal], local[selected, vertical], c=color,
               s=3 if len(indices) < 5000 else 1.4, alpha=0.8, linewidths=0, rasterized=True)


def plot_region(region, max_display_points=20000, section_axis="y", section_position=None, section_width=4.0):
    """Display the original region with a top-down view and a configurable elevation slice.

    ``section_axis='y'`` selects a local Y band and plots X versus Z; ``'x'``
    selects a local X band and plots Y versus Z. ``section_position`` is local,
    measured from ``region['origin']``. ``None`` chooses the ROI midpoint.
    Display sampling never changes ``region['xyz']`` or processing records.
    """
    plot_comparison(region, [], max_display_points, section_axis, section_position,
                    section_width, title="Before filtering: selected evaluation region")


def plot_comparison(region, results, max_display_points=20000, section_axis="y",
                    section_position=None, section_width=4.0, title=None):
    """Display original and ground partitions on exactly shared coordinates and limits.

    ``results`` is a list of dictionaries with ``name`` and ``ground_mask`` over
    ALL processing points (including the surrounding margin). An optional
    ``coverage_mask`` marks points represented by saved outputs; missing matches
    remain gray and are counted. Statistics and axis bounds use the full
    evaluation/core input. One common display subset is used by every method;
    the elevation slice is sampled independently after selecting its full band.

    Displays a Matplotlib figure and returns ``None``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results = list(results)
    for result in results:
        _validate_result_identity(region, result)
    prepared = _prepare(region, max_display_points, section_axis, section_position, section_width)
    local, core = prepared["local"], prepared["core"]
    width = 5 if not results else 4.2 * (len(results) + 1)
    fig, axes = plt.subplots(2, len(results) + 1, figsize=(width, 9), squeeze=False)
    horizontal = prepared["horizontal_dimension"]
    horizontal_label = "XY"[horizontal]
    position, band_width = prepared["section_position"], prepared["section_width"]
    for column, result in enumerate([None, *results]):
        top, section_ax = axes[:, column]
        if result is None:
            name = "Original input"
            counts = f"{prepared['core_count']:,} evaluation points"
            _scatter(top, local, prepared["top_indices"], 0, 1, color=COLORS["original"])
            _scatter(section_ax, local, prepared["section_indices"], horizontal, 2, color=COLORS["original"])
        else:
            name = result["name"]
            ground, nonground, missing = _result_masks(result, len(local))
            ground_count, non_count, missing_count = [int((core & mask).sum()) for mask in (ground, nonground, missing)]
            counts = f"Ground {ground_count:,} · non-ground {non_count:,}"
            if missing_count:
                counts += f"\nUnassigned {missing_count:,}"
            for mask, color in [(missing, COLORS["unassigned"]), (ground, COLORS["ground"]), (nonground, COLORS["non_ground"])]:
                _scatter(top, local, prepared["top_indices"], 0, 1, mask, color)
                _scatter(section_ax, local, prepared["section_indices"], horizontal, 2, mask, color)
        top.set_title(name + "\n" + counts, fontsize=10)
        if prepared["band_dimension"] == 1:
            top.axhspan(position - band_width / 2, position + band_width / 2, color="#2563eb", alpha=0.10)
            top.axhline(position - band_width / 2, color="#2563eb", alpha=0.7, linewidth=0.7)
            top.axhline(position + band_width / 2, color="#2563eb", alpha=0.7, linewidth=0.7)
        else:
            top.axvspan(position - band_width / 2, position + band_width / 2, color="#2563eb", alpha=0.10)
            top.axvline(position - band_width / 2, color="#2563eb", alpha=0.7, linewidth=0.7)
            top.axvline(position + band_width / 2, color="#2563eb", alpha=0.7, linewidth=0.7)
        top.set(xlim=prepared["limits"][0], ylim=prepared["limits"][1], aspect="equal",
                xlabel="Local X [cloud units]", ylabel="Local Y [cloud units]")
        section_ax.set(xlim=prepared["limits"][horizontal], ylim=prepared["limits"][2],
                       xlabel=f"Local {horizontal_label} [cloud units]", ylabel="Local Z [cloud units]")
        section_ax.set_title(f"Elevation section: {section_axis.upper()} = {position:.2f} ± {band_width / 2:.2f}", fontsize=9)
        if not prepared["section_count"]:
            section_ax.text(0.5, 0.5, "No points in this section.\nChange section position or width.",
                            transform=section_ax.transAxes, ha="center", va="center")
        for ax in (top, section_ax):
            ax.ticklabel_format(style="plain", useOffset=False)
            ax.grid(alpha=0.15)
    handles = [Line2D([], [], marker="o", linestyle="", color=color, label=label, markersize=5)
               for label, color in [("Original", COLORS["original"]), ("Ground", COLORS["ground"]),
                                    ("Non-ground", COLORS["non_ground"]), ("Unassigned saved points", COLORS["unassigned"])]]
    fig.legend(handles=handles if results else handles[:1], loc="upper center", bbox_to_anchor=(0.5, 0.951),
               ncol=4 if results else 1, frameon=False, fontsize=9)
    fig.suptitle(title or "Same region, input points, origin, limits, and display subsets", fontsize=12, y=0.99)
    footer = (f"Loaded context (new-run input): {len(local):,}; evaluation region: {prepared['core_count']:,}. "
              f"Top-down display: {len(prepared['top_indices']):,}.\n"
              f"Section input: {prepared['section_count']:,}; section display: {len(prepared['section_indices']):,}. "
              "Blue band marks the section. Elevation scale is shared across panels.\n"
              + _origin_note(prepared["origin"]) + "\n" + UNIT_NOTE)
    fig.text(0.02, 0.015, footer, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.91), h_pad=2, w_pad=2)
    plt.show()


def interactive_3d(region, results, max_display_points=12000):
    """Return an optional Plotly figure with dropdown views and shared scene bounds.

    Use ``fig = interactive_3d(...); display(fig)`` in Jupyter. Interactive output
    needs Plotly plus a frontend that supports its MIME type. The notebook's
    Matplotlib panels remain the saved static reference. The returned figure can
    also be written to a standalone HTML file by explicitly calling
    ``fig.write_html(path, include_plotlyjs=True)``. This helper writes no files.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Interactive 3D unavailable: the selected kernel is missing plotly. Static Matplotlib figures remain available.")
        return None

    results = list(results)
    for result in results:
        _validate_result_identity(region, result)
    xyz, origin, local = _coordinates(region)
    core = np.asarray(region["evaluation_mask"], dtype=bool)
    if core.shape != (len(xyz),):
        raise ValueError("evaluation_mask must have one value per processing point.")
    if np.any(core & ~np.all(np.isfinite(local), axis=1)):
        raise ValueError("Evaluation region has non-finite coordinates.")
    indices = _sample_indices(np.flatnonzero(core), max_display_points)
    if not len(indices):
        raise ValueError("The evaluation region contains no points.")
    limits = [_limits(local[core, axis]) for axis in range(3)]
    fig = go.Figure()
    groups = []
    for result in [None, *results]:
        start = len(fig.data)
        if result is None:
            name = "Original input"
            partitions = [(np.ones(len(local), dtype=bool), "Original", COLORS["original"])]
        else:
            name = result["name"]
            ground, nonground, missing = _result_masks(result, len(local))
            partitions = [(ground, "Ground", COLORS["ground"]), (nonground, "Non-ground", COLORS["non_ground"]),
                          (missing, "Unassigned", COLORS["unassigned"])]
        for mask, label, color in partitions:
            selected = indices[mask[indices]]
            if not len(selected):
                continue
            points = local[selected]
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                      name=label, marker=dict(size=2, color=color, opacity=0.8),
                                      visible=result is None,
                                      hovertemplate="Local X %{x:.3f}<br>Local Y %{y:.3f}<br>Local Z %{z:.3f}<extra>" + label + "</extra>"))
        groups.append((name, list(range(start, len(fig.data)))))
    buttons = []
    subtitle = f"{len(indices):,} display points / {int(core.sum()):,} evaluation points; display sampling only"
    for name, group in groups:
        visible = [i in group for i in range(len(fig.data))]
        buttons.append(dict(label=name, method="update", args=[dict(visible=visible), dict(title=dict(text=f"{name}<br><sup>{subtitle}</sup>"))]))
    fig.update_layout(
        title=dict(text=f"Original input<br><sup>{subtitle}</sup>"), height=620, margin=dict(l=5, r=5, b=60, t=100),
        scene=dict(xaxis=dict(title="Local X [cloud units]", range=list(limits[0])),
                   yaxis=dict(title="Local Y [cloud units]", range=list(limits[1])),
                   zaxis=dict(title="Local Z [cloud units]", range=list(limits[2])), aspectmode="data"),
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.10, xanchor="left", yanchor="top")],
        uirevision="shared-ground-region", legend=dict(x=0.85, y=1),
        annotations=[dict(text=html.escape(_origin_note(origin)) + "<br>" + UNIT_NOTE, x=0, y=-0.1,
                          xref="paper", yref="paper", showarrow=False, xanchor="left", font=dict(size=10))],
    )
    return fig
