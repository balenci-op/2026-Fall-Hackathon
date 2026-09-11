"""Lightweight presentation of the completed, bounded quality pass.

Only saved figures and JSON are read. This module never scans LAS, groups points,
runs inference, or writes review labels. Processing remains in object_quality.py.
"""
from pathlib import Path
from IPython.display import Image, Markdown, display
from ground_experiment_views import show_table
from object_quality import read_json


def show_summary(report):
    rows = []
    for item in report["comparisons"]:
        count = item["counts_and_source"]
        dense = next(row for row in item["prediction_summary"] if row["role"] == "dense")
        rows.append({"candidate": item["parent_candidate_id"].split("_")[-1],
                     "sparse_points": count["sparse_points"], "dense_points": count["dense_candidate_points"],
                     "same_box_density_gain": round(count["same_box_density_multiplier"], 1),
                     "original_seeds_retained": f"{count['original_seed_exact_selected_count']}/{count['sparse_points']}",
                     "dense_crop_edge": count["dense_touches_crop_boundary"],
                     "CLIP_suggestion_unverified": dense["top1"],
                     "view_disagreement": dense["view_disagreement"]})
    show_table(rows)


def show_candidate(report, index):
    item = report["comparisons"][index]
    counts = item["counts_and_source"]
    display(Image(filename=item["figure_path"], width=1050))
    print(item["parent_candidate_id"], "|", item["selection_reason"])
    print("Raw source:", Path(counts["source_raw"]).name,
          "| Original sample:", Path(counts["source_sample"]).name)
    show_table([{"measurement": name, "value": counts[name]} for name in (
        "sparse_points", "dense_candidate_points", "wider_raw_context_points",
        "same_xyz_box_sample_points", "same_xyz_box_raw_points", "same_box_density_multiplier",
        "original_seed_records_recovered", "original_seed_exact_selected_count",
        "original_seed_local_csf_ground_count", "dense_touches_crop_boundary",
        "sparse_nn_median", "dense_nn_median", "grouping_eps", "grouping_runtime_s")])
    for role in ("sparse", "dense", "context"):
        prediction = read_json(item["predictions"][role])
        title = "Wider context only — a region prediction" if role == "context" else f"{role.title()} candidate only"
        display(Markdown(f"**{title}.** Cosine similarities, not calibrated probabilities. "
                         f"View disagreement: **{prediction['top1_disagreement_fraction']:.0%}**."))
        show_table(prediction["top5"])
        show_table([{"view": view["view_label"], "leading_description": view["top1"],
                     "leading_cosine": view["top3"][0]["cosine_similarity"]}
                    for view in prediction["per_view"]])
