# Candidate quality comparison

Manual labeling is paused. Open `object_review.ipynb` with **LiDAR Object Review
(.venv-ground)** and use the **Candidate quality comparison** link at the top.
That section loads saved results only; it does not scan raw LAS or rerun inference.

This bounded pass inspected 10 existing groups and selected three different
structures. One chunked read of `Run 1 Laser Left.las` collected all three dense
regions. Existing CSF settings and one spacing-based grouping configuration were
applied per region. All original seed records were recovered from raw LAS.

| Original candidate | Sparse points | Dense proposal points | Original seeds retained | Dense crop edge |
|---|---:|---:|---:|---|
| 00012 — inspect first | 308 | 32,146 | 298 / 308 | No |
| 00081 | 394 | 20,163 | 242 / 394 | No |
| 00019 | 139 | 31,325 | 86 / 139 | Yes |

Start with **00012** because its structure is clearer, almost all original seeds
remain, and it stays inside the crop. Its four-view comparison is saved in the
notebook and in [comparison_00012.png](outputs/objects/quality/20260911T074133_107528Z/comparison_00012.png).
[00081](outputs/objects/quality/20260911T074133_107528Z/comparison_00081.png) loses part
of its original vertical extent; [00019](outputs/objects/quality/20260911T074133_107528Z/comparison_00019.png)
shows a clear crop-edge failure. No complete object or class is established.

**Diagnosis:** source sampling was the dominant initial sparsity cause. The same
XYZ boxes contain 53–58 times more raw points. The existing renderer already used
every candidate point; comparisons keep the same point radius and resolution.
Grouping and complete-object boundaries are now the main limitation.

Real local CLIP inference was run on sparse, dense, and context views separately.
All three dense proposals lead with a bush/shrub description; their viewpoint
disagreement is 25%, 50%, and 0% for 00012, 00081, and 00019 respectively. These are
unverified cosine-similarity rankings, not probabilities. Zero disagreement does
not make the visibly clipped 00019 a good crop.

**Single recommended next step:** inspect 00012 against its wider context to
decide whether its boundaries are complete enough before resuming manual labeling.

[Settings, counts, provenance, and prediction references](outputs/objects/quality/20260911T074133_107528Z/report.json)
are saved with the crops, membership, renders, and previous-notebook backup. Raw
files, existing ground experiments, sparse results, and review records were
preserved. No human labels, model training, broad search, or external data upload
occurred. This bounded pass is finished.
