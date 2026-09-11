# New candidate comparison

Open **object_review.ipynb → New candidate comparison** with the existing
**LiDAR Object Review (.venv-ground)** kernel. The new cells load saved outputs;
the whole notebook does not need to rerun. Manual labeling remains paused.

Inspect **b2a866edbb1fac31_00027 first**: its vertical structure and upper
crosspieces are much clearer, and all 58 original seed points remain.

- [Vertical comparison figure](outputs/objects/new_candidates/20260911T140448_622999Z/comparison_b2a866edbb1fac31_00027.png)
- [Vertical interactive HTML](outputs/objects/new_candidates/20260911T140448_622999Z/interactive_b2a866edbb1fac31_00027.html)
- [Planar comparison figure](outputs/objects/new_candidates/20260911T140448_622999Z/comparison_1d45ed4734454e12_00041.png)
- [Planar interactive HTML](outputs/objects/new_candidates/20260911T140448_622999Z/interactive_1d45ed4734454e12_00041.html)

Open HTML in a browser by double-clicking it in Explorer, or use the notebook's
`OPEN_BROWSER=True` cell. Plotly is embedded locally; no notebook renderer, CDN,
or server is needed. Toggle candidate/context in the legend. Interactive display
is capped at 20,000 points per trace; static renders and processing use all points.

| Geometry / ID suffix | Sparse | Dense group | Full region | Seed retention |
|---|---:|---:|---:|---:|
| Vertical / 00027 | 58 | 11,891 | 20,719 | 58 / 58 |
| Planar / 00041 | 234 | 21,205 | 42,331 | 234 / 234 |

Both dense groups include connected structures reaching crop edges. The full
**region containing potentially multiple objects** is the explicit fallback;
neither group is presented as a complete object.

The unchanged CLIP model produced real candidate/context rankings. For the
vertical candidate, bicycle (0.239) narrowly leads utility pole (0.237), with 50%
view disagreement; region context leads with utility pole (0.261). These are
uncalibrated similarities and unverified descriptions, not accuracy evidence.

Remaining issues are **grouping boundaries and model recognition**. Static
structure is now inspectable; local HTML bypasses the missing VS Code renderer.

**One next step:** inspect the vertical structure in the browser and determine
which connected parts belong in one object before resuming labeling.

Stopped after two scouting regions, two selections, one raw-file pass, and one
grouping attempt per crop. No convincing third candidate was added. Previous
experiments, raw files, and reviews were preserved; no human labels were written.
[Settings, provenance and prediction references](outputs/objects/new_candidates/20260911T140448_622999Z/report.json)
are saved alongside the figures and HTML.
