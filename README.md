# Run 1 spatial development pipeline

This workspace contains a read-only profiling script and a reproducible Run 1
sampling/ground-separation pipeline. No raw LAS file is modified.

## Explore the inspection notebook

Open [`inspect_las.ipynb`](inspect_las.ipynb) in VS Code and select a Python kernel
with `laspy`, `numpy`, and `pyproj`. Notebook execution also requires `ipykernel`
in that environment. The notebook explains the inspection code step by step;
`inspect_las.py` remains available for command-line use.

By default, **Run All** reads LAS headers and loads the existing profiling report.
Set `RUN_FULL_PROFILE = True` to scan the configured files in chunks. Report
export is separately controlled by `SAVE_REPORTS` and writes to a new directory
under `reports/notebook/`. The notebook keeps the coordinate-unit conflict visible.

## Compare ground-filter experiments visually

Open [`ground_experiments.ipynb`](ground_experiments.ipynb) for the visual workflow.
Its saved figures and tables use the actual cached sample, baseline, and both CSF
iterations. `inspect_las.ipynb` remains the metadata and statistics notebook.

In VS Code, choose **Select Kernel > Python Environments > .venv-ground**
(`.venv-ground\Scripts\python.exe`). The local kernel display name is
**LiDAR Ground Experiments (.venv-ground)**. This environment has the plotting
and notebook dependencies installed; their declarations are in
[`requirements-ground-notebook.txt`](requirements-ground-notebook.txt).

Run the notebook from the top. Choose `INPUT_SIDE`, edit `ROI_CENTER_LOCAL` and
`ROI_SIZE`, and inspect the top-down map and elevation cross-section. The initial
region is a populated example, not an identified missing-road location. Plot
coordinates subtract the displayed common origin; they remain in unresolved
source coordinate units. Processing uses all cached sample points in the region
plus a margin; display limits only reduce the points drawn.

For your first new experiment, keep the region fixed, change one value in
`CSF_PARAMETERS` (for example `class_threshold`), enable `RUN_NEW_EXPERIMENT`,
and run the parameter, compute, and comparison cells. Every fresh run gets a
unique folder under `outputs/experiments/notebook/`, including its inputs,
parameters, masks, LAS partitions, counts, timing, and comparison table. Enter
your notes in `OBSERVATIONS` and use the review-save cell to record them in a
separate snapshot. Full raw-data processing is not part of this notebook.

The implementation is available in
[`ground_experiment_tools.py`](ground_experiment_tools.py) and
[`ground_experiment_views.py`](ground_experiment_views.py). Filtering reuses the
functions in `sample_and_ground.py`; it does not invoke that script's command-line
entry point. Saved full-sample CSF results and fresh local results have different
processing extents, which the notebook explicitly records.

## Run the pipeline

For candidate-object review, see the notebook workflow at the end of this README.

Use the installed Python environment and run from this folder:

```powershell
$python = "$env:LocalAppData\Programs\Python\Python312\python.exe"
& $python -m pip install -r requirements-inspection.txt
& $python .\sample_and_ground.py `
  ".\Run 1 Laser Left.las" `
  ".\Run 1 Laser Right.las" `
  --inputs-dir outputs\run1_inputs `
  --experiment-dir outputs\experiments\baseline `
  --method baseline `
  --target-points 2000000 `
  --voxel-size 0.25
```

The sampler uses the intersection of the two source 3D header bounds as one
shared ROI. Each source is sampled independently with a deterministic spatial
hash, so rerunning the command produces the same selection for each source.
Original coordinates and point attributes are copied into separate derived LAS
files; source identity is represented by the left/right filenames and manifest.
When the ROI, target point count, and voxel setting match
`outputs/run1_inputs/run1_inputs_manifest.json`, the sample and preview LAS
files are reused. Change those settings only when you intentionally want new
development inputs.

## Outputs

Open these directly in CloudCompare.

Reusable development inputs live in `outputs/run1_inputs/`:

- `run1_left_sample.las`, `run1_right_sample.las`: approximately 2 million point development samples.
- `run1_left_preview_voxel.las`, `run1_right_preview_voxel.las`: optional voxel previews.

Ground-filter experiments live under `outputs/experiments/<method>/`.
The current lowest-point proxy is preserved as `experiments/baseline/`:

- `run1_left_ground.las`, `run1_right_ground.las`: ground candidates.
- `run1_left_non_ground.las`, `run1_right_non_ground.las`: complementary non-ground candidates.
- `baseline_manifest.json`: method, parameters, counts, runtime, and CRS record.

Rerunning the same method with the same experiment directory overwrites only
that method's generated LAS files and manifest. A new method gets a new folder,
such as `experiments/csf/` or `experiments/smrf/`. Raw LAS files are never
overwritten or deleted.

## Ground separation

The first geometry-only stage finds the lowest point in each 1-unit XY cell and
marks points within 0.35 coordinate units of that local minimum as ground. This
is an explainable development proxy, not semantic classification. Review both
ground and non-ground files in CloudCompare before building any detector.

## CRS and units

`pyproj` is used to compare the LAS VLR and supplied sidecar WKT with EPSG:6553.
Physical units remain **unresolved**: source LAS headers declare US survey feet,
while export sidecars say `Export unit: Meter` and declare meter axes. The sidecar
also cites EPSG:6553, which uses US survey feet. The earlier pipeline and manifests
assume meters without converting coordinates; their `_m` parameter names and
`resolved_for_processing` status are not independent verification. The visual
notebook uses the term **coordinate units** and preserves original coordinates.
Confirm horizontal and vertical units before interpreting physical thresholds.

## CSF experiment

The `experiments/csf/` folder uses the recognized Cloth Simulation Filter from
`cloth-simulation-filter`, run on the cached samples. Current parameters are
1.0 coordinate-unit cloth resolution, 0.8 threshold, rigidness 3, time step 0.65,
500 iterations, and slope smoothing enabled. The earlier 2.0 / 0.5 / rigidness 2 /
200-iteration result is preserved under `experiments/csf/previous_2m_rigidness2/`.
Compare both in the visual notebook or CloudCompare. Ground/non-ground separation
does not establish object identities or an accuracy level.

## Review candidate objects with local CLIP

**Current status: manual labeling is paused.** Open the notebook's
**Candidate quality comparison** section for the completed three-region dense
comparison. Start with candidate **00012**. See [the short handoff](OBJECT_QUALITY_HANDOFF.md)
for figures, findings, and the single recommended next step. The quality section
loads saved results without scanning raw LAS or computing new proposals.

Open [`object_review.ipynb`](object_review.ipynb) in the workspace root. Select
**LiDAR Object Review (.venv-ground)** or `.venv-ground\Scripts\python.exe`.
The environment includes CPU PyTorch, Transformers, SciPy, and plotting support;
the public `openai/clip-vit-base-patch32` weights are downloaded locally.

Run from the top to inspect spacing, extract a small batch from saved CSF
non-ground points, and review real RGB views and CLIP similarities. The default
region uses 4,039 non-ground points and presents six candidates. Clusters can be
fragments or merged objects. Scores are not calibrated probabilities, and the
notebook does not assign human labels automatically.

Change `CANDIDATE_INDEX` in section 4 to navigate. Rerun selection, display,
render, and prediction cells; matching caches are reused. In section 7, paste the
displayed candidate ID into `REVIEW_FOR_CANDIDATE_ID`, then choose
`DECISION` (`accept`, `correct`, or `unlabeled`), set `CORRECTED_LABEL` when
correcting, choose `CROP_QUALITY`, and enter `NOTES`. Run those controls, then
enable `SAVE_REVIEW` and run the save cell. Return the switch to `False` afterward.
Reviews write immediately and reload
after a restart; they do not train or alter CLIP.

Artifacts are separated under `outputs/objects/{sources,candidates,renders,inference,reviews}`.
Source and clustering settings determine candidate membership; rendering and model
settings have separate caches. Original sample row numbers and input hashes make
point membership reproducible. Rejected/noise points remain available. Only public
model files are downloaded; point clouds and images are not uploaded.

Implementation: [`object_candidates.py`](object_candidates.py),
[`object_views.py`](object_views.py), and [`object_clip.py`](object_clip.py).
Dependency declarations are in [`requirements-object-review.txt`](requirements-object-review.txt).
To recreate CPU support in the project environment, install PyTorch from its
official CPU index before the remaining dependencies:

```powershell
.\.venv-ground\Scripts\python.exe -m pip install "torch>=2.6,<3" "torchvision>=0.21,<1" --index-url https://download.pytorch.org/whl/cpu
.\.venv-ground\Scripts\python.exe -m pip install -r requirements-object-review.txt
```
