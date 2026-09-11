# Review candidate objects

Double-click **Launch Candidate Review.cmd** in this project folder. It opens the
local review page in your browser and runs the server in the background using the
existing `.venv-ground` Python environment. No notebook kernel is needed. Launching
again reopens the running application.

The address is **http://127.0.0.1:8765/**. Closing the browser tab leaves the server
running. Double-click **Stop Candidate Review.cmd** when finished; saved reviews
remain on disk. If startup fails, the launcher shows the relevant log path.

## Review and resume

1. Enter your reviewer name and inspect the first candidate,
   `b2a866edbb1fac31_00027`. Switch between **Candidate only**, **Context only**, and
   **Combined**; use the camera buttons and the separate candidate/context image
   tiles to check its shape and surroundings.
2. Choose **Object class** and **Crop quality** independently. The defaults are
   **Unknown / cannot identify** and **Unclear**. A recognizable fragment or a
   crop containing several objects may still be unsuitable as a class example.
   Use **Custom class…** to add your own class, and add notes when useful.
3. Click **Save & next →**. The application writes the review immediately and
   advances. **Previous**, **Next**, and **Candidate queue** revisit items without
   saving a new decision. Unsaved drafts survive this navigation in browser
   memory; save before closing or reloading the page.
4. Reopen the launcher to resume the saved position. Existing saved decisions load
   with their candidate. Use **Download reviews CSV** and **Download review
   manifest** for portable snapshots.

Suggestions, model names, and derived geometry evidence start hidden. **Reveal
suggestions & geometry** exposes existing evidence without assigning an object
class or changing the form. The app conservatively records whether a candidate's
suggestions have ever been revealed through this app, including earlier sessions;
it cannot determine exposure in previous notebooks or outside the app. Keep that
distinction when interpreting human labels.

The prepared batch has **20 candidates: five existing dense anchors and 15
components from existing caches**. Five have saved suggestions; the other 15 have
no model predictions. Some crops are fragments. Their locations fall into three
conservative spatial groups, so these are not 20 independent object examples.

The application uses existing candidate artifacts. Coordinates retain the
workspace's unresolved units: LAS headers say US survey feet; export sidecars say
meters. Object labels and review decisions do not resolve that disagreement.

## Saved files

| Location | Contents |
| --- | --- |
| `outputs/objects/reference_set/reviews/` | Human decisions and their source/prediction provenance. |
| `outputs/objects/reference_set/session.json` | Saved position and optional reviewer name; progress is read from saved reviews. |
| `outputs/objects/reference_set/classes.json` | Editable class choices, including classes added through the UI. |
| `outputs/objects/reference_set/exposures/` | Recorded reveals of existing model/geometry evidence. |
| `outputs/objects/reference_set/exports/` | Latest `reference_set.csv` and `manifest.json`, refreshed after each save. |
| `outputs/objects/reference_set/server_*.log` | Server output and separate error logs for each launch. |

Keep the reference-set folder together with its original candidate artifacts.
Candidate membership and source identifiers connect a review to the exact points
that were inspected. Existing notebook reviews remain associated with their
original candidate memberships. Earlier review revisions remain in history when
you revise a decision; refreshed exports do not erase that history.
To edit `classes.json` directly, stop the application first and relaunch it afterward.

For troubleshooting, run the server in a visible PowerShell window from this
folder:

```powershell
& ".\.venv-ground\Scripts\python.exe" ".\review_app.py" --port 8765
```

Use `--no-browser` to suppress opening a tab. `--data-dir "C:\path\to\temporary reviews"`
stores test decisions separately from the real reference set. Stop an existing
server first, or select a different port.

## Later evaluation

Use only human-confirmed, usable candidates with explicit class labels when
building a future class dataset. Keep rejected, uncertain, and incomplete crops
in the review record so the acceptance rate and selection decisions remain
visible. No training or accuracy estimate is produced by this review workflow.

Before any train/holdout split, assign related candidates to the same group:
overlapping crops, shared source points, repeated views or samples of the same
object, and corresponding left/right observations. Follow these links
transitively: if A overlaps B and B overlaps C, all three stay together. Split
whole groups, never neighboring points or individual renders from one object.
The current batch contains left-scan data only. Verify corresponding objects
before including right-scan observations. In **Optional: link a shared object**,
give related candidates the same tag to join their evaluation groups. Such tags
can join the automatic groups but cannot split them.
This proposed grouping applies the principle of keeping dependent observations
together during evaluation. [Grouped evaluation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)

Freeze the holdout groups before tuning prompts, extraction settings, or a model.
The five existing anchors have already informed development. Hiding suggestions
in this interface does not make those examples independent held-out evidence.
Report confirmed sample counts and independent group counts by class, together
with crop-quality exclusions. Separate results for reviews made before and after
model suggestions were revealed; check whether class and source coverage differ.
A small, selectively reviewed batch supports a preliminary qualitative check;
it does not establish object-classification accuracy for the corridor.
