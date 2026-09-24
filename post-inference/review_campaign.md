# Collaborative Review Campaign — spec SSoT

How a small team traverses **every** polygon in the South 2025Q3 candidate inventory and turns it
into a human-verified RTS inventory. This file is the single source of truth for the campaign
**protocol** — queue construction, claiming, verdict semantics, merge rules, output schema.

It is **not** the SSoT for the products it consumes or produces as data: inventory facts, counts,
tiers and caveats live in `south_products.md`; the size bands and precision-grid maths live in
`scripts/score_qc_ratings.py`; the `rts_class` rule lives in
`scripts/export_south_products.py:assign_rts_class`.

## 1. Objective

`south_rts_candidates.gpkg` carries no human verification. The only human signal that exists is the
2026-07 solo QC pass — a 280-polygon stratified sample, all 280 rated — and every precision number in
`south_products.md` is an extrapolation from it. This campaign replaces that extrapolation with a
**census**: a verdict on all 60,167 polygons, contributed by 2–3 reviewers working in parallel.

## 2. Scope (locked)

| Decision | Value |
|---|---|
| Inventory | `south_rts_candidates.gpkg` — all **60,167** polygons, threshold 0.30, MMU≈0 |
| Verdict vocabulary | `rts` · `false` · `unsure` — identical to `qc_ratings.csv`, so the two rounds pool |
| Per-item output | **verdict only** — no FP reason codes, no geometry flags, no boundary redrawing |
| Direction | **precision pass** — reviewers judge what the model found; missed slumps are out of scope, so recall on 2025 imagery stays unmeasured |
| Reviewers | 2–3 |
| Host | open GCE VM, no sign-in — <http://8.229.247.193/> (§10.3) |

Effort: ≈134 person-hours at ~8 s/item, i.e. 45–67 h per reviewer. The campaign is therefore designed
to be **stoppable at any point with a defensible product** (§3).

## 3. Queue order — `max_prob` descending

Batches are ordered by descending `max_prob`. The property this buys is that at any moment the
campaign's claim is exactly:

> every polygon with `max_prob ≥ p` is human-reviewed

which is a statement in the attribute the product family already exposes, reproducible by any user
with a one-line filter. Area-descending would front-load km² slightly better but yields a cut running
across the confidence tiers, which is harder to state and harder to use. Probability-descending
front-loads area anyway — the high tier is 32% of objects but 77% of the 688.2 km².

Two properties of this ordering that must be briefed, not papered over:

1. **The head of the queue is not the precision peak.** `south_products.md` caveat 1: object
   precision *peaks* at 0.65 and falls above it — confident look-alikes survive any cut while true
   detections fragment. The first batches are not "the easy true ones"; `max_prob ≈ 1.0` is where the
   hardest false positives live.
2. **Response bias.** A probability-sorted queue produces long runs of the same correct answer, which
   anchors raters. Mitigated by **shuffling item order within each batch** (seed 42). Batches stay
   probability-ordered, so the `≥ p` property survives at batch granularity, but no reviewer is shown
   200 consecutive items that are all the same call.

**Batch size 200** — roughly 25–30 minutes of rating. Large enough that claim overhead is
negligible, small enough that a reviewer who walks away strands little work.

## 4. Artifact layout

```
gs://rts-arctic-usw1/inference/2025q3_south/
  internal/review_crops/<rts_id>_t.jpg        tight crop  (3× the feature, ≥250 m)
  internal/review_crops/<rts_id>_w.jpg        wide  crop  (10× the feature, ≥1.5 km)
  internal/review_crops/<rts_id>_t_plain.jpg  the same two views without the
  internal/review_crops/<rts_id>_w_plain.jpg  red outline (§4.2)
  review/manifest.parquet                   the queue (§5)
  review/claims/<batch_id>                  ClaimStore, atomic create-if-absent
  review/done/<batch_id>                    ClaimStore, source of truth on restart
  review/verdicts/<batch_id>.jsonl          immutable, written once per completed batch
```

Crops live under `internal/`, never `products/`. They are PlanetScope-derived and marked
*not for redistribution* — the same reason `rgb_chips/` was moved out of `products/` in the
2026-07-18 bucket audit (`south_products.md`). Rendering is by `scripts/build_review_crops.py`, whose
crop geometry comes from `review/crops.py`, shared with the offline pack builder
`scripts/build_qc_rating_page.py` so the two views of a polygon are pixel-identical. (The shared
module lives in `review/`, not `post-inference/`, because the latter's hyphen makes it unimportable
as a Python package.)

### 4.1 The chip archive had to be rebuilt first

`internal/rgb_chips/` as it stood on 2026-08-03 held **29,850** chips, built by
`scripts/build_rgb_chips.py` against the 10,984-polygon **0.65** product `south_rts.gpkg`. The 0.30
candidate inventory references **118,586** tiles, so that archive covered only **17,980 of 60,167
polygons (29.9%; 452 of 688.2 km²)** — the other 70% would have been served blank crops and rated
`unsure`, silently gutting the campaign.

Fixed by re-running the same script against `south_rts_candidates.gpkg`. Three changes it needed to
work at this scale, all in `scripts/build_rgb_chips.py`:

- `--workers` — the work is network-bound quad reads (~0.5 s each), so 88,736 tiles is ~12 h serial
  and minutes in parallel;
- skip-existing by default, so the run resumes and the 29,850 existing chips are not re-fetched;
- `gdalbuildvrt -input_file_list` — 118k paths overflow the command-line length limit.

**Reading a crop is indexed, not scanned.** A windowed read against the mosaic VRT costs ~2.1 s
because GDAL walks its source list once per window — 62 h for the archive. `chip_index()` parses the
VRT's sources once, and each polygon's crops are read from a micro-VRT of the two or three chips it
touches: **0.070 s/polygon** of worker time, and the full inventory rendered in ~7 min on 90 workers.

Polygons still without imagery (quad coverage gaps — `south_products.md` caveat 8) are listed in
`no_imagery.csv` and are **still served**; the reviewer brief says to rate them `unsure`. Emptiness is
determined from the source pixels (`review.crops.has_imagery`), not from the rendered JPEG: the
burned-in red outline and its antialiasing make a rendered empty crop indistinguishable from a dark
real one.

**Measured result (2026-08-03).** All 60,167 polygons rendered; **20 (0.03%)** have no imagery, down
from the ~70% the old archive would have produced. Archive: 120,334 JPEGs, **3.2 GB**, mean 23.1 KB
per crop. Zero polygons are missing a crop file.

That measurement was right about the *tight* view and blind to the wide one, and the blindness
mattered.

#### The context gap (found 2026-09-23)

`has_imagery` probes the **tight** window only, so a polygon whose object is visible but whose 1.5 km
surroundings are half black passed the check silently. Sampling 150 rendered polygons found **44% of
wide views at least 10% black and 15% more than half black**, concentrated almost entirely in the
high-confidence band:

| conf_class | mean black | >=10% black | >50% black |
|---|---|---|---|
| high (p >= 0.652) | 33.7% | **73.3%** | 28.3% |
| medium | 22.6% | 51.4% | 17.1% |
| low (p < 0.452) | 3.0% | 7.3% | **0%** |

The cause is not a failed download and not a stale render. `build_rgb_chips.py` chipped only the
tiles a *detection* references, and the wide crop is `10x` the feature — so it reaches tiles no
detection sits in. Measured against the real geometry of all 60,167 polygons and all 118,586 chip
names:

- the wide views need **621,141** tiles; **118,586** were chipped, **502,555** were not;
- of those 502,555, **496,556 (98.8%) sit on quads already in the bucket** — never chipped, not
  missing;
- only **5,999 (1.2%)** have no downloaded quad. They fall in 175 quad cells, of which **173 sit on
  the edge of the downloaded footprint** (coastline, AOI boundary) and **2 are fully enclosed holes**
  — 75.65N/112.94E and 71.50N/52.47E, 91 tiles between them. Those two are the only genuine
  acquisition gaps in the whole inventory.

The size correlation explains the confidence-class pattern entirely: high-confidence polygons are
larger (median extent 592 units against 91 for low), so `10x` sends their context window far past the
chipped tiles. Median wide side by class: high 7,309 units needing 36 tiles with 6 present; low 1,500
units needing 6 with 4 present. **Only 12.8% of polygons had every tile their wide view
needed** (400-polygon sample; the 621,141 / 118,586 / 502,555 tile counts above are exact,
taken over all 60,167).

**Fix.** `build_rgb_chips.py --wide-context` selects tiles by each polygon's wide crop bounds
(`review.crops.crop_bounds`, the same function the renderer uses) instead of by the polygon, and
computes tile bounds from the stride grid rather than the domain tile list, because a context window
legitimately runs past the inference AOI. Then re-render **with `--overwrite`**: the resume check in
`build_crops` skips any polygon that already has its four JPEGs, so without it a rebuilt chip archive
changes nothing.

**New tripwire.** `review.crops.imagery_fraction` measures how much of a window the mosaic carries,
and `build_review_crops.py` writes every polygon under 90% to `partial_context.csv`. `no_imagery.csv`
keeps its all-or-nothing meaning; the new file is what would have caught this in August.

**Stale artifact.** `inference/2025q3_south/quad_index_2025q3.csv` points every row at
`gs://pdg-planet-data/...`, deleted in the migration (404). Regenerate it against
`gs://rts-arctic-usw1/global_quarterly/2025/q3/` with `scripts/build_quad_index.py` before any
chipping run.

**Measured result of the rebuild (2026-09-24).** Run on a temporary `n2-standard-32`
(`rts-chips`, deleted afterwards), against a quad index regenerated to **309,102** quads:

| | before | after |
|---|---|---|
| chips in the archive | 118,586 | **621,158** (0 errors, 33 min at 238 chips/s) |
| `partial_context.csv` | n/a | **2,463 (4.09%)**, from ~87% of polygons lacking full context |
| `no_imagery.csv` | 20 | **20** — unchanged, as expected |
| wide views ≥10% missing (n=400 pixel sweep) | 44% | **11%** |
| wide views >50% missing | 15.3% | **1.2%** |
| wide views fully clean | 56% | 68% |

What remains is real and now *visible*: the flagged crops show a straight quad-footprint boundary,
striped and captioned, which matches the finding that 173 of the 175 absent quad cells sit on the
edge of the downloaded footprint.

**Cutover done 2026-09-24**, server-side copy of `review_crops_v2/` over `review_crops/`, verified
with `scripts/gcs_parity.py` — **PARITY PASS**, 240,668 objects / 7,167,023,924 bytes identical by
name, size and checksum. Note that `gcloud storage rsync` left **55,758 objects at their original
timestamp**: those renders are byte-identical, because a tight `_plain` crop of an isolated,
fully-covered polygon gains no outline, no neighbour and no striping. The skew confirms it — 30,382
`_t_plain` and 15,954 `_t` against only 2,047 `_w`, the variant that gained the most. An unchanged
timestamp is not a skipped copy, but sampling could not prove that, which is why the full walk ran.

Artifacts: crops staged at `internal/review_crops_v2/` (240,668 objects, 7.2 GB — larger than the
old 3.2 GB because previously-black pixels now carry imagery), chips at `internal/rgb_chips/`
(621,158, 311 GB), `internal/rgb_chips.vrt` (758 MB), and the regenerated index at
`quad_index_2025q3_regen.csv`. **Cutover is a server-side copy of `review_crops_v2/` over
`review_crops/`**, deliberately left as a separate step because the campaign is live; the bucket's
7-day soft-delete policy covers a rollback. The app must be redeployed afterwards so `CROP_VERSION`
busts the 24-hour browser cache.

### 4.2 The outline toggle

The red outline is **drawn into the pixels** by `render_crop`, so it cannot be switched off in the
browser. `outline=False` renders the same window without it, and the archive therefore carries four
JPEGs per polygon rather than two. The app hands the client both URLs and the `o` key (or the
*outline* button) swaps between them.

Why it is worth the extra 3 GB: an outline is a claim about where the feature ends, and a reviewer
judging whether a slump is real should be able to see the imagery unannotated. The setting is
**sticky** across items and reloads, so working a whole batch unoutlined is a deliberate mode rather
than a per-item peek. The outlined view stays the default, and the prefetcher warms whichever pair is
on screen.

### 4.3 Neighbouring detections, and absence made visible

Two things a reviewer could not see before 2026-09-23.

**Neighbours.** Other candidates falling inside a crop are outlined in `NEIGHBOUR_COLOR` (`#3fd0ff`,
0.9 px, 70% alpha) beneath the target's red — so a reviewer can tell a lone feature from one of a
cluster, which is most of what "context" means here. All 60,167 candidates count as neighbours, down
to p = 0.30. They are drawn in both the tight and wide outlined views and **suppressed entirely in
the `_plain` pair**, because that view exists for judging pixels unprimed. The colour is chosen to
fail the red-pixel predicate `tests/test_review_crops.py` uses to police the toggle, so that test
keeps its meaning.

**Absence.** A boundless read fills 0 and renders black, which is also what dark water and shadowed
ground look like once JPEG has been at them — the reason §4.1's context gap went unnoticed for seven
weeks. Missing pixels are now drawn as diagonal grey striping (`NO_DATA_DARK`/`NO_DATA_LIGHT`, 8 px
period) and captioned `NO IMAGERY <pct>` when more than 2% of the window is absent. The caption is a
fact about the imagery, not a hint about the model, so it appears in the `_plain` pair too —
otherwise that view would still be lying.

Workers get neighbours through the pool initializer, the same route as the chip index: the parent
builds `_polygon_index(gdf)` once and each worker filters it by bbox (`_neighbours_for`), because a
worker is handed only the polygon it is rendering.

## 5. The manifest

`scripts/build_review_manifest.py` → `review/manifest.parquet`, one row per **queue item**:

| Column | Meaning |
|---|---|
| `rts_id` | polygon id from `south_rts_candidates.gpkg` |
| `batch_id` | `b00000` … — batches numbered in `max_prob`-descending order |
| `seq` | position within the batch (shuffled, seed 42) |
| `injected` | `False` for the item's coverage assignment; `True` for a replicate copy (§7) |
| `max_prob`, `conf_class`, `area_m2`, `centroid_lat`, `centroid_lon` | context shown to the reviewer |

Coverage invariant: **every `rts_id` appears exactly once with `injected == False`.** Injected rows
are additional and are excluded from all coverage accounting. The build is deterministic — re-running
it on the same inventory produces a byte-identical file.

## 6. Claim protocol

`review/store.py` wraps `inference/claim.py:ClaimStore` **unchanged**. A review batch is claimed
exactly as an inference shard was: `if_generation_match=0` create-if-absent, so two reviewers can
never hold the same batch. `done` markers are the source of truth on restart.

**A review claim lasts one week** (`STALE_AFTER_S = 604800.0` — seconds, measured from the last
heartbeat). The inference queue reclaimed a shard after 30 minutes because a crashed worker's shard
was pure loss. A reviewer is not a crashed worker: their part-rated batch is sitting in their
browser's `localStorage` and is still good tomorrow, so a short TTL would hand the same 200 polygons
to a second reviewer and have them rated twice. A week covers a weekend and a holiday — close the
tab, go home, come back Monday — while still letting a truly abandoned batch return to the pool on
its own rather than needing §6.1.

The heartbeat still runs every 60 s and pushes the expiry out, though at a week's TTL that rarely
matters. Its practical value is the timestamp: the only record of when a held batch was last touched.

**Changing the TTL needs a redeploy, not just an edit.** The app runs from a container image, so the
constant is baked in at build time — edit `review/store.py`, then `scripts/deploy_review_vm.sh`.

### 6.2 The TTL needs an IAM grant, and ran dead without it

From 2026-08-04 to 2026-09-23 **neither the heartbeat nor the stale reclaim did anything**, on a
runtime identity holding `roles/storage.objectCreator` + `roles/storage.objectViewer`.
`objectCreator` grants create-if-absent and nothing else — **not overwrite, not delete** — so:

| call | operation | what happened |
|---|---|---|
| `heartbeat()` | overwrite | raised; `/api/heartbeat` 500'd; the browser beat has no `.catch`, so it was invisible |
| `reclaim_if_stale()` | delete | a blanket `except` swallowed the 403 and returned False, every time |
| `mark_done()` claim cleanup | delete | also failed, but is best-effort — which is why 108 claim objects existed for 88 done batches |

The signature in the bucket: every claim read `claimed_at == heartbeat_at` to the microsecond, and a
claim opened that morning was still `metageneration=1` after a day of 60-second beats. The cost was
**18 batches (3,600 polygons) stranded**, and because the oldest was `b00001` the contiguous-prefix
headline (§9) sat frozen at `max_prob >= 1.0` while 29% of the inventory was rated.

Granted 2026-09-23, and verified live — `/api/next` handed back the 49-day-stale `b00001`, and a
heartbeat then advanced `heartbeat_at` past `claimed_at`:

```bash
gcloud iam roles create rtsReviewClaimReaper --project=abruptthawmapping     --title="RTS review claim reaper" --permissions=storage.objects.delete --stage=GA

gcloud storage buckets add-iam-policy-binding gs://rts-arctic-usw1     --member=serviceAccount:rts-review-app@abruptthawmapping.iam.gserviceaccount.com     --role=projects/abruptthawmapping/roles/rtsReviewClaimReaper     --condition='title=claims-prefix-only,expression=resource.name.startsWith(
        "projects/_/buckets/rts-arctic-usw1/objects/inference/2025q3_south/review/claims/")'
```

Uniform bucket-level access is on, so the condition applies; the blast radius is claim objects,
which are reconstructible, and the bucket carries a 7-day soft-delete policy. Two guards stop it
rotting again: `reclaim_if_stale` now catches only `NotFound` (the lost-the-race case) so a 403
surfaces, and the app probes create/overwrite/delete against the claims prefix at start-up and logs
an error naming the missing permission. `tests/test_claim.py` carries a create-only fake bucket that
reproduces the failure — the original fake modelled atomicity but not IAM, which is why no test
could have caught this.

Reviewer lifecycle:

```
GET  /api/next        claim the first unclaimed batch in queue order → items + crop URLs
GET  /crop/<key>      one crop JPEG, streamed from the bucket (§10)
POST /api/heartbeat   every 60 s while a batch is open
POST /api/batch       submit 200 verdicts as one JSONL → mark_done → claim is released
GET  /api/progress    campaign state (§9)
```

Verdicts buffer in `localStorage` on every keystroke and are submitted **once**, at batch end. A
reload, a dropped connection, or a Space that went to sleep costs nothing — this is the failure that
lost a full rating round in the GEE-rater era (`south_products.md`, and the docstring of
`scripts/build_qc_rating_page.py`). Re-submitting an already-`done` batch is accepted and ignored;
submission is idempotent.

### 6.1 Releasing a stranded batch

A batch claimed by someone who has genuinely stopped stays out of the pool for a week and blocks the
contiguous-prefix headline (§9). To get it back sooner, delete one object — the batch returns to the
queue with no verdicts lost, because nothing was ever written:

```bash
# who holds what, and when they last touched it
gcloud storage ls gs://rts-arctic-usw1/inference/2025q3_south/review/claims/
gcloud storage cat gs://rts-arctic-usw1/inference/2025q3_south/review/claims/b00042

# release it — only after checking with the holder, who may still have it open
gcloud storage rm gs://rts-arctic-usw1/inference/2025q3_south/review/claims/b00042
```

Check with the holder first. Their unsubmitted verdicts live in their browser, and releasing the
claim does not delete them — if they later submit, the batch is accepted from whoever gets there
first and the loser is told it was already completed. Duplicated *effort* is possible; a corrupted or
double-counted *record* is not.

**The model's probability is never shown to the reviewer.** A rater told the model is 99% confident
is primed to agree, and this campaign exists to *measure* the model, not to confirm it. The UI shows
only the batch position and the polygon's area. `max_prob` still drives the queue order, so a
determined reviewer could infer a tier from position — a far weaker cue than a number on screen next
to the buttons.

## 7. Quality control of the QC

Two checks, together **0.5% overhead** (~300 extra ratings):

- **~300 injected replicates.** An item already rated by one reviewer is injected into a later batch
  belonging to a *different* reviewer. Item-level, not whole duplicate batches: at 0.5% a batch-level
  scheme would be 1.5 batches and useless for κ, whereas 300 paired judgements spread through the
  campaign give Cohen's κ with SE ≈ 0.05 — enough to separate substantial agreement from poor.
  Injected items are indistinguishable from ordinary items in the UI.
- **Agreement with the 2026-07 pass — free.** The 280 already-rated polygons are inventory members
  that get reviewed in their natural probability position anyway, so the merge simply joins
  `qc_ratings.csv` against the new verdicts and reports the confusion matrix. No extra ratings, and
  it still catches systematic drift between the two rounds.

**Adjudication.** Replicate disagreements plus every `unsure` are re-served by the same app under a
filtered manifest, to the domain expert. No new code path — a manifest flag.

## 8. Merge and output

`scripts/merge_review_verdicts.py` reads every `review/verdicts/*.jsonl` and writes:

| Output | What |
|---|---|
| `review_verdicts.csv` | `rts_id, qc_verdict` — the pooled census, shaped exactly like `qc_ratings.csv` so `scripts/score_qc_ratings.py` consumes it unchanged |
| `south_rts_verified.gpkg` | the candidate schema plus `qc_verdict`, `n_reviews`, `reviewers`, `agreement`, `reviewed_at` |
| `south_rts_verified_true.gpkg` | `qc_verdict == 'rts'` — the headline human-verified inventory |
| `qc_false_hard_negatives.gpkg` | every `false` — extends the v3 hard-negative seed set beyond the current 152 |

Merge rules:

- A verdict whose `rts_id` is not in the manifest is a **hard error**, not a silent drop.
- Where an item has both a coverage verdict and an injected replicate verdict, the **coverage**
  verdict is authoritative; the replicate contributes only to κ. `agreement` records whether they
  matched.
- Polygons with no verdict yet keep `qc_verdict = NULL` and are excluded from `..._verified_true`.
  A partial campaign therefore produces a valid, honestly-labelled product.

Re-scoring the census through `score_qc_ratings.py` yields the same tier × size grid the sampled pass
produced, with Wilson intervals collapsing to near-zero width. `assign_rts_class` is untouched — the
verified layer *adds* columns; it does not redefine the model-derived class rule.

## 9. Progress reporting

`GET /api/progress` and the merge report both state:

- **headline** — the lowest `max_prob` for which every polygon is reviewed (the §3 claim)
- polygons reviewed / 60,167, and the verdict split
- % of the 688.2 km² inventory area reviewed (free to compute, so reported)
- per-reviewer counts, and running κ once ≥30 replicate pairs exist

## 10. Deployment

The app is a plain FastAPI container (`computing/Dockerfile.review`, ~690 MB) serving `review/app.py`
on port 7860. It is deliberately host-agnostic: all state is in GCS, so where it runs is reversible.

**Whichever host, no imagery is ever stored on it.** Crops are **streamed from GCS through the app**
(`GET /crop/<key>`) and never written to the host's disk; it holds only the manifest slice for the
open batch and the verdicts in flight. The proxy checks every requested key against the crop prefix,
so it cannot be turned into a reader for the rest of the bucket.

An earlier draft handed the browser **signed GCS URLs** instead. That is now rejected on two counts,
and §10.4 records why the first one is decisive:

- signing needs either a service-account key or `iam.serviceAccounts.signBlob` on the runtime
  identity, and **this project's operators can grant neither**;
- a signed URL is a bearer token — for its lifetime, anyone it reaches can fetch the pixels. Proxied
  crops stay behind the same front door as everything else, which matters for a licence-restricted
  PlanetScope derivative.

The cost of proxying is one extra hop for a ~30 KB JPEG: **169 ms per crop** measured end-to-end from
outside the bucket's region (2026-08-03), against ~8 s of human judgement per item, with five items
prefetched ahead. Whole-campaign egress is ~3 GB.

### 10.1 Host: Cloud Run behind IAP

**Throughput does not bear on this choice.** The rating loop is entirely client-side: `rate()`,
`nav()` and `show()` make no network calls, `/api/next` returns all 200 items in one response,
verdicts accumulate in `localStorage`, and one POST submits the batch. Rating itself is **zero
requests**; the only traffic is the batch fetch, a 60-second heartbeat, and image loads the browser
issues five items ahead of the reviewer. An earlier version of this document rejected options on
per-item round-trip cost; that argument was wrong, because this design has no per-item round trip.
The decision rests on cost, credentials and setup instead.

| | Cloud Run + IAP (**chosen**) | HF Docker Space | GCP VM |
|---|---|---|---|
| Cost | **~$0** — free tier covers the traffic; scales to zero | $9/mo PRO | ~$13/mo `e2-small` |
| Key material | **none** — bucket read/write only, no signing permission | a service-account key in Space secrets | none, if it reads with its own identity |
| Reviewer accounts | **existing Google accounts** | new HF accounts | whatever auth you build |
| Attribution | **authenticated** (IAP assertion) | self-declared name | self-declared name |
| Imagery | stays in-project | third-party host | stays in-project |
| Setup | one script, **but two IAM grants need an admin (§10.4)** | push a repo | IP + TLS + auth |

Cloud Run also needs no concurrency work: claiming is already atomic in GCS
(`if_generation_match=0`), so the app is safe to autoscale by construction.

Since 2026 both Gradio and Docker Spaces "run on compute and require a paid plan to create: PRO for
personal accounts, Team or Enterprise for organizations". Only **static** Spaces are free, and a
static Space cannot hold credentials — its variables *and secrets* are readable from the browser via
`window.huggingface.variables` — so it could only work by giving each reviewer direct GCS access
through browser-side Google OAuth. `scripts/build_hf_space.sh` keeps the Space path working as a
documented fallback.

The offline pack builder (`scripts/build_qc_rating_page.py`) remains working as an escape hatch for a
reviewer who cannot get an account or needs to work offline; it reads the same crop geometry and
emits the same verdict CSV shape.

### 10.2 Identity

Behind IAP the reviewer is taken from the verified `x-goog-iap-jwt-assertion` and **the client cannot
override it** — attribution feeds the inter-rater κ and the product's audit trail, so it must not be
a field anyone can type. The plain `X-Goog-Authenticated-User-Email` header is documented as
compatibility-only, so it is never trusted on its own. Without an assertion (local runs, the offline
pack, a host without IAP) the supplied name is used, and `/api/me` reports
`authenticated: false` so the UI can say so. A *present but unverifiable* assertion is a 401, never a
silent fallback.

### 10.3 Host in use: an open GCE VM

**The campaign runs on `rts-review` at <http://34.83.225.204/>, with no sign-in.** (It ran on
`rts-review-vm` at `8.229.247.193` until the 2026-08-28 migration; that VM and its PDG static IP
are gone.) Cloud Run could
not be opened — see §10.4: `--allow-unauthenticated` grants `roles/run.invoker` to `allUsers`, which
is the same `run.services.setIamPolicy` that blocked the reviewer grants, and it was tried and
refused. A VM's front door is a firewall rule rather than an IAM policy, which the operators *do*
control, so it is the only open path needing no project admin.

Chosen by the user on 2026-08-04 with the trade-offs stated. What it costs:

- **No authentication.** Anyone who reaches the IP can rate, and can read the crops. The URL is
  unlisted and internally held, which is the whole of the access control.
- **Attribution is self-declared** — the typed name, not a verified identity. κ and the audit trail
  are only as good as reviewers being honest about who they are, which for a 4-person internal team
  is fine; do not read more into the κ than that.
- **No TLS.** A bare IP cannot hold a real certificate, so traffic is clear text. **This is now
  an availability risk, not just a confidentiality one** (found 2026-09-24): Chrome enables
  HTTPS-Upgrade by default *in Incognito*, silently rewrites the URL to `https://`, and the app
  has nothing on 443 — so the page fails with `ERR_TIMED_OUT` and looks like an outage rather
  than a scheme problem. Verified: `https://` times out, `http://` returns 200. Incognito is
  therefore **not** a usable way to bypass a stale cache here. If that enforcement reaches
  normal browsing windows, the campaign goes dark for every reviewer at once. The cheap fix
  when it matters is a DNS name plus a managed certificate, or fronting the VM with a
  load balancer that terminates TLS.
- **~$13/mo** always-on, and patching and restarts are yours. `--restart always` plus a boot-time
  startup script means a reboot restores service unattended.

The image travels through the campaign bucket as a tarball rather than Artifact Registry, because
granting the runtime SA registry access would be another blocked IAM write. `docker load` at boot,
then `docker run`.

The Cloud Run service stays deployed and IAP-fronted but unreachable. Both hosts read the same GCS
state, so if the two bindings ever land, switching is just telling reviewers a different URL —
claims, verdicts and progress carry over untouched.

### 10.4 What this project's operators may grant

The first deploy attempt (2026-08-03) failed on `iam.serviceAccounts.setIamPolicy`. Probing
`projects:testIamPermissions` as `yyang@woodwellclimate.org` showed the shortfall is general — the
account holds no `setIamPolicy` of any kind on this project:

| Permission | Held | Needed for |
|---|---|---|
| `iam.serviceAccounts.create`, `actAs` | ✅ | creating and attaching the runtime SA |
| `storage` admin on `rts-arctic-usw1` | ✅ | the bucket bindings (already applied) |
| `run.services.create/update` | ✅ | deploying |
| `artifactregistry.repositories.uploadArtifacts` | ✅ (**the account, not Cloud Build**) | pushing the image |
| `clientauthconfig.brands.create` | ✅ | the OAuth brand — and one **already exists**, so the prerequisite flagged earlier is moot |
| `iam.serviceAccounts.setIamPolicy` | ❌ | signBlob self-impersonation — **designed out**, see §10 |
| `run.services.setIamPolicy` | ❌ | letting the IAP service agent invoke the service |
| `iap.webServiceVersions.setIamPolicy` | ❌ | granting reviewers `iap.httpsResourceAccessor` |

A third wall appeared at the image push: **Cloud Build** runs as the compute default service account,
which lacks `artifactregistry.repositories.uploadArtifacts` on `pdg-artifact-registry`, and granting
it needs an IAM write we do not hold. The operating account *can* push, so the script builds the
image locally and pushes it directly — Cloud Build is out of the path entirely, and the image
deployed is byte-for-byte the one tested here.

Two further consequences:

1. **Signing was designed out rather than waited on.** Streaming crops through the app needs only the
   bucket read that is already granted, so the missing permission stops mattering — and the ask
   shrinks by one role.
2. **The remaining two are unavoidable for any Google-authenticated host.** They are IAM policy
   writes on the service and on IAP; without them the service deploys but nothing can reach it.
   `scripts/deploy_cloud_run.sh` therefore treats both as best-effort and reprints any refused
   binding as a verbatim command for an admin. The ask is `roles/run.admin` + `roles/iap.admin` on
   `pdg-project-406720`, or those two bindings run on the operators' behalf.

Note that a service-account **key** *could* be minted (`iam.serviceAccountKeys.create` is held). That
is deliberately not done: a key is long-lived credential material for a bucket holding the whole
inventory, and it would not solve the access problem anyway — only the signing one, which no longer
exists.

## 11. Runbook

### 11.1 One-time build

```bash
# 0. chips for every referenced tile (§4.1) — skip-existing, so this is a no-op
#    once the archive is complete. Measured 2026-08-03: 88,736 chips in 13 min
#    at 64 workers (~114 chips/s, 0 errors), then ~4 min to rebuild the VRT.
python scripts/build_rgb_chips.py \
    --gpkg .../south_rts_candidates.gpkg \
    --tile-list .../tiles_2025q3_domain_full.csv \
    --quad-index .../quad_index_2025q3.csv \
    --out-dir .../review --workers 64

# 1. crop archive — measured 2026-08-03: all 60,167 polygons in ~7 min on 90
#    workers (120,334 JPEGs, 3.2 GB). Resumable: an interrupted run continues.
python scripts/build_review_crops.py \
    --candidates .../south_rts_candidates.gpkg \
    --chips-vrt  .../rgb_chips.vrt \
    --out-dir    .../review_crops --workers 90
gsutil -m rsync -r .../review_crops \
    gs://rts-arctic-usw1/inference/2025q3_south/internal/review_crops/

# 2. queue
python scripts/build_review_manifest.py \
    --attributes .../south_rts_attributes.parquet --out .../manifest.parquet
gsutil cp .../manifest.parquet \
    gs://rts-arctic-usw1/inference/2025q3_south/review/manifest.parquet
```

### 11.2 Deploy — the open VM (in use)

```bash
scripts/deploy_review_vm.sh
```

Idempotent, and the whole thing: builds the image, ships it to
`internal/deploy/rts-review.tar.gz`, opens `tcp:80` to `0.0.0.0/0` for tag `rts-review`, reserves the
static IP, creates the VM with a boot-time startup script, then polls until `/api/progress` answers
200. Re-running on an existing VM refreshes the metadata and resets it, so a rebuild is one command.

**Live 2026-08-04: <http://8.229.247.193/>** — `rts-review-vm`, `e2-small`, `us-west1-a`, running as
`rts-review-app@`, static IP `rts-review-vm-ip`, firewall rule `rts-review-allow-http`. Verified over
the public IP: the rater page loads, `/api/progress` reports **301 batches / 60,167 items**,
`/api/me` correctly reports no identity (so the UI prompts for a name), a crop serves as a 30,647-byte
JPEG, and the manifest is still refused through `/crop` with a 404. No batch was claimed during these
checks, so campaign state is untouched.

**Reviewers need nothing but the link** — no account, no install, no SSH. Open it, type a name once
(kept in `localStorage`), rate with `1`/`2`/`3` and the arrow keys.

Operating it:

```bash
# boot / container log
gcloud compute instances get-serial-port-output rts-review-vm --zone us-west1-a | tail -40

# stop the meter between sessions (the static IP and the queue survive)
gcloud compute instances stop rts-review-vm --zone us-west1-a
gcloud compute instances start rts-review-vm --zone us-west1-a
```

Because the endpoint is unauthenticated, `GET /api/next` will claim a batch for **anyone** who calls
it. It is not linked from the page — the UI calls it from JavaScript — so crawlers will not trip it,
and a claim goes stale after 30 minutes and returns to the pool. Keep the link internal.

### 11.3 Deploy — Cloud Run behind IAP (parked)

```bash
scripts/deploy_cloud_run.sh reviewer1@woodwellclimate.org reviewer2@woodwellclimate.org
```

**Deployed 2026-08-03: <https://rts-review-bl6ow4qsaa-uw.a.run.app>** (revision `rts-review-00001`,
`Ready`, IAP on, running as `rts-review-app@`). Confirmed from the Cloud Run logs that the container
starts there and reads the campaign with only the bucket roles — *"campaign loaded: 60467 items in
301 batches"* — and that an unauthenticated request is intercepted by IAP and redirected to Google
sign-in (302 to `accounts.google.com`), not served. It is **not usable yet**: see the refused
bindings below.

The script is idempotent: it creates the runtime service account, grants `objectViewer` +
`objectCreator` **on the bucket only**, builds and pushes the image, and deploys with
`--no-allow-unauthenticated --iap` in `us-west1` (the bucket's region). It then attempts the two
access bindings — IAP service agent → `run.invoker`, and each reviewer →
`iap.httpsResourceAccessor`.

**Those two were refused** on 2026-08-03 under the operators' current roles (§10.4) — the deploy
itself also warns `Setting IAM policy failed`. The script does not abort; it finishes the deploy and
prints each refused binding as a command to hand to a project admin, e.g.

```bash
gcloud iap web add-iam-policy-binding --member=user:<email> \
    --role=roles/iap.httpsResourceAccessor \
    --region=us-west1 --resource-type=cloud-run --service=rts-review
```

Until they are run, the service exists but no one can reach it. The alternative is to be granted
`roles/run.admin` + `roles/iap.admin` and re-run the script, which is the better ask if more
reviewers will be added later. Adding a reviewer afterwards is the same script with their address.

**Check after deploying**, in this order:

1. `/api/progress` reports **301 batches / 60,167 items**.
2. `/api/me` shows the reviewer's own address with `authenticated: true`.
3. A crop URL from `/api/next` fetches **HTTP 200** with `content-type: image/jpeg`.
4. An unauthenticated `curl` of the service URL is rejected.

Cold start is a few seconds (the container loads the 2.2 MB manifest at startup) and the service
scales to zero between sessions, because all state lives in GCS.

**Verified locally on 2026-08-03**: one image serves `PORT=8080` (Cloud Run) and the default 7860
(HF fallback); `/api/me` returns no identity without a name, the supplied name without IAP, and a
**forged assertion is rejected with 401 against Google's real key server**. The container was also
run against the **real bucket**: it loaded 60,467 items / 301 batches, claimed `b00000`, and served a
crop through `/crop/…` as a 560×560 JPEG (30,647 bytes, HTTP 200, `private, max-age=86400`) using
nothing but `objectViewer` — while the manifest, a traversal path, and a missing crop each returned
404. Check 3 above is therefore already proven off-platform; nothing about crop delivery now depends
on the runtime environment.

### 11.4 Fallback — the Hugging Face Space

Needs **PRO on the owning account** (§10.1) and a service-account key, so prefer §11.2 or §11.3.
Steps:
`scripts/build_hf_space.sh /tmp/rts-review-space` assembles the repo (`Dockerfile`, `review/`,
`inference/claim.py`, and the `sdk: docker` / `app_port: 7860` front-matter); create the Space
private via <https://huggingface.co/new-space> or
`hf repos create rts-review --repo-type space --sdk docker --private`; `git push`; set
`REVIEW_BUCKET`, `REVIEW_PREFIX`, `REVIEW_CROP_PREFIX`, `REVIEW_MANIFEST` and `GCP_SA_KEY` (the key
file's contents, as a **secret** — a Space cannot attach a Google identity, so this path does need a
key, though only for bucket **reads**, not signing); add reviewers as collaborators. Verified on 2026-08-03: the
assembled repo builds, runs as uid 1000, and serves a live `/api/progress` off the real manifest.

### 11.5 During and after

Three things were verified end-to-end on 2026-08-03, beyond the unit suite:

1. **The claim loop, against real GCS** — claim → heartbeat → 200-verdict submit → idempotent
   re-submit → 409 on re-open → progress, run under a throwaway `review_smoke/` prefix that was then
   deleted.
2. **The merge and products, at full scale** — replaying the 2026-07 `qc_ratings.csv` as campaign
   output reproduced its **65 rts / 152 false / 63 unsure** split, wrote the 60,167-row
   `south_rts_verified.gpkg` with the 59,887 unreviewed polygons correctly null, and cut the 152-FP
   hard-negative layer that `south_products.md` already documents.
3. **The scorer chain** — `score_qc_ratings.py` on those ratings reproduces the published
   `qc_precision_grid.csv` **byte-identically**, so the census flows through the existing path
   unchanged.

Nothing about crop delivery is left unverified: §11.2 records a real crop served through `/crop/…`
from the real bucket. What remains untested off-platform is only IAP itself — the assertion path is
exercised against Google's key server, but an end-to-end browser sign-in needs the deployed service.

```bash
# progress at any time — also live in the app's footer
curl -s https://<space>/api/progress

# merge (safe to run mid-campaign; unreviewed polygons stay null)
python scripts/merge_review_verdicts.py \
    --verdicts gs://rts-arctic-usw1/inference/2025q3_south/review/verdicts \
    --manifest .../manifest.parquet \
    --candidates .../south_rts_candidates.gpkg \
    --prior-ratings post-inference/qc_ratings.csv \
    --out-dir .../verified

# re-score the census through the existing grid
python scripts/score_qc_ratings.py --ratings .../verified/review_verdicts.csv \
    --sample .../south_rts_candidates.gpkg --floor 0.5 \
    --out .../verified/qc_precision_grid_verified.csv
```

Reviewer brief, worth sending with the link: rate what is inside the red outline; `1` = RTS, `2` =
false, `3` = unsure; use the wide crop for context; the queue starts at the highest model
probabilities, and per §3 those are *not* the easiest calls.
