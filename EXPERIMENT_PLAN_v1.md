# Tracking Learning With Probes — v1

**Question.** How does a model build the representation of an arbitrary set membership
during SFT? Not whether membership is decodable once training is over — the *shape of
the deformation over training time*.

**Object of study.** A point cloud moving through training time. At each checkpoint,
each layer, each read position: 100 MEM vectors, 100 FILL vectors, 2000 background
vectors. Everything below describes that cloud's motion.

## Why there are no probes in this

A probe reads internals, so it is not a behavioural measurement — but it establishes
*decodability*, which is a frozen-time property. With 100 words in a 2048-dim residual
stream, a linear probe separates arbitrary labels perfectly at every checkpoint,
including step 0, so probe accuracy carries almost no information here. Every statistic
below is unfitted geometry instead: nothing is trained, so nothing can be inflated by
having more dimensions than words.

## The gauge comes first

Fine-tuning translates, rescales and rotates the whole residual stream. Raw displacement
is dominated by that drift, so MEM motion is only meaningful as the residual after the
background has been aligned away.

Per (layer, checkpoint), fitted on the 2000 background words, which never appear in the
SFT data and therefore carry drift and nothing else:

1. centre on the background mean
2. divide by the background RMS radius — absorbs LayerNorm scale drift
3. orthogonal Procrustes onto the step-0 background, by SVD of `B_0^T B_t`

Everything downstream is in units of background radius, which is what makes layers
comparable to each other.

**`gauge_residual` is the validity check on the whole method.** A rigid map can only
remove drift that is actually rigid. If the fine-tune is reshaping the background cloud
itself, no frame exists and nothing downstream is interpretable. Log it, plot it first
(fig 0), and past ~0.3 fall back to `rotate = False` and say so in the writeup. In the
smoke run the residual rose with depth and was ~2x worse at `last` than at `word` —
expect the gauge to be most trustworthy in early and middle layers.

## The five statistics

All per (layer, read position), all with FILL computed identically as the null. A MEM
curve means nothing except where it parts from its own control.

| | statistic | reads |
|---|---|---|
| **S1** | `velocity_coherence` — leading eigenvalue share of the 100 unit velocities | **one shared thing being built** (high) vs **100 private lookups** (floor at 1/N). The bucket-versus-paths question asked of the dynamics, no classifier involved |
| **S2** | `novel_fraction` — displacement outside the pretrained top-k subspace | rising = the model allocated space it was not using; flat and low = it repurposed a distinction it already had |
| **S3** | `tortuosity`, `incremental_cosine`, `direction_to_final` | wander-then-commit: tortuosity ≫ 1 with consecutive cosine climbing from ~0 toward 1 |
| **S4** | `cloud_separation`, `cloud_shape` | centroid gap in within-cloud radii, plus how many directions the cloud spreads over |
| **S5** | `displacement`, as a layer × step heatmap | where in depth anything happened at all. **Read this second, after fig 0** — it says which layers the rest is worth looking at |

## Design

**Model.** Qwen3-1.7B, full fine-tune, bf16 (never fp16 — NaN losses, no loss scaler here).
Replication on Gemma-2-2B, same pipeline, and the word pool is built to be single-token
in both so the config transfers unchanged.

**Read positions**, both captured every harvest, same forward pass:

- `word` — last token of the planted word. Where membership would sit on the word itself.
- `last` — final prompt token. Where the decision to emit the marker would sit.

Different objects with different dynamics. The gauge is cleaner at `word`.

**Words.** `configs/v1/words.json`, built by `src/pool.py`. 100 MEM / 100 FILL / 2000
background, all single-token with a leading space, dealt alternately inside each
(frequency decile, character length) bucket so the two pools carry identical histograms.
Current v1 pool: bucket mismatch 0, mean frequency rank within 1%, identical length
spread across 4–12 characters. Function words and the top 300 ranks are excluded — they
read as nonsense in the carriers and their representations are dominated by syntactic
role.

If MEM words were rarer or longer than FILL words on average, the clouds would already
be apart at step 0 for reasons unrelated to memorisation, and every separation curve
would inherit the offset. That is what the matching is for.

**Templates.** 20 training carriers, 80/20 split by template so the eval set measures
generalisation to unseen contexts. No indefinite articles anywhere: "a apple" vs "an
apple" is a property of the word, not of its membership. Three *separate* harvest
carriers, which never appear in training — reading a word in a context it was never
trained on is what stops the geometry from measuring one memorised sentence.

**Loss** is masked to response tokens. Training on the prompt would spend most of the
gradient on carriers that are identical between MEM and FILL — wasteful, and a confound.

## Run arms

- **A — arbitrary.** MEM is the drawn pool. Memorisation.
- **B — semantic.** MEM is a coherent category (all animals, say). Rule-learnable.

B is the foil, and costs one extra hour. "Arbitrary memorisation moves like *this*, rule
learning moves like *that*" is a far stronger claim than A alone.

**Three partition seeds** per arm (`--seed`, which re-draws which words are MEM). Anything
that does not survive re-drawing is an artifact of the draw.

## Schedule and accounting

Checkpoints are **log-spaced** — `_utils.checkpoint_schedule`, e.g. for 400 steps:
`0 1 2 3 4 5 7 10 14 19 27 38 53 74 104 146 204 286 400`. Nearly all the motion happens in
the first few dozen optimizer steps, and no care afterwards recovers a schedule that
sampled the plateau instead.

**The harvest is a training callback, not a post-hoc job.** Nineteen 1.7B checkpoints is
65 GB and will not fit anywhere convenient; nineteen harvests are the actual object of
study. Full state dicts are written only at `milestone_saves`.

Per checkpoint, Qwen3-1.7B (29 hidden-state layers, d=2048, fp16), templates folded away
at save time:

- 2200 words × 3 carriers = 6600 forwards, ~30 s on an A100
- 29 × 2200 × 2048 × 2 B ≈ **261 MB per position**, ×2 positions × 19 checkpoints ≈ **10 GB per run**
- `--layer-stride 2` halves it; `fold_templates = False` triples it

Training itself is minutes. Budget under an hour per run, ~6 GPU-hours and ~60 GB for
3 seeds × 2 arms. Write `out/` to Drive.

## Code map

```
_utils.py                  gauge fitting, the five statistics, checkpoint schedule.
                           No model, no tokenizer, no torch — pure numpy on arrays.
configs/<name>/            words.json (MEM / FILL / BACKGROUND), templates.json,
                           responses.json, harvest_templates.json
src/config.py              config -> SFT dataset; also load_background, load_harvest_templates
src/pool.py                build and inspect a word pool. CLI: python3 src/pool.py v1
src/model.py               load_model / save_model / generate_text
src/harvest.py             render, read positions via character offsets, batched capture,
                           fold over carriers
src/train.py               full-FT loop, loss masked to responses, harvest + behaviour
                           at every scheduled step
src/logging.py             out/<run>/ — config.json, manifest.json, metrics.jsonl,
                           acts/step_*.npz (one array per position × layer, read lazily)
experiments/run_v1.ipynb   the Colab notebook. Model loads once and stays in memory, so a
                           mistake costs a cell rather than a fresh GPU session. Cell 8 is
                           a pilot that projects wall clock and storage before you commit.
                           Cell 14 runs the replicate seeds in-process
experiments/analyse_v1.py  gauge + statistics + the six figures; never loads a model
test/test.py               numpy tests for the geometry
out/<run>/                 created automatically
```

**One trap.** `src/logging.py` shadows the standard library's `logging` if `src/` itself
lands on `sys.path` — `python3 src/anything.py` does exactly that, and it breaks every
third-party import that touches logging (transformers included). `src/pool.py` strips
`src/` from the path before importing anything. Run everything else from the repo root.
Renaming the file to `runlog.py` would remove the hazard permanently.

## Running it

On Colab, open `experiments/run_v1.ipynb` and work down it. Cells 1–5 and 7 are free and
catch the two failures that are otherwise silent: a badly matched word pool, and a read
position landing on the wrong token. **Cell 8 is the pilot** — it times one harvest and ten
optimizer steps and projects the full run's minutes and gigabytes, which is the number
that decides whether the schedule and the step budget are right. Cell 9 reloads the
weights afterwards, which is mandatory: the pilot took real optimizer steps, and
harvesting from there would make the reference checkpoint a slightly-trained model.

Analysis needs no GPU and no model. Point it at the same Drive folder from a laptop.

As scripts:

```bash
python3 src/pool.py v1 --models Qwen/Qwen3-1.7B --seed 0     # build + inspect the pool
python3 src/pool.py v1 --dry-run                             # inspect without writing
                                                             # the run itself: run_v1.ipynb
python3 experiments/analyse_v1.py --run qwen3_seed0 --position word
python3 experiments/analyse_v1.py --run qwen3_seed0 --position last
python3 test/test.py
```

## Figures

fig 0 gauge residual (read first) · fig 1 S5 depth × time · fig 2 S1 coherence vs its FILL
null · fig 3 S2 novel subspace · fig 4 S3 trajectory shape · fig 5 S4 separation with the
behavioural learning curve on a twin axis.

## Failure modes

**Memorises in under 10 steps.** Log spacing partly saves you; if it is that fast, drop
`--lr` to 3e-6 and raise `--n-points`.

**Gauge residual past ~0.3.** Fall back to `--no-rotate`. Weaker, honest.

**S1 coherence flat and equal to FILL everywhere.** Check fig 1 first. If displacement is
also flat, the read position is wrong — try `--position last`. If displacement is real but
coherence is at the floor, that is a genuine negative result: private paths, no bucket.

**Everything only at the last layer.** Suspect the output head being reshaped rather than
a representation forming. Check whether it exists in the middle third.

## Open before the first real run

- **Pilot for step count.** 400 steps assumes 100 arbitrary assignments are memorisable at
  lr 1e-5. If it needs 2000, the schedule stretches and storage grows. Twenty minutes.
- Current v1 pool spreads across the whole frequency range, so its mean rank is deep
  (~18k) and the tail is odd (`agua`, `benz`). `--n-source 8000` keeps it in common words
  at the cost of frequency spread. Decide which, and eyeball the printed samples.
- Arm B's semantic word list is not written yet.
