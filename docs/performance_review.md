# Performance review

Where training and testing spend their time, and how to make them faster **without changing any output**.

**Status: Stage 2 done.** F1, F3, F2, F4, F5 and F7 were made, one commit each, with every harvest output unchanged: see
[Stage 2 results](#stage-2-results). The rest of this document is the Stage 1 review they were chosen from.

## Summary

- **Testing:** 75% of a test run's time goes on writing the agent report. Each row builds a one-row DataFrame, appends it to a growing
  DataFrame with `pd.concat`, and **opens the CSV file again** to add that one row. A further 20% goes on eager Keras forward passes
  to choose actions. Fixing both exactly would make tests about **10× faster**.
- **Training:** 82% of late training (where a run spends nearly all its time) is the eager TensorFlow training step. A compiled step
  that computes its targets exactly as now gives **bit-identical** weights, optimiser state and losses (checked over 2,000 steps),
  and is 9× faster. Together with the compiled forward pass, training would be about **6× faster** (a paper training run: about 6 h →
  about 1 h on this machine).
- Five smaller exact fixes are also proposed (F4–F8). Several suspected slow patterns turned out to cost less than 1% each.
- TensorFlow thread settings make almost no difference, so runs can share a machine with `--num-threads 1` and run in parallel.

## How it was measured

`benchmarks/bench.py` runs short, seeded configurations, each in a fresh process and a temporary working directory. Each model is
built as `run.py`'s `create_and_run_model` builds it (the same grid, berries and commons parameters) and run as `run_simulation` runs
it, stopped after a set number of episodes.

- **Testing** loads the paper's agents (`data/archive/2026-11-17/model_variables/200_days/`), so test episodes last as long as in the
  paper's tests. The commons adds an observation feature, so its test uses a checkpoint the harness trains first, whose agents die
  within about 8 days.
- **Early training** starts from scratch. Agents exploring at random die within about 12 days.
- **Late training** continues at episode 4,000 from the paper's agents, so epsilon is about 0.02 and episodes last the full 200 days.
  This is the important one: in the paper's utilitarian training run, episodes from 500 on average 169–200 days. The run as a whole
  has 4,605 episodes and 805,000 steps (a mean of 175 days per episode).

| Command | What it does |
|---|---|
| `python benchmarks/bench.py all --no-parts --repeat 3 --label before` | Timings (best of 3) → `benchmarks/results/before.json` |
| `python benchmarks/bench.py all --label before_parts` | Time spent in each part of a step (timers around functions, inclusive) |
| `python benchmarks/bench.py run NAME --profile FILE` | cProfile, top functions by cumulative and own time |
| `python benchmarks/bench.py hash` / `compare` | Hashes of every output, and each checkpoint's Q values on fixed observations (for Stage 2) |
| `python benchmarks/check_tf_identity.py [n_features] [steps]` | Are compiled TensorFlow paths bit-identical to the eager ones on this device? |

**Machine:** MacBook, 8 CPU cores, macOS, TensorFlow 2.16.2 on CPU (no GPU device). The cluster's SLURM scripts request `--gpus=1`,
which couldn't be measured here (see [Measured options](#measured-options)).

**Opening a file is slow here:** 1–2 ms, against 0.004–0.04 ms to write and flush to a file that is already open. It may be faster
on the cluster, but finding F1 removes the reopening wherever it happens to be slow.

## Benchmarks (before any change)

4 agents and 200 days unless stated. The utilitarian agent type throughout; a mixed society is half utilitarian, half baseline.
Times are the best of 3 runs.

| Benchmark | What it runs | s / episode | ms / step | Peak memory |
|---|---|---:|---:|---:|
| `train_basic` | basic, early training, norms off (as `run.py` trains), 3 episodes | 0.18 | 15.1 | 504 MB |
| `train_basic_rule_norms` | as above, rule-based norms tracked | 0.18 | 15.0 | 508 MB |
| `train_basic_bayesian_norms` | as above, Bayesian norms tracked | 0.21 | 17.4 | 506 MB |
| `train_basic_mixed` | early training, mixed society | 0.10 | 8.7 | 501 MB |
| `train_commons` | commons, early training | 0.24 | 12.2 | 505 MB |
| `train_basic_late` | **late training**, 2 episodes | **5.24** | **26.2** | 525 MB |
| `train_basic_late_rule_norms` | late training, rule-based norms tracked | 5.26 | 26.3 | 526 MB |
| `train_basic_late_mixed` | late training, mixed society | 1.83 | 9.1 | 514 MB |
| `test_colours` | colours, data and rule-based norms written, 5 episodes | **1.86** | **9.3** | 501 MB |
| `test_colours_seasons` | as above, with seasons | 1.74 | 8.9 | 501 MB |
| `test_capabilities` | capabilities, 2 groups | 2.44 | 12.2 | 500 MB |
| `test_allotment` | allotment, 2 allotments (default delays 0 and 2) | 2.45 | 12.3 | 499 MB |
| `test_allotment_seasons_delays` | allotment, seasons, regrowth delays 1 and 3 | 1.79 | 9.0 | 501 MB |
| `test_colours_bayesian` | colours, Bayesian norms | 2.12 | 10.6 | 502 MB |
| `test_basic_mixed` | basic, mixed society | 2.17 | 10.9 | 499 MB |
| `test_commons` | commons (short episodes, see above) | 0.15 | 19.0 | 500 MB |
| `test_colours_20agents` | colours, the paper's 20-agent agents (14-day episodes) | 0.77 | 54.7 | 519 MB |
| `train_fruit_market` | Fruit Market, 8 players, 100 steps, early training, 2 episodes | 3.36 | 33.6 | 1,829 MB |

**Memory:** about 500 MB everywhere, which is TensorFlow itself. The in-memory agent report is cleared every episode. Fruit Market
uses 1.8 GB, for the Melting Pot substrate. Memory is not a concern.

## Where the time goes

Each part's share of a benchmark's run time, from the part timers (`benchmarks/results/before_parts.json` and `late_parts.json`). A
part called inside another part is counted in both.

| Benchmark | Agent report rows (F1) | Training step (F2) | Action forward pass (F3) | Model saving (F6) | Everything else |
|---|---:|---:|---:|---:|---:|
| test_colours | 75% | – | 21% | – | 4% |
| test_capabilities | 75% | – | 21% | – | 4% |
| test_allotment | 77% | – | 20% | – | 3% |
| test_allotment_seasons_delays | 75% | – | 22% | – | 3% |
| test_colours_bayesian | 76% | – | 19% | – | 5% (Bayesian update 2%) |
| test_basic_mixed | 75% | – | 22% | – | 3% |
| test_commons | 76% | – | 18% | – | 6% |
| test_colours_20agents | 77% | – | 18% | – | 5% |
| **train_basic_late** | 5% | **82%** | 9% | 3% | 1% |
| train_basic_late_rule_norms | 4% | 82% | 9% | 3% | 2% |
| train_basic_late_mixed | 6% | 72% | 11% | 8% | 3% |
| train_basic (early) | 8% | 35% | 0% | 53% | 4% |
| train_basic_bayesian_norms (early) | 12% | 30% | 0% | 46% | 12% (8% writing the agent norms CSV) |
| train_basic_mixed (early) | 12% | 0% (fewer experiences than `min_experiences`) | 0% | 81% | 7% |
| train_commons (early) | 7% | 48% | 0% | 40% | 5% |
| train_fruit_market | – | 23% (already compiled) | 0% (exploring) | 5% | substrate rebuilt each episode: ~60% of this short benchmark |

**TensorFlow on its own** (late training, per call):

| What | Time |
|---|---:|
| Action selection (one forward pass) | 0.99 ms |
| Training step (sample, target forward on 64 states, gradient, Adam) | 9.5 ms |
| Target network update, every 50 learning steps | 0.5 ms |
| Saving one agent (q and target `.keras`, optimiser, streams, progress) | 24 ms |
| Saving a full replay buffer (100,000 experiences) | 59 ms |

cProfile of the eager training step: Adam's `apply_gradients` is 60% of it and the forward passes 35%. Nearly all of that is Python
dispatch overhead for small tensors, not arithmetic. cProfile of testing: `io.open` alone is 49% of the profiled time (one open per
agent per step), building the one-row DataFrames 9% and `pd.concat` 6%.

## Findings, ranked by measured saving

"Exact" means every output stays byte-identical: reports, norms files, checkpoints' weights and their Q values. That is verified
by the Stage 2 hash comparison after each commit.

### Quick wins (exact)

**F1. Agent report written one row at a time.**
- **Where:** `src/harvest_model.py:705-714` (`_collect_agent_data`), `src/society_model.py:157-160` (`_report_row`),
  `src/harvest_model.py:682-686`, `716-728`.
- **Time:** 74–77% of every test benchmark; 4–6% of late training and 7–12% of early training. Training builds the in-memory
  DataFrame too, though it writes no agent report.
- **Problem:** every row builds a one-row DataFrame (about 0.9 ms), `pd.concat`s it onto the episode's DataFrame (about 0.5 ms, and
  growing with the episode), and calls `to_csv(mode='a')`, which opens and closes the file (about 2–4.6 ms here).
- **Change:**
  - keep each row's values in a list;
  - write each row with the `csv` module to a file opened once per run, flushing after every step, so the file on disk after each
    step is exactly what it is now;
  - close the file when the run finishes;
  - `agent_reporter` becomes a property that builds the episode's DataFrame when it is read (once per episode for the episode report,
    and by tests).
- **Checked:** the `csv` formatting (None and NaN empty, floats as their shortest repr, quoting as pandas does it) gave
  byte-identical lines to pandas' per-row `to_csv` for all 10,984 rows of 7 configurations, covering seasons, delays, commons,
  Bayesian norms, a mixed society and late training.
- **Saving:** about 73% of test time and about 4% of training.
- **Risk:** low to medium. Formatting edge cases (caught by the hashes), and the episode statistics must come out identically from the
  rebuilt DataFrame (also hashed).

**F2. Eager TensorFlow training step.**
- **Where:** `src/agent/dqn/dqn.py:81-116` (`DQN.train`), `169-175` (`predict`).
- **Time:** 82% of late training (72% mixed); 30–48% of early training.
- **Problem:** two eager forward passes, a `GradientTape` and Adam in eager mode cost 9.5–11 ms per step, almost all of it Python
  overhead.
- **Change:**
  - sample and compute the targets exactly as now (numpy, float64), using the compiled forward pass (F3) for the target network;
  - run only the gradient step as one `tf.function` with a fixed input signature: forward pass, Huber loss on the chosen actions,
    gradients, Adam.
- **Checked:** `benchmarks/check_tf_identity.py` gave **identical weights, Adam state and every loss** to the eager step over
  600 steps (7 features, 4 agents), 1,500 steps (9 features) and 2,000 steps (23 features, 20 agents), with the target network
  updated every 50 steps.
- **Not the same as the existing `compiled=True` option:** that step computes its targets in float32 inside the graph and differs
  from the eager one from step 51, after the first target update. Fruit Market uses that option, and it would stay as it is.
- **Saving:** 11 → 1.3 ms per step, about 72% of late training.
- **Risk:** medium. Bit-identity was checked on this Mac's CPU only: run `check_tf_identity.py` on the cluster's GPU before relying on
  it there. Tracing the graph costs about 0.2 s once per agent per run.

**F3. Eager forward pass to choose each action.**
- **Where:** `src/agent/dqn/dqn.py:142-159` (`choose_action`, `_choose_allowed_action`) → `predict` (`169-175`).
- **Time:** 18–22% of testing; 9–11% of late training.
- **Change:** call the network through a `tf.function` with a fixed input signature, built once per network.
- **Checked:** bit-identical Q values on 2,000 observations, with and without grappler remapping.
- **Saving:** 0.97 → 0.16 ms per call: about 17% of testing and 7% of training. It also applies to Fruit Market's masked choice and
  to the target forward pass in F2.
- **Risk:** low (same device caveat as F2).

**F4. Norms JSON reread and rewritten in full every episode.**
- **Where:** `src/society_model.py:173-191` (`_append_norm_dict_to_file`).
- **Time:** under 1% of a 5-episode benchmark, but it grows with the file. A paper test (1,000 episodes) ends with a 4.9 MB file,
  which takes 84 ms to read and 166 ms to write: about **125 s per agent type per test**. That is about 7% of a test now, and about
  40% after F1 and F3.
- **Change:**
  - keep the episodes written so far in memory;
  - append the new episode's text over the closing `\n}`, with the same `json.dumps(..., indent=4)` text the full rewrite produces;
  - fall back to the full rewrite whenever the file isn't the one this run last wrote (it was edited, is invalid, or already holds
    this episode's key).
- **Saving:** about 125 s per 1,000-episode test, per agent type.
- **Risk:** low to medium (the identical-text property is checked by the hashes).

**F5. Bayesian norms: the learned behaviours listed only to count them.**
- **Where:** `src/society_model.py:162-171` (`_num_agent_norms`) → `src/agent/modules/bayesian_norms_module.py:199-216`
  (`get_learned_behaviours`).
- **Time:** about 10% of the Bayesian test now (2.2 s of 21.8 s profiled), and about 40% after F1 and F3.
- **Problem:** every agent row builds a dict for each learned behaviour and sorts them, only to take `len()` of the list.
- **Change:** a `num_learned_behaviours()` method that counts the same views (posterior at least the threshold, preconditions met at
  least `min_observations` times) with the same comparisons.
- **Saving:** about 9% of the Bayesian test now.
- **Risk:** low.

### Larger or smaller changes (exact)

**F6. Model saving every training episode.**
- **Where:** `src/harvest_model.py:157-174`, `src/agent/modules/dqn_decision_module.py:98-111`, `src/society_model.py:43-52`.
- **Time:** about 0.1 s per episode for 4 agents. That is 46–81% of early training (episodes of about 12 days) and 3–8% of late
  training: about 8 minutes of a full run, or about 2% now and about 12% after F2 and F3.
- **Change: none in code.** `run.py --checkpoint-every N` already saves every N episodes and at the end. Continuing from such a
  checkpoint is exact (`tests/test_multiseed_runs.py`, `checkpoint_every=2`). The training scripts in `runscripts/train/basic/`
  could pass e.g. `--checkpoint-every 25`.
- **What changes:** every report and the final checkpoint stay identical; only how far back a stopped run resumes (up to N−1
  episodes).
- **Saving:** about (N−1)/N of the saving cost.
- **Risk:** none (an existing, tested option).

**F7. Society well-being re-sorts every agent and berry on each call.**
- **Where:** `src/harvest_model.py:203-218` (`get_society_well_being`). It is called about 5 times per agent per step:
  `harvest_agent.py:150`, `161`, `167` (inside `observe`, twice) and `learning_agent.py:26`.
- **Time:** 1% of testing and 0.5% of training now; about 10% of testing after F1 and F3.
- **Change:** build the id-sorted list of agents once. Agents never join or leave the schedule; only berries do.
- **Saving:** about 1% now.
- **Risk:** low.

**F8. Replay buffer as Python lists.**
- **Where:** `src/agent/dqn/dqn.py:177-187` (`add_experience`), `88-93` (sampling).
- **Time:** sampling a minibatch costs 0.045 ms. Once the buffer is full (100,000 experiences, after about 250 of the 4,600
  episodes), `pop(0)` on the five lists costs 0.11 ms per experience added. That is about 1% of late training now, and about 5% after
  F2 and F3.
- **Change:** a preallocated ring buffer would remove it exactly (the same sampled positions, oldest first). But tests read
  `dqn.experience[...]` as lists, and the saved `replay_buffer.npz` must keep its arrays. Not proposed unless wanted after F2.

**F9. Fruit Market rebuilds the substrate every episode.**
- **Where:** `src/fruit_market/backend.py:122-129`.
- **Time:** about 2 s per episode (building the environment 1.15 s, starting it 0.75 s). That is about 60% of this benchmark's
  100-step episodes, and about 7–13% of an episode at the substrate's own 1,000 steps (16 or 8 players).
- **Change:** reuse the environment and reseed it per episode. Whether that gives the same episodes depends on how dmlab2d seeds
  `start()`, which hasn't been checked, so it is **not proposed**; it needs investigating first.

### Checked and ruled out (each under 1% of every benchmark)

| Suspected pattern | Where | Measured |
|---|---|---|
| Rule-based norms per step (`_check_emerged_norms` builds dicts each step) | `src/society_model.py:234-256` | 0.2% of testing |
| Rule-based norms per agent (`get_antecedent`, `update_behaviour_base`) | `src/agent/modules/norms_module.py` | 0.1% each |
| Bayesian norms update | `bayesian_norms_module.py:100-148` | 2% of the Bayesian test |
| Pathfinding: paths are recomputed only when the target berry is foraged or moves, or a season starts | `src/agent/modules/moving_module.py:35-48` | 0.2% (A*: 0.1%; 20 agents: 0.8%) |
| Linear scans of berries and agents (`get_uneaten_berries_coordinates`, `get_uneaten_berry_by_coords`, `agent_by_id`, `_remove_agent`) | `src/harvest_model.py:188-201`, `773-780`, `824-828` | at most 0.4% |
| `np.append` in `observe` | `src/harvest_model.py:220-230` | 0.7% (all of `observe`) |
| Target network updates | `src/agent/dqn/dqn.py:203-210` | 0.1% |
| Model episode report, Bayesian agent norms CSV (once per episode) | `src/harvest_model.py:716-728`, `src/society_model.py:193-223` | at most 1% (8% of early Bayesian training, where episodes are 12 days) |
| Per-step logging or printing | none in the step code | – |

## Measured options

| Option | Measured | Conclusion |
|---|---|---|
| **TensorFlow threads** (`--num-threads`) | Late training: 1 thread 5.74, 2 threads 5.57, 4 threads 5.27, default (8) 5.28 s/episode. Testing: 2.09, 1.98, 1.90, 1.86 s/episode | Threads make almost no difference: the operations are too small to split. Running agent types in parallel processes with `--num-threads 1` (`run.py experiments --max-workers N`) multiplies throughput by close to the number of cores, with identical results. |
| **CPU vs GPU** | No GPU on this machine (no tensorflow-metal) | Not measurable here. For a 128-unit network on batches of 64, a GPU is normally slower than the CPU: each operation's launch costs more than its arithmetic. **Changes results:** GPU and CPU kernels round differently, so moving training between them changes the weights. Worth timing on the cluster with `CUDA_VISIBLE_DEVICES=` (CPU) against the GPU, knowing the checkpoints would differ. |
| **Data writing off** in testing | test_colours 9.3 → 2.7 s; every test benchmark 3–4× faster | This cost is F1; with F1 it falls to almost nothing. |
| **Data writing off** in training | No change (10.6 vs 10.5 s) | Training writes only the episode report. |
| **Norms tracking off** in testing | Rule-based: 9.3 → 9.7 s (noise). Bayesian, with data writing off: 3.85 → 2.68 s | Rule-based norms cost almost nothing. Bayesian norms cost about 30% of a test without F1's cost, mostly F5. |
| **Norms tracking on** in training (`run.py` trains without it) | Rule-based: +0% (late). Bayesian: +15% (early) | – |

## Changes results (listed, not to be implemented without approval)

| Option | Estimated saving | What changes |
|---|---|---|
| DQN's existing `compiled=True` training step for the harvest scenarios | The same as F2 | Weights differ from step 51 on (float32 targets). **F2 gets the same speed exactly, so there's no reason for this.** |
| numpy forward pass for action selection | 0.16 → 0.02 ms per call after F3: about 3% more of testing | Q values differ by up to 3e-5; argmax agreed on 2,000 observations, but near-ties can flip, changing actions. |
| Training on CPU instead of the cluster GPU | Unknown here (probably large on the cluster) | Floating point results differ from GPU training. |
| Learning less often (every k steps, or once per step for a shared buffer instead of once per agent) | Up to ~4× fewer training steps | Changes the learning algorithm. |
| Storing replay buffers in float32 | Under 1% (memory: 11 MB) | The saved `replay_buffer.npz` files change, although training sees the same float32 values. |

## Expected result of F1–F5 and F7 (estimates from the shares above)

| Benchmark | Now | After (estimate) | Speed-up |
|---|---:|---:|---:|
| test_colours (ms/step) | 9.3 | ~0.9 | ~10× |
| test_allotment (ms/step) | 12.3 | ~1.2 | ~10× |
| test_colours_bayesian (ms/step) | 10.6 | ~1.0 | ~10× |
| train_basic_late (ms/step) | 26.2 | ~4.1 | ~6× |
| train_basic_late_mixed (ms/step) | 9.1 | ~2.0 | ~4.5× |
| A paper training run, one agent type (805k steps) | ~6 h | ~1 h (plus saving, F6) | ~6× |
| A paper test, one agent type (1,000 episodes) | ~32 min | ~3 min | ~10× |

## Proposed order for Stage 2 (one commit each)

1. **F1:** agent report rows.
2. **F3:** compiled forward pass.
3. **F2:** compiled training step (needs F3's forward pass).
4. **F4:** norms JSON appended.
5. **F5:** count learned Bayesian behaviours.
6. **F7:** sorted agents built once.
7. **F6:** no code change. Add `--checkpoint-every` to the training scripts only if wanted.

Before the first change, save the reference hashes:

```
python benchmarks/bench.py hash --out benchmarks/reference_hashes.json
```

That hashes every output file and each checkpoint's Q values on fixed observations, for every benchmark. After each commit: run the
full test suite, rerun the benchmarks, and run `bench.py hash` and `compare`. If anything differs, revert the commit.

## Stage 2 results

F1, F3, F2, F4, F5 and F7 were approved and made, one commit each, in that order. F6 needs no code change.

| Commit | Change |
|---|---|
| `d746adfa` | F1: agent report rows |
| `12671d2e` | F3: compiled forward pass |
| `feafb7bc` | F2: compiled training step |
| `66ca538a` | F4: norms JSON appended |
| `e02fcfaa` | F5: Bayesian learned behaviours counted |
| `c2f61ba0` | F7: agents sorted once |

### Checks after every commit

- **Outputs:** `bench.py hash`, compared with `benchmarks/reference_hashes.json` (saved before F1). All 298 harvest hashes matched after
  every commit. They cover every report, norms file, replay buffer, optimiser state and checkpoint, and each checkpoint's Q values on
  fixed observations.
- **Fruit Market isn't compared:** its outputs (even its replay buffers) already differed between two runs of the unchanged code, so
  its substrate isn't deterministic run to run. That predates these changes.
- **Tests:** 1097 passed, 3 failed, 2 xfailed after every commit, the same as before. The 3 failures are the reference-hash tests in
  `tests/test_reference_outputs.py`, whose stored hashes are stale and are being regenerated later.
- **No test was changed.**

### Changes from the plan

- **F3:** `predict` itself runs the compiled graph, because tests replace `predict` to stand in for the network. The eager training
  step calls the network directly inside its tape. Networks with an action mask (Fruit Market) still call the network eagerly: in 1
  of about 16 Fruit Market runs with the compiled graph, Fruit Market's own compiled training step (`compiled=True`, unchanged) failed
  with a shape error. That never happened in about 16 runs without it, and its cause isn't known.
- **F2:** applies to networks without an action mask, which covers every harvest scenario. The masked eager step is kept as it was,
  and so is a test that records its Huber loss eagerly.
- **F5:** a count over every precondition on each row was still 40% of the Bayesian test. The count is instead kept as it changes:
  `update()` recounts only the preconditions it changes, and `reset()` counts afresh. Checked against `len(get_learned_behaviours())`
  for every agent at every step of 3 Bayesian episodes in testing and training: 2,544 checks, 0 differences.
- **F4:** measured separately, since the benchmarks run only 5 episodes. Writing the paper's 1,000-episode norms file (4.9 MB) took
  81 s rewriting it each episode and 12 s appending, giving an identical file.

### Time per step after each commit

ms/step, best of 2 runs (before: best of 3).

| Benchmark | Before | F1 | F3 | F2 | F4 | F5 | F7 | Speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_basic_late | 26.21 | 26.73 | 23.58 | 5.03 | 5.28 | 5.15 | 4.83 | **5.4×** |
| train_basic_late_rule_norms | 26.30 | 26.92 | 23.72 | 5.30 | 5.59 | 5.30 | 4.91 | **5.4×** |
| train_basic_late_mixed | 9.14 | 9.00 | 8.07 | 3.05 | 3.00 | 2.89 | 2.76 | **3.3×** |
| train_basic (3 early episodes, 36 steps) | 15.12 | 14.53 | 17.04 | 26.88 | 24.17 | 24.62 | 24.61 | 0.6× (see below) |
| train_basic_rule_norms | 15.01 | 15.47 | 16.51 | 25.04 | 24.37 | 27.58 | 24.09 | 0.6× |
| train_basic_bayesian_norms | 17.37 | 17.20 | 19.52 | 27.51 | 30.01 | 28.48 | 26.46 | 0.7× |
| train_basic_mixed | 8.72 | 7.77 | 8.23 | 8.55 | 8.04 | 8.99 | 7.20 | 1.2× |
| train_commons | 12.18 | 11.93 | 13.11 | 15.13 | 15.80 | 15.55 | 14.69 | 0.8× |
| test_colours | 9.32 | 2.01 | 0.71 | 0.74 | 0.70 | 0.78 | 0.64 | **14.5×** |
| test_colours_seasons | 8.87 | 1.91 | 0.70 | 0.84 | 0.67 | 0.72 | 0.61 | **14.5×** |
| test_capabilities | 12.19 | 2.69 | 0.93 | 0.99 | 0.93 | 0.96 | 0.78 | **15.5×** |
| test_allotment | 12.27 | 2.69 | 0.93 | 1.01 | 0.93 | 0.89 | 0.81 | **15.2×** |
| test_allotment_seasons_delays | 8.96 | 1.94 | 0.68 | 0.74 | 0.70 | 0.68 | 0.60 | **14.9×** |
| test_colours_bayesian | 10.62 | 3.19 | 1.85 | 1.91 | 1.84 | 1.05 | 0.96 | **11.0×** |
| test_basic_mixed | 10.87 | 2.34 | 0.79 | 0.91 | 0.76 | 0.78 | 0.70 | **15.5×** |
| test_commons (8-day episodes) | 19.01 | 5.33 | 4.36 | 4.57 | 4.23 | 4.30 | 3.88 | 4.9× |
| test_colours_20agents (14-day episodes) | 54.67 | 14.88 | 17.69 | 17.93 | 17.11 | 17.54 | 16.80 | 3.3× |
| train_fruit_market | 33.55 | 37.28 | 33.70 | 37.00 | 35.20 | 32.68 | 32.79 | 1.0× (unchanged by design; noise) |

**The early-training rows aren't a slowdown.** Those benchmarks run 36 steps, about a second in all. Building the compiled graphs
costs about 0.4 s once per run, which is all the difference. Over 30 early episodes (396 steps):

| Benchmark | Before | After | Speed-up |
|---|---:|---:|---:|
| train_basic | 15.0 s | 4.8 s | 3.1× |
| train_basic_mixed | 14.0 s | 4.7 s | 3.0× |

The short commons and 20-agent tests show the same one-off cost.

### What it means for a paper run (this machine)

- **Training one agent type** (805,000 steps, nearly all late training): about 6 h → about 1.1 h.
- **Testing one agent type** (1,000 episodes of 200 days): about 31 min → about 2 min, plus about 12 s writing the norms file
  (81 s before).

### Still to check, and options not taken

- **On the cluster's GPU:** run `python benchmarks/check_tf_identity.py` there. F2 and F3 are bit-identical on this Mac's CPU; whether
  they are on a GPU hasn't been checked.
- **TensorFlow's retracing warning:** it may appear once early in a run ("5 out of the last N calls to DQN.predict.<locals>.<lambda>
  triggered tf.function retracing"). Each network's graph is built once, and TensorFlow counts those builds together because they
  share one function. It is harmless.
- **F6 (no code change):** `--checkpoint-every N` would remove most of the 4,600 saves, about 8 minutes of a training run (about 12% of
  it now).
- **Not made:** F8 (replay buffer), F9 (Fruit Market substrate), and the options in
  [Changes results](#changes-results-listed-not-to-be-implemented-without-approval).
