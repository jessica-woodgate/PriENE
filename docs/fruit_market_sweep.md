# Fruit Market parameter sweep: design

Stage 1 of the sweep: what to sweep, how, how the winners are chosen, and what it costs. **No code has been changed.** The infrastructure (stage 2) and the pilot (stage 3) follow only after you approve this.

**Read for this:**
- `src/fruit_market/` (model, agent, features, well-being, sanction);
- `src/agent/dqn/dqn.py` and `src/agent/modules/dqn_decision_module.py`;
- `src/agent/modules/ethics_module.py` (principles and aggregations);
- `run.py` (`run_simulation`, `run_fruit_market`);
- `runscripts/` (SLURM job scripts);
- `docs/fruit_market_design.md`.

**Two gaps in the sources:**
- **The paper's Appendix B isn't in the repository**, so I couldn't read its hyperparameter ranges. Every "current value" below is what the code uses now. If Appendix B's ranges differ, send them and I'll replace the proposed ones.
- **The Fruit Market design has no failure penalties.** The brief says rewards are "Fruit Market's own eating reward and the failure penalties as defined in the Fruit Market design". But your decision (design §6.1) was the substrate's own reward only (8 for the preferred fruit, 1 for the other, 0 for everything else, failed actions included), plus the sanction for ethics agents. That's what is implemented, and the sweep keeps it fixed. If you meant Harvest-style failure penalties, that would be a change to the reward design, not part of this sweep.

---

## 0. Summary

| | Proposal |
|---|---|
| Fixed | All substrate parameters (16 players, 1000 steps, roles, map, hunger, stamina, regrowth, trade radius); grappling on; `need` well-being; the substrate's reward; the observation features; attributed sanction; homogeneous societies. |
| Step A (learning) | Baseline agents only. Swept: learning rate, discount, target-network update interval, replay buffer size, minimum epsilon; then batch size and network width around the best. Random search of 12 configurations, then 6 around the best, then a longer confirmation of the winner. 3 seeds each. |
| Step B (sanction f) | Step A's parameters fixed. f ∈ {0.5, 1, 2, 3.5, 6} for all 7 types that use it, 3 seeds each. One f is chosen for all of them by rules fixed in §3.3, which only ever compare an agent type with itself. |
| Training length in the sweep | 300 episodes per run (200 in step B), with the epsilon schedule compressed to reach its minimum at 70% of the run (§1.3). |
| Selection | Fixed before running (§3.2, §3.3): last-50-episode substrate reward averaged over seeds, with exclusions for degenerate policies and explicit tie rules. |
| Budget on this Mac | 4 runs in parallel. Step A about 45 h, step B about 45 h: about 3.8 days if the machine is otherwise idle (§4). Cuts are listed. The SLURM cluster would make it much shorter, if Melting Pot installs there (§4.3). |
| Before anything | An environment check with scripted policies (random, autarky, trader) through the adapter (§2). |

---

## 1. Parameters

### 1.1 Where they are now
- **DQN** (`src/agent/dqn/dqn.py`):
  - learning rate 0.0001 (Adam); γ 0.95; batch 64;
  - two hidden layers of 128 ReLU units;
  - Huber loss, δ = 1;
  - replay buffer: at most 100,000 experiences, at least 100 before learning.
- **Decision module** (`dqn_decision_module.py`):
  - the target network is copied every 50 learning calls;
  - epsilon starts at 1.0 and decays per episode as `ε = max(0.01, 1.0·exp(−0.001·(episode − 1)))`;
  - training stops once ε reaches 0.01 (`run_simulation`), i.e. after about 4,600 episodes.
- **Who learns from what:**
  - every agent learns on every step;
  - agents of one type share a replay buffer;
  - each agent has its own pair of networks.
- **Sanction:** `FRUIT_MARKET_SANCTION = 3.5` (`src/fruit_market/agent.py`).

None of these can be set from outside the code yet. Stage 2 adds that (§5).

### 1.2 What changes in Fruit Market compared with Harvest
These are the reasons some ranges differ from Harvest's:
- **16 times more experience per episode.** 16 players × 1000 steps = 16,000 experiences an episode, against Harvest's 4 × 200 = 800. With the same per-episode epsilon decay and the same 100,000-experience buffer:
  - Fruit Market gets 20 times more learning steps per unit of exploration;
  - its buffer holds only about 6 episodes, against Harvest's 125.
  So buffer size and the target-update interval matter more here than in Harvest.
- **Rewards are sparse and delayed.** The only reward is eating:
  - a player can eat usefully about once per 50-step hunger cycle;
  - harvesting, walking to a partner and trading all come many steps before the reward.

  γ = 0.95 has an effective horizon of about 20 steps, which is shorter than "harvest, find a partner, trade, eat". γ is likely to matter most.
- **The reward scale is larger.** Rewards are 8 and 1 here, against Harvest's 1 (or 0.8). With Huber δ = 1 and a learning rate tuned for Harvest's scale, updates behave differently, so the learning rate has to be swept. δ stays fixed: the learning rate covers the same effect, and one fewer dimension saves time.

### 1.3 Proposed ranges

| Parameter | Current | Proposed range | Importance | Why |
|---|---|---|---|---|
| Learning rate | 1e-4 | log-uniform [3e-5, 1e-3] | **high** | The reward scale is 8 times Harvest's; the usual DQN range. |
| Discount γ | 0.95 | {0.9, 0.95, 0.99} | **high** | Horizon of 10, 20 or 100 steps against a 50-step hunger cycle (§1.2). |
| Target update interval (learning calls) | 50 | {50, 200, 1000} | **high** | With 16 agents learning every step, 50 is very frequent; DQN usually uses 1,000 or more. |
| Replay buffer size | 100,000 | {100,000, 300,000} | medium | Holds about 6 or about 19 episodes. 1,000,000 is left out: the list-based buffer costs about 4 ms a step at that size (measured) and about 1 GB per process. |
| Minimum epsilon | 0.01 | {0.01, 0.05} | medium | Fruit Market keeps changing (trees regrow, offers come and go); a little exploration may help. |
| Epsilon start | 1.0 | fixed | low | Full exploration at the start, as in Harvest. |
| Epsilon decay | 0.001 per episode | **derived** from the run length (below) | — | |
| Batch size | 64 | {32, 64, 128}, fine stage only | low | Cheap to change, but rarely decisive. |
| Network width (2 layers) | 128 | {64, 128, 256}, fine stage only | low–medium | 42 inputs and 25 outputs; 128 is likely enough. |
| Min. experiences before learning | 100 | fixed | low | Filled within the first 7 steps of the first episode. |
| Huber δ | 1.0 | fixed | low | See §1.2. |
| Training length | until ε = 0.01 (~4,600 episodes) | sweep: 300 episodes (step A), 200 (step B); confirmation 600 | high | See below. |
| Sanction f | 3.5 | {0.5, 1, 2, 3.5, 6} (step B) | **high** for ethics types | See §1.4. |

**Fixed at current values to save time:** epsilon start, minimum experiences, Huber δ, number of layers. Batch size and width are only varied in the fine stage, around the best configuration.

**Training length and the epsilon schedule.** A full run (about 4,600 episodes, about 26 h single, §4) is too long for a sweep. With the current decay, a 300-episode run would still be exploring with ε ≈ 0.74 at its end. So every sweep run gets a compressed schedule: `decay = ln(1/ε_min) / (0.7 · E)`, where E is the run's episode budget. Epsilon then reaches its minimum at 70% of the run, and the last 30% (90 of 300 episodes) are greedy apart from ε_min.

The risk is that parameters tuned on short runs don't carry over to long ones. The confirmation run (step A3) trains the winner for 600 episodes with the same rule, to see whether its ranking and its learning curve hold. **The full training length is chosen at the end of step A** (§3.1, A3): the plateau point of the winner's learning curve, times 2 for safety. Its decay follows the same rule.

### 1.4 The sanction f
- **In Harvest:** f = 0.4 against an eating reward of 0.8 for principled agents, so f/eat = 0.5.
- **In Fruit Market:** eating gives 8 for the preferred fruit and 1 for the other. In the pilot, 18–35% of fruit eaten was the preferred one, so an average eat was worth about 3.

So the Harvest-equivalent f is about 4 relative to the best eat (8 × 0.5), or about 1.5 relative to the average eat.

How often each part of the reward pays also matters:
- the attributed sanction is non-zero only on steps where the agent's action changes someone's well-being (harvests, eats, trades);
- eating rewards come at most about once per 50 steps.

Proposed values: **f ∈ {0.5, 1, 2, 3.5, 6}**. That is 0.06, 0.125, 0.25, 0.44 and 0.75 of the best eat, spaced roughly logarithmically, including the current 3.5. The lowest value tests "sanction swamped by the substrate reward", the highest "sanction swamps it". The measure of swamping is the sanction's share of each agent's total absolute reward (§3.3).

---

## 2. Environment check first (run in stage 3, before any sweep)

Scripted policies are run in the original substrate through the adapter (`FruitMarketModel` with the real backend). Their actions replace the DQN's choices, and everything else is unchanged. Each runs 16 players for 5 episodes of 1000 steps; the run is about 1 min a policy, since there's no learning.

| Policy | What it does each step |
|---|---|
| **Random** | Uniform over the 25 actions. |
| **Autarky** | If hungry (HUNGER ≥ 1) and holding fruit: eat its own fruit, or the other fruit if it has no own fruit. Otherwise walk to the nearest ripe tree of its own fruit (turn to face it, then step forward or sideways) and stand on it until it has harvested. Never offers or grapples. |
| **Trader** | Harvests its own fruit as autarky does. While holding at least one own fruit and no other fruit, it posts the fair 1-for-1 offer (give 1 own, want 1 other: action 17 or 21), then walks towards the nearest compatible partner. When hungry, it eats the other fruit if it holds any, otherwise its own. Never grapples. |

**Reported per policy** (mean ± sd over players and episodes):
- fruit eaten;
- substrate reward;
- share of preferred fruit eaten;
- mean HUNGER and steps hungry;
- trades;
- `need` well-being (mean and minimum);
- stamina.

**Expected:** trader > autarky > random on substrate reward, because each traded-for fruit is worth 8 instead of 1. Autarky should keep hunger lower than random.

**If the trader doesn't beat autarky,** the adapter is checked before any tuning:
- offers reaching `MY_OFFER`;
- partners found by `features.partners`;
- egocentric directions;
- the trade events.

The policies use the same features the agents see (`features.py`). So they also check that a good policy can be computed from the agents' observation.

---

## 3. Sweep design

### 3.1 Step A: learning parameters, baseline agents only

Homogeneous societies of 16 baseline agents, so the parameters aren't chosen for how well any principle does. The winner is used for every agent type afterwards.

- **A1, coarse: random search, 12 configurations × 3 seeds × 300 episodes.**
  - Configuration 0 is the current values (with the compressed epsilon schedule). The other 11 are sampled over the five important parameters (learning rate log-uniform; γ, target interval, buffer and ε_min uniform over their values), with batch 64 and width 128.
  - Random search rather than a grid, because with five parameters of uneven importance it covers each one's values better per run than a grid of the same size. 3 × 3 × 3 × 2 × 2 = 108 grid points against 12 samples, which still give each value of γ and of the target interval about 4 configurations.
  - The configurations are sampled once from a fixed sampling seed, and written to the sweep config before anything runs.
- **A2, fine: 6 configurations × 3 seeds × 300 episodes**, around A1's winner:
  - learning rate ×0.5 and ×2;
  - the next value either side of the parameter A1 showed to matter most (largest spread of the score across its values);
  - batch 32 and 128;
  - width 64 and 256.

  That is up to 8 candidates, cut to 6 by dropping the two changes A1 showed to matter least. A1's winner is carried into A2's ranking with its existing runs, not rerun.
- **A3, confirmation: A2's winner and the runner-up × 3 seeds × 600 episodes.**
  - Checks that the ranking holds over a longer run.
  - Gives the learning curve from which the full training length is chosen: the first episode at which the 50-episode moving average of substrate reward is within 5% of its final value, times 2, rounded up to 100.

  If the runner-up wins at 600 episodes, it is chosen, and that is reported.

### 3.2 Step A selection rule (fixed now)
For each run, take the last 50 episodes (episodes 251–300, or 551–600 in A3).

**Score** = the mean, over 3 seeds, of each seed's mean substrate reward per agent per episode in those episodes.

**Exclusions** (degenerate policies; checked on those 50 episodes, every seed). A configuration is excluded if, in any seed:
- a single action makes up more than 80% of all actions (constant no-op, a constant move or turn);
- grappling (hold, shove, pull) makes up more than 50% of all actions;
- the loss is NaN or infinite, or the last 50 episodes' mean loss is more than 100 times the median over configurations.

**Never trading** (zero trades in all 50 episodes of all seeds) is a flag, not an exclusion. Whether self-interested learners trade is something to observe, not to impose. A flagged configuration only loses a tie (below), and the flag is reported.

**Ties.** Configurations whose score is within one pooled standard error of the best score are tied. The pooled standard error is `sqrt(sd_best² / 3 + sd_other² / 3)` with sd over seeds. Among tied configurations, the winner is decided by, in order:
1. lower sd over seeds (stability);
2. not flagged for never trading;
3. earlier convergence: the first episode at which the 20-episode moving average reaches 90% of the last-50 mean;
4. fewer parameters changed from the current values;
5. smaller network, then smaller batch (cheaper).

**Fruit eaten** is reported alongside the score but doesn't rank: with the substrate's reward it is nearly the same measure, except that preferred fruit counts 8 times as much.

### 3.3 Step B: the sanction f

Step A's parameters are fixed. For each f in {0.5, 1, 2, 3.5, 6} and each type that uses the sanction (utilitarian, maximin, egalitarian, average, majoritarian, optimist, veto), in homogeneous societies: **5 × 7 × 3 seeds = 105 runs of 200 episodes.** Step B's runs are shorter than step A's because its measures are about stability and balance, not peak performance. The schedule is compressed in the same way.

**How f is chosen.** Fixed now, and identical for every type; no measure compares one type with another, and PriENE's aggregations count exactly as the single principles do. For each type *t* and each f, over the last 50 episodes of each seed:
- **S(t, f), sanction share:** mean over seeds of Σ|sanction| / (Σ|substrate reward| + Σ|sanction|), summed over the type's agents and the 50 episodes.
- **C(t, f), instability:** the coefficient of variation over seeds of the per-seed mean substrate reward (sd / |mean|; infinite if the mean is 0).
- **D(t, f), degenerate:** true if the step A exclusions apply in any seed.
- **K(t, f), not converged:** true if the 20-episode moving average of the total reward (substrate reward + sanction) still changes by more than 10% between episodes 150 and 200.

f is **eligible** only if, for **every** one of the 7 types:
- 0.05 ≤ S(t, f) ≤ 0.5, so the sanction is neither swamped nor dominant;
- D(t, f) is false;
- K(t, f) is false.

Among eligible values, the chosen f has the lowest **mean over the 7 types of C(t, f)**, each type weighted equally. Ties (within 10% of the lowest) go to:
1. the f whose mean S over the 7 types is closest to 0.25 (a balanced share);
2. then the f closest to the current 3.5.

If **no** f is eligible:
- take the f with the fewest (type, condition) failures, then apply the same order;
- report which types and conditions failed;
- propose a finer second pass (three values between the two best) before full training, for your approval.

**What is never used to choose f:**
- the substrate reward or fruit eaten of any type, beyond the degenerate checks;
- any comparison between types, in particular PriENE's aggregations against the single principles;
- any norm or equality outcome.

These are what the experiments measure, so they can't also choose the parameters.

**Reported anyway, for information:** per-type substrate reward, fruit eaten, trades and Gini under each f, after the choice and clearly separated from it.

### 3.4 Seeds and aggregation
- **3 seeds per configuration** in every step: run seeds 1, 2, 3, each passed to `run.py -s`.
- **What a seed fixes:** PriENE's randomness (network initialisation, exploration, replay sampling) and each episode's substrate seed.
- **What it doesn't:** the substrate's own variation, since Melting Pot isn't reproducible from its seed (design §12). Seed-to-seed spread therefore includes the substrate's own variation, which is why spread over seeds is part of every rule.
- **Aggregation:** per seed, a mean over the last 50 episodes; per configuration, the mean and sd of those 3 values. Learning curves are 20-episode moving averages, mean ± sd over seeds.

### 3.5 Early stopping
Checked at episode 100 of each run, when ε is about 0.11 under the compressed schedule. A run is stopped, and its configuration's remaining seeds are not started, if:
- its last-20-episode mean substrate reward is **below the random policy's** (§2), so it is learning nothing useful;
- or it's degenerate on episodes 81–100 by the step A rules;
- or its loss is NaN.

Stopped runs are kept and reported as stopped, never deleted. No "below median" rule: a slow starter (say high γ) can overtake later, and a 12-configuration search can't afford to lose it.

---

## 4. Budget

### 4.1 Measured on this Mac (13 Nov 2026)
Measured while your Harvest training (PID 81285) was also running, so these are pessimistic.

| | ms per step (16 learning agents) | s per 1000-step episode |
|---|---|---|
| 1 process, baseline | 19.7 | about 21 |
| 1 process, utilitarian / veto | 21–24 | about 23 |
| 3 in parallel | 26 each | about 27 each, so about 9 overall |
| 5 in parallel | 35 each | about 36 each, so about 7 overall |

**4 parallel processes** is the planned setting: about 31 s per episode each, about 8 s per episode overall. The M1 Pro has 6 performance and 2 efficiency cores.

### 4.2 Totals (4 in parallel, machine otherwise idle)

| Step | Runs | Episodes each | Hours (≈ runs × episodes × 31 s ÷ 4) |
|---|---|---|---|
| Environment check | 3 policies | 5 | under 0.1 |
| A1 coarse | 36 | 300 | 23 |
| A2 fine | 18 | 300 | 12 |
| A3 confirmation | 6 | 600 | 10 (two waves: 4 runs, then 2) |
| B | 105 | 200 | 45 |
| **Total** | 165 | | **about 90 h, about 3.8 days** |

Early stopping makes this shorter. Running alongside a Harvest training run makes it slower.

**If that's too long, cut in this order:**
1. Step B: f values 5 → 4 (drop 6 if the pilot's sanction shares are already over 0.5 at 3.5; otherwise drop 0.5). Saves 9 h.
2. Step B: 200 → 150 episodes. Saves 9 h more.
3. A1: 12 → 9 configurations. Saves 6 h.
4. A3: only the winner (3 runs, one wave). Saves 5 h.

All four give about 61 h, about 2.5 days. Seeds stay at 3 in every case.

### 4.3 The cluster
Your SLURM scripts (`runscripts/`) suggest a cluster. Every sweep run is independent, so the whole sweep is one job array, and wall time would be about the longest single run (about 2–4 h), if enough nodes are free.

The catch: Melting Pot's `dmlab2d` wheels for Linux need glibc 2.35 or newer (`manylinux_2_35`), and many clusters run older systems. The runner will be written so the same command works both here and in a job script. Whether it installs on the cluster has to be tried there.

---

## 5. What stage 2 will need (for information; built only after approval)

- **A training configuration.** The hyperparameters in §1.1 are constants in code now. Stage 2 adds a `TrainingConfig` with today's values as defaults, passed from the model down to `DQNDecisionModule` and `DQN`, plus a training episode budget. Harvest's runs use the defaults, so the 950 reference hashes are checked after the change, which touches shared code. The sanction becomes part of it, for Fruit Market.
- **Setting it from the sweep.** `run.py train` gets one non-interactive option, `--training-config FILE`. Every other answer still comes through the prompts, which the runner fills in, as `tests/reference_outputs.py` does. This is a flag although you prefer prompts: the file is for tools, not for people at the keyboard. If you'd rather, the runner can call the training code directly instead. *(Decision 2 below.)*
- **The sweep file** (YAML; PyYAML is already in prenv) lists the scenario, the fixed prompt answers, and the swept parameters with their ranges.
  - **Allowed:** only the training parameters of §1.3 and the sanction.
  - **Refused, with an error:** anything else, including every substrate parameter, episode length, players, grappling, well-being and observation mode.
- **Per-run folders.** Each run executes with its own folder as working directory: `sweeps/<scenario>/<step>/<config id>/seed<k>/`. Every relative path `run.py` writes (`data/results/current_run/`, `data/model_variables/trained/`) then lands inside it, so nothing is overwritten and existing results aren't touched. A run is complete when its folder has `run.json` with `"status": "done"`. Restarting skips completed runs and restarts incomplete ones from scratch.

---

## 6. Decisions for you

All decided (13 Nov 2026):
- **1.** No failure penalties: the substrate's own reward only, plus the sanction for ethics agents.
- **2.** Training parameters come from a config file per scenario:
  - `configs/training/harvest.yaml` holds today's values, used by every harvest scenario;
  - `configs/training/fruit_market.yaml` holds the same values with Fruit Market's sanction.

  `run.py` picks the scenario's file automatically, and `--training-config FILE` swaps in another. The sweep writes each run's file and passes it (§7).
- **3.** No cluster for now. The sweep will be run later on a more powerful computer, so the tool runs anywhere `run.py` does.
- **4.** The proposed ranges (§1.3).
- **5.** f ∈ {0.5, 1, 2, 3.5, 6}.
- **6.** The selection rules in §3.2 and §3.3 as written (my recommendation, which you delegated). They're implemented in `src/sweep/summarise.py`.

1. **Failure penalties:** the design has none (substrate reward only). Keep it that way for the sweep?
2. **How the runner sets parameters:** a `--training-config FILE` flag on `run.py train` (recommended), or the runner calling the training code directly?
3. **Where to run:** this Mac, 4 in parallel (about 3.8 days, or about 2.5 with the cuts; slower while Harvest training runs), or try the cluster?
4. **Ranges:** the proposed ones (§1.3), or Appendix B's? Send them if they differ.
5. **Sanction values:** {0.5, 1, 2, 3.5, 6} as proposed?
6. **Selection rules** (§3.2, §3.3): approve as written, so they're fixed before anything runs.

---

## 7. Sweep infrastructure (stage 2, built 13 Nov 2026)

### Training configs (`src/training_config.py`, `configs/training/`)
- **One file per scenario.** `harvest.yaml` (every harvest scenario) holds the values Harvest agents have always trained with; `fruit_market.yaml` holds the same with sanction 3.5.
- **What a file covers:**
  - learning rate, γ, batch size, hidden units, Huber δ;
  - minimum and maximum experiences, target-update interval;
  - epsilon start, minimum and decay;
  - `max_training_episodes` (null: train until epsilon reaches its minimum, as before);
  - the sanction.
- **Choosing a file.** `run.py train|test` uses the scenario's file; `--training-config FILE` uses another. A training run whose config differs from its scenario's records it in the checkpoint's `settings.json`, so runs with different parameters can't share a checkpoint. A Fruit Market test uses the config its checkpoint was trained with.
- **Refused:** a config, or a sweep, that names a scenario parameter (players, steps, grappling, well-being, hunger, regrowth, trade radius, seeds, …) or an unknown name gets an error.
- **Harvest unchanged.** With its own file a Harvest run is unchanged: all 950 reference hashes matched after this change.
- **New report columns.** Fruit Market's episode report gained the three columns the selection rules read: `max_action_share`, `grapple_share` and `mean_abs_sanction`.

### The sweep tool (`src/sweep/`)
- **Sweep files:** `configs/sweeps/*.yaml` give the scenario, the step, the agent types, the seeds, the episodes, the search (random, grid or list) over training parameters, the early-stopping thresholds and the selection rule. `fruit_market_A1.yaml` and `fruit_market_B.yaml` are written. A2 and A3 are written once A1 has a winner, since their values depend on it (§3.1).
- **Fixed answers.** The answers to `run.py`'s prompts come from the tool (`spec.scenario_answers`), not the sweep file. For Fruit Market they are always the substrate's defaults, so a sweep can't change the scenario.
- **Run:** `python -m src.sweep.run configs/sweeps/fruit_market_A1.yaml --workers 4` (`--dry-run` lists what's left to do).
  - Each run is `python run.py train -s <seed> --training-config <its file>`, run in its own folder: `sweeps/<scenario>/<step>/<configuration>/<agent type>/seed<seed>/` (`sweeps/` is git-ignored).
  - Each folder holds the run's `training_config.yaml`, a `run.json` (configuration, values, seed, answers, full training config, git commit, times, status and reason) and `run.log`.
  - **Stopping and restarting** is safe:
    - finished, early-stopped and skipped runs are skipped;
    - a run that had saved progress is continued from its last finished episode;
    - a failed run is retried.

    `configurations.json` stops a changed sweep file from mixing with an earlier version's runs.
- **Early stopping** (§3.5) is checked while a run is going, from its episode report. A stopped configuration's remaining seeds are marked skipped.
- **Summarise:** `python -m src.sweep.summarise configs/sweeps/fruit_market_A1.yaml` writes, in the sweep's folder:
  - `sweep_results.csv`: a row per run, plus mean and sd rows per configuration;
  - `ranking.csv`, by the step's rule;
  - `curves/<configuration>.png`;
  - `summary.md`: the recommended configuration, the ranking, and how sensitive the score is to each parameter.
- **Tests:** `tests/test_sweep.py` and `tests/test_training_config.py` cover config parsing, refusing scenario parameters, run folders, logging, resuming, a changed sweep file, early stopping, seeding (same seed, same actions; different seed, different actions), and the summary and both selection rules on fake sweeps.

### Before running the real sweep
1. Run the environment check (§2), and set `early_stopping.min_reward` in both sweep files to the random policy's mean substrate reward per agent per episode.
2. After A1: write `fruit_market_A2.yaml` (a list search around the winner, §3.1), then `fruit_market_A3.yaml`.
3. After A3: save the chosen config as `configs/training/fruit_market_step_a.yaml`, and set it as `base_config` in `fruit_market_B.yaml`.
4. On the other computer: install as in `requirements-fruit-market.txt`, then run each step with `--workers` set to what that machine can take.
