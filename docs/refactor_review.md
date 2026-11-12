# Refactor review

Review of the code on `dev` at `a24e1e8` (plus the reference-output script added for this review). No code has been changed.
The goal is a smaller codebase with identical outputs for the same seed, so every finding below is graded by whether it could change
the order or number of random draws, floating point summation order, the order agents or berries are created or scheduled, or the
order of output rows or columns. Any change that could is marked **high** risk.

## Safety net

| | |
|---|---|
| Tests | **879 passed, 0 xfail, 0 skipped** (`pytest`, 4m 14s with coverage) |
| Coverage of `src/` | **86%** (2733 statements, 383 missed); `run.py` isn't imported by any test, so it isn't measured |
| Reference outputs | `tests/reference_outputs.py`: **32 runs** (93 steps), **950 output files** hashed, saved in `tests/reference_data/` |
| Deterministic | **Yes**: a second full run (separate processes, 4 workers, in a different order, while a training run was using the CPU) matched all 950 hashes; the paper graphs run, added last, matched on a rerun too |

### Coverage gaps

| Module | Cover | Not covered | Covered instead by |
|---|---|---|---|
| `data_handling/data_analysis.py` | 27% | everything but the Gini and statistics helpers | `run.py graphs` runs in the reference set (current-run and paper graphs), except the `get_normalised` path (dead, see B1) |
| `data_handling/norm_processing.py` | 38% | the whole pipeline | the reference graphs run, except `_generalise_norms` and its helpers (parked, see B2) |
| `checkpoints.py` | 89% | `days_in_name`, `checkpoint_max_days` (157-174) | every `run.py test` step in the reference set (days come from `settings.json`) |
| `harvest_model.py` | 97% | defensive raises (89, 200, 385, 387, 434, 616, 624, 659), invalid-JSON warning (785-787) | nothing needed: these only fire on corrupted state |
| `run.py` | not measured | all of it | every reference step goes through `run.py`'s prompts, except `-l` (wandb) and render |

No characterisation tests are needed beyond the reference set: everything I propose refactoring is either covered by tests or by the
reference hashes. The two exceptions are flagged where they appear (A15 run.py prompts: covered by the reference set, but console text
isn't hashed; B1/B2: dead code, so there is nothing to characterise).

### The reference script

`python tests/reference_outputs.py` reruns everything and compares; `--save` saves a new reference (never used during refactoring);
`--only <text>` runs a subset. Runs go through `run.py` itself, answering its prompts, so prompts, output and checkpoint names,
`settings.json`, `agent_types.json` and the continue-training prompt are all covered. The only change is that runs are cut short
(training after 10 episodes, testing after 4, with 10-day episodes); allotment regrowth delays have no prompt, so they are run directly.

| Covered | |
|---|---|
| Scenarios × seasons × society | every scenario × seasons on/off × homogeneous / mixed-split (+ mixed-separate for allotment): 22 runs, each trained, then tested with rule-based norms and with Bayesian norms (and once with norms off, without seasons) |
| Agent types | a different agent type for each of those runs, so all 8 are trained and tested; plus baseline alone, and `all` (every type) trained in basic and tested in colours, capabilities and allotment |
| Other options | 6 agents with 3 allotments (mixed) and 3 capabilities; commons with its own K, r and R0; allotment regrowth delays [1, 3] |
| Continued training | basic, allotment (mixed-separate) and commons, stopped after 5 episodes and continued to 10 through the continue prompt |
| Graphs | `run.py graphs` for the current run (all 8 types in colours) and for `paper` |

Hashes: every CSV and JSON byte for byte; checkpoints by their model config and weights (`.keras` files record the time they were
saved, so their bytes differ between identical runs); graph images aren't hashed (their bytes carry library metadata), the data
behind them is.

## Findings

Lines are from `a24e1e8`. "Draws" says whether the change could alter random draws, summation order, creation/scheduling order or output
row/column order.

### A. Duplication

| # | Where | Change | Lines saved | Risk | Draws / order |
|---|---|---|---|---|---|
| A1 | `_init_agents`: `harvest_model.py` 319-326, `colours_harvest.py` 49-56, `capabilities_harvest.py` 57-70, `allotment_harvest.py` 185-196 | One `HarvestModel._init_agents`, with a hook for an agent's region (the full grid, or its allotment) and `allocation_id` taken from `_get_allocation_id` when the scenario has allocations. The `assert` in the base and the `NumAgentsException` in two scenarios become the exception; `num_living_agents` (set, never read) goes | 25 | low | Agents are still created in id order, each placed as it is created, so placement draws are unchanged |
| A2 | `_init_berries`: `basic_harvest.py` 24-35, `colours_harvest.py` 34-47, `capabilities_harvest.py` 42-55, `allotment_harvest.py` 147-158; abstract stub `harvest_model.py` 315-317 | One `HarvestModel._init_berries` looping over allocations, with the same region hook as A1 (commons keeps its own, which places by free cell) | 32 | low | Berries are created in the same order (allocation, then berry), so ids and placement draws are unchanged; basic's single allocation already holds `num_start_berries` |
| A3 | Scenario constructors: regrowth-delay rejection in `basic` 15-16, `colours` 15-16, `capabilities` 16-17, `commons` 44-45; the build sequence (allocations → agents → berries → seasons) in all four allocation scenarios | A class attribute saying whether a scenario takes regrowth delays, checked at the start of `HarvestModel.__init__`; one `_build` template method for the sequence | 12 | low | None: same calls in the same order; the check still happens before anything is created |
| A4 | Even splits with the remainder to the first: `allotment_harvest.py` 85-96 (columns), 110-119 (agent groups), 121-133 (berry shares); `capabilities_harvest.py` 27-40 (groups); `harvest_model.py` 418-424 | One `_even_sizes(total, n)` and the existing `_split_into_groups` | 18 | low | None: integer arithmetic only |
| A5 | Report columns listed twice: agent report `harvest_model.py` 630-652 and 713-733; model episode report 661-680 and 749-773 | Each report's columns defined once, by the function building a row; the empty reporter built from those names | 35 | low-medium | Column order is kept by building both from one ordered list. The empty frame's dtypes must match exactly (they decide concat dtypes, and so CSV formatting): it will be built the same way, `{column: []}`; the hashes check this |
| A6 | `_mean_loss`, `_mean_reward`, `_mean_epsilon`: `harvest_model.py` 1043-1072 | One `_mean_over_agents(value)` | 15 | low | Summation stays in schedule order. The quirk that a sum of exactly 0 returns `int` 0 (written `0`, not `0.0`) must be kept, and `_mean_loss`'s training gate |
| A7 | Central-tendency dicts: `data_analysis.py` 297-336, `norm_processing.py` 261-289 | One helper building mean/median/stdev/summary for a list of (name, column), keeping key order | 45 | low | None: same pandas calls on the same columns; output column order kept by the list order |
| A8 | `_display_end_episode`: `data_analysis.py` 223-231 (8 near-identical calls) | Loop over a table of (column, title, label) | 4 | low | None (images only) |
| A9 | Reward tables: `harvest_agent.py` 307-332 | Ethics rewards as the baseline table with its differences | 8 | low | None: rewards are only looked up by key, never iterated |
| A10 | Bayesian feature specs: `harvest_agent.py` 377-408 (each spec spread over 6 lines) | A compact table of (name, index, thresholds) | 22 | low | Spec order decides predicate order, and so `behaviour_base` order and the Bayesian CSV: kept by keeping the table order |
| A11 | Allocation filters: `harvest_model.py` 176-200 (`get_uneaten_berries_coordinates`, `get_uneaten_berry_by_coords`), `moving_module.py` 43-50, 134-138 (restricted/unrestricted branches) | `allocation_id is None or b.allocation_id == allocation_id`; the moving module passes its allocation or None | 12 | low | None: same iteration over `self.berries` |
| A12 | Principle dispatch: `ethics_module.py` 35-47, 51-59, 84-96 (if-chains over principles and aggregations) | Lookup tables | 10 | low | None |
| A13 | Exceptions: `harvest_exception.py` (259 lines, 28 classes, each storing attributes for its `__str__`) | Each class builds its message once and passes it to `Exception`; `HarvestException`'s no-op `__init__` goes | 140 | low | None: no outputs; messages stay word for word (tests check them). Attributes like `.file_name` would go: nothing reads them |
| A14 | `run.py` 207-218 (scenario if-chain in `create_and_run_model`), 53-56 (two hand-written lists of 8 file names) | A scenario → class table; list comprehensions | 8 | low | None |
| A15 | `run.py` number prompts: `get_number_input` 250-263, `get_integer_input` 308-323, `get_season_length_input` 275-291 | One helper | 15 | medium | No results change, but each prints its own prompt and error text; keeping every message word for word makes the merged helper fiddly. Console text isn't hashed |
| A16 | `dqn_decision_module.py` 104-105 (`makedirs` twice for the same folder) | Once | 1 | low | None |

### B. Dead code

| # | Where | Change | Lines saved | Risk | Notes |
|---|---|---|---|---|---|
| B1 | `data_analysis.py`: the `get_normalised` path. `_normalise_step_across_episodes` 95-118, `_days_left_to_live_results` / `_berries_consumed_results` 166-192, `_calculate_column_across_episode` 288-295, `_calculate_max/_min/_total` 268-272, 285-286, `_display_dataframe` 253-261, branches in 65-69, 80-81, 89-91 | Remove. `run.py` always passes `get_normalised=False` (lines 64, 94), so none of this can run | 95 | low | **Your decision**: it is an analysis feature that is switched off, not broken. No test uses it |
| B2 | `norm_processing.py` 36-39, 163-259: `_generalise_norms`, `_convert_to_rule_list`, `_generate_norms_tree`, `_print_tree` | Remove | 97 | low | **Your decision**: the comment calls it "a parked feature, not dead code … pending a decision". Nothing calls it |
| B3 | Attributes set but never read: `HarvestModel.end_day` (91, 150), `.total_episode_reward` (372), `.agent_id` (101, 362); `HarvestAgent.max_berries` (190), `.days_survived` (72, 179, 194, 299); `ethics_module.day` (`harvest_agent.py` 127) and `EthicsModule.society_well_being` (25); `DQN.total_episode_reward`, `DQNDecisionModule.total_episode_reward`; `RenderPygame.running` and the unused `pygame.time.Clock()`; the `"all_principles"` render colour (no such agent type) | Remove | 18 | low | None of these reach an output |
| B4 | Unreachable branches: `run.py` 364-365 (argparse already restricts the option); `_add_agent` `if a.agent_type != "berry"` (`harvest_model.py` 361, always true); `_build_norms_module`'s raise (`harvest_agent.py` 345, the model rejects unknown types first); `"age" in value` (`norms_module.py` 104, always true); `done = False` before it is assigned (`harvest_agent.py` 119); the `allocation_id` if/else (`harvest_agent.py` 83-86, one line); `@abstractmethod` on a class that isn't an ABC (`harvest_model.py` 25, 315) | Remove or simplify | 12 | low | None |
| B5 | Unused import `scipy.stats` (`data_analysis.py` 7) (pyflakes finds no others) | Remove | 1 | low | None |
| B6 | Used only by tests: `BayesianNormsModule._precondition_satisfied` (385-391), `get_all_behaviours_by_certainty` (222-228), `episode_report` (230-270); `_initialise`'s `return self` | **Ask before removing**: each has tests in `test_bayesian_norms_module.py`, which would have to go too | 0-55 | low | `episode_report` looks like a useful inspection tool; `_precondition_satisfied` looks like a leftover |
| B7 | `MovingModule`'s `training` parameter and attribute (`moving_module.py` 13, 22, 26): documented as unused | Remove | 3 | low | **Changes a constructor that tests call** (`test_moving_module.py` 11-14, 125-126 pass `training=`), so the tests' calls would have to change too: your decision |
| B8 | Stale docstrings: `Berry` ("allocated_agent_id"), `NNetwork` ("MONNetwork"), `HarvestModel` ("max_with", "max_episode"), `_generate_resource_allocations` (mentions run.py prompts that no longer exist) | Fix the text | 0 | low | None |

Kept on purpose: `REMOVED_OPTIONS` and `RemovedOption` in `run.py` (they explain removed flags), and the `initial_health` /
`seasonal_berry_allocation` names in `data_analysis.py` (needed to read old results).

### C. Simplification

| # | Where | Change | Lines saved | Risk | Draws / order |
|---|---|---|---|---|---|
| C1 | `MovingModule._move` (`moving_module.py` 90-115): four near-identical branches | A table of direction → (dx, dy), with the same bounds check and `OutOfBounds` arguments | 12 | low | None. (`_calculate_distance` and the A* search are left exactly as they are: the float priorities decide tie-breaking between paths) |
| C2 | `get_society_well_being` (`harvest_model.py` 202-216): `np.append` in a loop | Build a list, then `np.array(..., dtype=float)` | 4 | low-medium | Same order; the dtype must stay float64 even when every value is an integer 0, so it is set explicitly |
| C3 | Emerged-norm bookkeeping (`harvest_model.py` 836-888) | Tighten `_add_agent_behaviour_to_snapshot` / `_update_episode_emerged_norm` | 10 | medium | Dictionary insertion order becomes JSON key order in `_emerged_norms.json`, so it must be kept exactly |
| C4 | `_check_mixed_society` (`allotment_harvest.py` 75-83) repeats the even-number check in `_half_population` (`harvest_model.py` 348-354) | Reuse it (same message) | 2 | low | None |
| C5 | `render_pygame.py` 57-62: the same colour lookup in both branches | Once | 2 | low | None |

Not proposed: anything touching `_choose_surplus_berries`, the season and regrowth draws, `_generate_zipf_distribution`, DQN training or
replay sampling, or the order of `schedule.agents` iteration. These are all where draws happen, and they are already short. Also not
proposed: making the agent report faster (it concatenates a DataFrame every step, which is O(n²) per episode). That is a performance
change, not a size one, and the row dtypes it would touch decide the CSV formatting.

### D. Structure (notes for the environment-agnostic interface, not for this task)

- **D1. `HarvestModel` does four jobs** (1071 lines): environment dynamics, seasons and regrowth, reporting (CSV/JSON writing), and norm
  analysis. Splitting out a reporter and a norms tracker would leave a model class that is only the environment, which is what a Melting
  Pot adapter would replace.
- **D2. The observation layout is defined in three places**: `HarvestAgent.observe` (features in order), `_calculate_n_features`, and
  the Bayesian specs' hard-coded indices (0-3, then 4 + others). The rule-based antecedents define a fourth, different view. One
  observation spec (names, indices, thresholds) would let an agent describe its inputs independently of the environment.
- **D3. The agent reaches into the model**: `get_society_well_being` with two boolean flags (three different views), `can_throw_to`,
  `get_living_agents`, `move_agent_to_cell`, and the moving module's `get_cell_contents` and berry lookups. Naming the three well-being
  views, and putting movement behind an environment interface, would be the seams for another environment.
- **D4. Actions are environment-specific strings** (`throw_<id>`), parsed back in `_perform_action` and collapsed for norms in two
  ways (the rule-based consequent rule and the Bayesian `norm_action_index`). An action spec would let both norms modules share one
  collapse.
- **D5. `run.py` does everything at import time** (argument parsing, prompts, runs), which is why the reference script has to run it
  with `runpy`. A `main()` would make it importable; and `create_and_run_model` / `run_all` take 20+ positional parameters, which a run
  settings object would replace.
- **D6. The output folder** `"data/results/current_run/"` is written out 10 times across `harvest_model.py`, `commons_harvest.py` and
  `run.py`; one constant would make it configurable.

### E. Debug output (your decision: removing it changes console output, not results)

| # | Where | What it prints |
|---|---|---|
| E1 | `norms_module.py` 115 | "agent N clipping behaviour base, length is …" every time a rule-based behaviour base is clipped: every 10 days per agent once it holds over 100 behaviours, so a lot during long test runs |
| E2 | `data_analysis.py` 198 | "no data for … skipping …" when a swarm plot has no data (a reasonable warning) |
| E3 | `data_analysis.py` 421, 429 | exceptions from ANOVA / Tukey tests (useful: the graphs still finish) |
| E4 | `harvest_model.py` 786 | warning when an emerged-norms file isn't valid JSON (useful) |

There are no longer any print statements in the scenario constructors. My suggestion: remove E1, keep E2-E4.

## Prioritised steps

Each step is one commit, independent of the others, checked with the full test suite and the reference hashes.

| Step | Findings | Lines saved | Risk |
|---|---|---|---|
| 1 | B3, B4, B5, B8: unused attributes, unreachable branches, the unused import, stale docstrings | 31 | low |
| 2 | A13: exception classes | 140 | low |
| 3 | A6: `_mean_*` helpers | 15 | low |
| 4 | A4 + C4: even splits; allotment's mixed-society check | 20 | low |
| 5 | A1: one `_init_agents` | 25 | low |
| 6 | A2 + A3: one `_init_berries`; scenario constructor template and regrowth-delay check | 44 | low |
| 7 | A5: report columns defined once | 35 | low-medium |
| 8 | A11 + C1 + C2: allocation filters, moving module, well-being list | 28 | low-medium |
| 9 | A12 + A9: ethics dispatch tables; reward tables | 18 | low |
| 10 | A10: Bayesian feature-spec table | 22 | low |
| 11 | A7 + A8: analysis central tendencies and violin plots | 49 | low |
| 12 | A14 + A16 + C5: run.py tables, makedirs, render | 11 | low |

Total for steps 1-12: **about 390 lines** of the 5051 in `src/` and `run.py` (about 8%).

Needing your decision first:

| Finding | Lines saved |
|---|---|
| B1: remove the `get_normalised` analysis path | 95 |
| B2: remove the parked norm generalisation | 97 |
| B6: remove test-only Bayesian methods (and their tests) | up to 55 |
| B7: remove `MovingModule`'s `training` parameter (and update the tests' calls) | 3 |
| A15: merge `run.py`'s number prompts (console text kept word for word) | 15 |
| C3: tighten emerged-norm bookkeeping (medium risk) | 10 |
| E1: remove the clipping print | 1 |

With all of those: **about 665 lines** (about 13%).

Also for you to decide: whether to commit `tests/reference_data/outputs/` (the reference run's files, 31 MB) or only
`tests/reference_data/hashes.json` (which is all the comparison needs) with `outputs/` in `.gitignore`.
