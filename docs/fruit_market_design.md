# Fruit Market scenario: design

Stage 1 of adding Melting Pot's `fruit_market__concentric_rivers` as a PriENE scenario, with scalar observations so the existing DQN can be used. **No code has been changed.** Implementation starts only once this design is approved.

Sources:
- upstream `google-deepmind/meltingpot` (main branch, cloned 13 Nov 2026). The files used are `configs/substrates/fruit_market.py`, `configs/substrates/fruit_market__concentric_rivers.py`, `lua/levels/trade/components.lua`, `lua/levels/stamina/shared_components.lua`, `lua/modules/avatar_grappling.lua`, `lua/modules/avatar_library.lua` and `lua/modules/component_library.lua`;
- the meltingpot-pz documentation site;
- this repository at `a15063e`.

Line references such as `components.lua:586` are to those files.

---

## 0. Summary

Decisions made on 13 Nov 2026 are marked **(decided)**. Every decision is now made (§14).

| Topic | Design |
|---|---|
| What changes from the original substrate | **Only two things:** (1) ethics agents learn from the substrate's reward plus an ethics sanction; (2) the norms modules bucket each step's action into a label. The actions, rewards, map, roles, player count, episode length and all game logic are the original substrate's. Our additions to the substrate are read-only observers (§5.0). **(decided)** |
| Backend | Upstream `dm-meltingpot`. Spatial features come from Melting Pot's own `LocationObserver` (positions, orientation) and a small read-only Lua observer of ours (tree type and ripeness, who is holding whom) (§2.9). **(decided)** |
| Install | Into `prenv` with `--no-deps`, pinned against prenv's current numpy, TensorFlow and Keras. A dry run without the pins would have upgraded numpy to 2.4.6; with them, nothing existing changes (§2.9). **(decided)** |
| Agent actions | The substrate's 25 actions, unchanged and in the same order. The one exception is the grappling option from your brief, which is on by default (original substrate); turned off, it masks actions 9–11 (§5). |
| Stepping | Simultaneous, as in the substrate. All players choose from the observation at step *t*, the substrate steps once, then each agent learns. `run_simulation` is reused unchanged (§3.3). |
| Rewards | The substrate's own reward (8 for eating the preferred fruit, 1 for the other, 0 otherwise). Ethics agents also get a sanction of ±3.5 (§6). **(decided)** |
| Ethics sanction | Every step, **attributed**: the sanction reflects only the agent's own action and its trade, as in Harvest. **Joint** is kept as an option for an ablation (§6.3). **(decided)** |
| Well-being | `need` (default) or `preference`, with the corrected preference form. Time spent hungry slowly reduces well-being, by 0.1 per hungry step (§7). **(decided)** |
| Norm labels | Derived after each step from the primitive action and its effect, including the three added labels, and passed to the unchanged norms modules (§9). **(decided)** |
| Refactors | Five, all behaviour-preserving and checked against the 950 reference hashes after each one (§3.2). |
| Scale and compute | The substrate's own scale: 16 players × 1000 steps **(decided)**. That's about **10 days of training per agent type** on this Mac, so about 80 days for all 8 types one after another. §11 has ways to speed this up that don't change the substrate. |
| Options | Asked at prompts, like every other `run.py` option (§10). |

The brief was cut off in section 9. §9.4 covers cooperative and uncooperative norms, and sections 10–14 cover options, checkpoints, compute, testing, the implementation plan and decisions.

---

## 1. What was read

- **Model:** `src/harvest_model.py`, `src/scenarios/*.py`.
- **Agent and modules:**
  - `src/agent/harvest_agent.py`;
  - `src/agent/modules/` (DQN decision, ethics, moving, rule-based and Bayesian norms);
  - `src/agent/dqn/` (DQN and network).
- **Run script, checkpoints and outputs:**
  - `run.py`, `src/checkpoints.py`;
  - `src/data_handling/` (analysis, norm processing, rendering).
- **Tests:** `tests/` (870 tests), including `tests/reference_outputs.py` and `tests/reference_data/hashes.json` (950 hashes).

The facts §3 relies on:
- `HarvestAgent` combines two things:
  - generic learning logic (decision, ethics, norms, reward assembly);
  - Harvest's body (health, berries, the move/eat/throw effects, and `observe()`).
- `HarvestModel` combines a different two:
  - generic society machinery: seeding, reports, emerged-norm bookkeeping, means and the Gini coefficient;
  - berry and grid logic.
- `run_simulation` uses only these members of the model: `step()`, `finish_episode()`, `training`, `epsilon`, `min_epsilon`, `episode`, `max_episodes`, `_mean_reward()`, plus `max_width`/`max_height` when rendering.
- `EthicsModule`, `NormsModule`, `BayesianNormsModule`, `DQN` and `NNetwork` contain no Harvest knowledge. The agent registers everything domain-specific with them: feature specs, consequent rules, the norm action index, and the action list.

---

## 2. Fruit Market: findings

### 2.1 Substrate and map
- **Map:** a 31 × 31 ASCII map (`WORLD.RGB` is 248 × 248 px, with 8 px sprites). It has:
  - three concentric river rings (168 river tiles);
  - 21 spawn points in the centre;
  - 534 potential-tree tiles (every ground tile, `~` and `P`).
- **Players:** 16 by default, with `default_player_roles = 8 × apple_farmer + 8 × banana_farmer`. Any number of roles can be passed to `build`. Each player needs a spawn point, so the map holds up to 21.
- **Episode length:** exactly 1000 frames (`maxEpisodeLengthFrames=1000`). There is no stochastic ending and no death: players are never removed.

### 2.2 The 25 actions (`fruit_market.py:712–759`)

Movement is egocentric. `move` values 1–4 are forward, right, backward and left relative to the player's facing (`avatar_library.lua:167`).

| Index | Name | Substrate components |
|---|---|---|
| 0 | NOOP | — |
| 1 | FORWARD | move=1 |
| 2 | BACKWARD | move=3 |
| 3 | STEP_LEFT | move=4 |
| 4 | STEP_RIGHT | move=2 |
| 5 | TURN_LEFT | turn=−1 |
| 6 | TURN_RIGHT | turn=+1 |
| 7 | EAT_APPLE | eat_apple=1 |
| 8 | EAT_BANANA | eat_banana=1 |
| 9 | HOLD | hold=1 |
| 10 | SHOVE | hold=1, shove=+1 |
| 11 | PULL | hold=1, shove=−1 |
| 12 | CANCEL_OFFER | offer_cancel=1 |
| 13–24 | offers | (offer_apple, offer_banana) as below |

Offer quantities: **positive means "I want", negative means "I give"** (`components.lua:569–594`). The loop is `for a in range(-3, 3)`, so the largest quantity a player can **want** is 2, while it can **give** up to 3. The set is therefore asymmetric.

| Index | (apple, banana) | Meaning | Relative to 1:1 |
|---|---|---|---|
| 13 | (−3, +1) | give 3 apples for 1 banana | generous |
| 14 | (−3, +2) | give 3 apples for 2 bananas | generous |
| 15 | (−2, +1) | give 2 apples for 1 banana | generous |
| 16 | (−2, +2) | give 2 apples for 2 bananas | fair |
| 17 | (−1, +1) | give 1 apple for 1 banana | fair |
| 18 | (−1, +2) | give 1 apple for 2 bananas | demanding |
| 19 | (+1, −3) | give 3 bananas for 1 apple | generous |
| 20 | (+1, −2) | give 2 bananas for 1 apple | generous |
| 21 | (+1, −1) | give 1 banana for 1 apple | fair |
| 22 | (+2, −3) | give 3 bananas for 2 apples | generous |
| 23 | (+2, −2) | give 2 bananas for 2 apples | fair |
| 24 | (+2, −1) | give 1 banana for 2 apples | demanding |

That gives 6 generous, 4 fair and 2 demanding offers.

### 2.3 Harvesting
Harvesting has no action of its own. It happens when a player **enters** a ripe tree's tile (`components.lua:185`):
- From then on, while the player stays on the tile, it harvests with probability `strongProbability = 1` for its own specialty and `weakProbability = 0.04` for the other fruit. It gets 2 fruit of either kind.
- Harvesting runs after movement in the same step. An apple farmer stepping onto a ripe apple tree therefore gets 2 apples that step.
- A banana tree takes it 25 steps on average, standing still. Stepping off cancels the attempt.
- If the tree regrows while the player is still on it, the player must leave and re-enter, because `onEnter` fires only once.

Other tree rules:
- **Tree types:** each potential tree is drawn once per episode by the Lua RNG: 90% empty, 5% apple, 5% banana. That is about 27 apple and 27 banana trees per episode, in fixed positions, all ripe at the start.
- **Regrowth:** a harvested tree is unripe for `regrowthTime = 50` steps.

### 2.4 Trading (`components.lua:496–796`)
- **Offers persist.** An offer stays until it is cancelled, replaced by a new offer, or a trade clears it.
- **Unaffordable offers are hidden but kept.** If a player cannot cover the **largest** amount it offers to give (`hasEnough`), its offer is hidden: `MY_OFFER` and `OFFERS` show (0, 0) and it cannot trade. The offer is still stored, and goes public as soon as the player's inventory covers it.
- **When offers match.** A and B are compatible when each gives at least what the other wants (`isCompatible`), both offers are valid (one positive and one negative component) and both are affordable. This is a dominance condition, not the docstring's `X = W, Y = Z`. Each player keeps only nearby offers that no other nearby offer strictly dominates. A trade needs each player to be in the other's non-dominated set.
- **Range.** "Nearby" means a Euclidean disc of radius 4 (`queryDisc`).
- **What changes hands is each side's *want*, not its give** (`resolve`: `min` of the two). A generous offer such as (−3, +1) matched with (+1, −1) trades 1 apple for 1 banana. "Generous" therefore means *willing to accept a worse rate*, not *gives more*.
- **Resolution order:** once per step, players are taken in a random order (Lua RNG). Each completed trade clears both offers and emits a `trade` event naming the two players and the offered and traded quantities.

### 2.5 Hunger (`avatar_library.lua:1124–1174`, `PeriodicNeed`)
- **The counter:** a hunger level starts at `delay = 50` and goes down by 1 each step. `HUNGER = max(1 − level/50, 0)`, so it rises by 0.02 per step from 0 (just ate) to 1 (50 steps without eating). It **keeps rising after that** (the level goes below 0), by 0.02 a step: after 200 steps without eating it is 4. *(Corrected during implementation: the first version of this doc said it stayed at 1.)*
- **Eating:** eating any fruit resets the level to 50. The reset doesn't add to the remaining level, so eating before HUNGER = 1 wastes the rest.
- **The cost:** `reward = 0`, so hunger costs no reward. Its cost is through stamina: `StaminaModulatedByNeed` removes 1 stamina every step while HUNGER ≥ 1.

### 2.6 Stamina (`shared_components.lua`)
- **Range and bands:** `maxStamina = 18`. The bands are red (0), yellow (1–6), green (7–12) and invisible (13–18). `STAMINA` is reported as value/18.
- **Moving:** a move action (`move ≠ 0`; turning is free) costs 1 stamina every 2 move steps (`decrementRate = 0.5`). After a costly step the player is frozen for 0 frames in green, 2 in yellow and 6 in red.
- **Recovering:** any step without a move recovers 1 stamina, unless the player stands on a `PreventStaminaRecoveryHere` tile. In the config that component is part of `potential_tree`, which covers every ground tile. Whether it fires on empty potential-tree tiles, which have no layer, has to be checked empirically before implementation (§12, item 6).
- **Water:** standing on a river tile costs 1 stamina per step (`TraversalCost`), with no reward penalty.

### 2.7 Grappling (`avatar_grappling.lua`)
- **Hold:** fires a beam 2 cells ahead. A player it hits cannot move or grapple for 2 frames, and the holder cannot move for 2 frames.
- **Shove and pull:** these include `hold=1`, so they hold and act in the same step. Shove moves the held player one cell in the holder's facing direction; pull moves it one cell back, towards the holder. Either works only on a player this player is holding.
- **READY_TO_SHOOT:** 1 when the player may hold, which means it is not frozen by someone else's hold.
- **Effect:** grappling never changes inventory, hunger or reward directly. Its effects are on position and movement: it blocks a player from a tree or a trade partner, or drags it into a river.

### 2.8 Rewards and observations
- **Reward:** only eating is rewarded. Eating the player's **preferred** fruit (the one it doesn't specialise in) gives 8, and the other gives 1 (`Taste`). Hunger, water and movement give 0.
- **Per-player observations:**
  - `RGB` 88×88×3 (an 11 × 11 window: 9 cells ahead, 1 behind, 5 to each side);
  - `READY_TO_SHOOT`, `STAMINA` and `HUNGER` (floats in [0, 1]);
  - `INVENTORY` (2: apples, bananas);
  - `MY_OFFER` (2, the public offer);
  - `OFFERS` (102 = 51 × 2: the public offers of everyone within radius 4, in arbitrary order, padded with zeros).
- **Global observation:** `WORLD.RGB` (debug only).
- **Not exposed by default:** positions, orientation, tree states, other players' hunger and inventories.

### 2.9 Upstream versus meltingpot-pz

| | Upstream `dm-meltingpot` 2.4.1 | meltingpot-pz |
|---|---|---|
| What it is | The reference implementation: Python config plus Lua, run in the dmlab2d engine | A native PettingZoo reimplementation in Python. The docs site describes `mp.fruit_market.concentric_rivers` |
| Fidelity | It *is* the substrate published results use | Aims at "behavioural parity". No parity tests, list of differences, repository URL or licence are published, and there is no PyPI package |
| Install on this Mac | `dmlab2d` 1.0.2 has a cp311 macOS arm64 wheel. `dm-meltingpot` requires Python ≥ 3.11 (prenv is 3.11.16) | Clone, then `uv sync` |
| Spatial state | Not exposed by default (see below) | `render_mode=None` gives categorical grids, including in `observation_mode="global"`: `TREES`, `AVATARS`, `AVATAR_ORIENTATIONS`, `AVATAR_STATES`, `AVATAR_STAMINA`, `BEAMS`, `SELF` |
| Events | `trade` events with player indices and quantities | Not documented |
| Seeding | `builder(..., env_seed=…)` sets the Lua RNG (`builder.py:144–189`) | Not documented |

**Recommendation: upstream.** It is the substrate you will cite, and its Lua RNG and event stream are documented. The missing spatial state is cheap to add without forking:

1. **Positions and orientation:** `LocationObserver` is already a Melting Pot component (`component_library.lua:806`). It is only behind the `_ENABLE_DEBUG_OBSERVATIONS` flag. Our builder appends it to each avatar's component list with `objectIsAvatar=True, alsoReportOrientation=True`, and adds `POSITION` and `ORIENTATION` to the observation names. This is Python config only.
2. **Tree states:** add one small Lua component on the scene object. `TreeStateObserver` emits `WORLD.TREES` (31 × 31 int: 0 none, 1 ripe apple, 2 ripe banana, 3 unripe apple, 4 unripe banana) by querying every `potential_tree`'s `FruitType`. It is loaded from our own level directory, whose `init.lua` requires the stock `trade` components and then ours. The builder already supports overriding `levelDirectory`. About 40 lines of Lua, in `src/fruit_market/lua/`.
3. **Grappling received:** the same component file can expose each avatar's `Grappling._heldBy` as `HELD_BY`, so "grapples received" can be recorded exactly.

Alternatives considered:
- **Parse `WORLD.RGB` into a categorical grid.** Tree sprites have distinct palettes for ripe and unripe of each fruit, so this works without Lua. But an avatar hides the tree it stands on, the water is animated, and Melting Pot warns the debug render is slow. Possible, but fragile.
- **Track trees in Python.** Positions would come from the first frame, and ripeness would follow harvests: a harvest is an inventory increase of 2 with no `trade` event, by a player standing on the tree, and the tree regrows 50 steps later. It needs no Lua, but it duplicates engine logic and its timing could be off by one. Useful only as a cross-check in tests.
- **Engine state.** dmlab2d doesn't give Python general access to game-object state, so an observer component is the supported route.

**Fallback:** `FruitMarketBackend` (§13, step 7) gives everything above to the rest of the code as a plain `FruitMarketState` snapshot. A meltingpot-pz backend could be added later if you want pure Python. Its results would need a parity check against upstream first: same seeds, same action sequences, compare inventories and trades.

**Dependency caution:**
- `pip install dm-meltingpot` lists **unpinned `tensorflow`**, so it could upgrade prenv's TF 2.16.2 / Keras 3.15.1 and endanger loading existing checkpoints. It also lists `pygame-ce`, which installs the same `pygame` module as your `pygame` 2.6.1 used by `render_pygame.py`.
- The substrate code path imports only `dmlab2d`, `dm_env`, `dm-tree`, `immutabledict`, `ml_collections`, `reactivex`, `absl` and **`chex`**. `chex` pulls in `jax`.
- **Dry run, 13 Nov 2026.**
  - *Without pins:* `pip install --dry-run` of those packages would install `numpy-2.4.6`, `ml_dtypes-0.6.0`, `jax-0.10.2` and `jaxlib-0.10.2`. That upgrades numpy from 1.26.4, which TensorFlow 2.16.2 needs, so **it must not be run unpinned**.
  - *With pins:* constraining prenv's current `numpy==1.26.4`, `ml-dtypes==0.3.2`, `tensorflow==2.16.2`, `keras==3.15.1`, `absl-py==2.5.0`, `protobuf==4.25.3` and `h5py==3.16.0` resolves to:
    - new packages only: `chex-0.1.90`, `dm-env-1.6`, `dm-tree-0.1.10`, `dmlab2d-1.0.2`, `immutabledict-4.3.1`, `jax-0.4.34`, `jaxlib-0.4.34`, `ml_collections-1.1.0`, `reactivex-6.1.0`, `toolz-1.2.0`;
    - then `dm-meltingpot-2.4.1` with `--no-deps`.

    Nothing already installed changes.
- **Install (decided):**

  ```
  pip install -c constraints-prenv.txt dmlab2d dm-env dm-tree immutabledict ml-collections reactivex absl-py chex
  pip install --no-deps dm-meltingpot==2.4.1
  ```

  Both files are committed: `requirements-fruit-market.txt` (the exact versions above) and `constraints-prenv.txt`. Afterwards, run the full test suite and the reference-output check, to confirm existing checkpoints still load and the hashes still match.

---

## 3. Reuse analysis and refactors

### 3.1 Module by module

| Code | Verdict | Notes |
|---|---|---|
| `DQN`, `NNetwork` | **Reuse; one small generalisation** | Network sizes come from `n_features` and `len(actions)`. A static action mask is needed: in `choose_action` (exploration and greedy) and in `train`'s `max` over next-state Q (§5). With no mask the code path is identical, so Harvest's random draws don't move. |
| `DQNDecisionModule` | **Reuse unchanged** (passes a mask through) | Checkpoint paths, epsilon decay, `progress.json` and continuing all work as they are. |
| `EthicsModule` | **Reuse unchanged** | Takes a well-being vector. Principles and aggregations need nothing Fruit Market-specific. Rounding to 6 decimals is harmless. |
| `NormsModule` (rule-based) | **Reuse unchanged** | Antecedent specs are registered. The consequent is the norm label from §9: pass the label as the "action" with an empty `consequent_rules` list. |
| `BayesianNormsModule` | **Reuse unchanged** | Feature specs index the observation vector, and `update(observation, norm_action_idx)` already takes an index. The environment supplies the per-step norm-action index instead of a static `norm_action_index` table. |
| `MovingModule` | **Not used** | Harvest's A* walk to the nearest berry is a macro action. Fruit Market agents move with primitive actions. Its distance helpers aren't reusable either: Fruit Market needs egocentric directions. |
| `HarvestAgent.step` / `perform_transition` | **Generalise** (R2) | The order "antecedent → ethics before → act → next state → sanction → attributes → norms → learn" is generic. Acting and attribute updates are Harvest's. Under simultaneous stepping, acting splits into *choose* (before the substrate step) and *settle* (after). |
| `HarvestAgent.observe`, `_calculate_n_features` | **Move to environment** (R3) | Become `model.observe(agent)` and `model.observation_size(agent)`. Harvest's version returns exactly the current vector. |
| `HarvestAgent` reward tables | **Not used by Fruit Market** (§6) | They stay as they are for Harvest. Fruit Market agents learn from the substrate's reward, plus the sanction for ethics types. |
| `HarvestModel` seeding, reports, norm bookkeeping, means, Gini | **Extract into a base class** (R1) | `_set_seed`, `reseed_for_continuing`, `_start_report`, `_continue_report`, `_report_row`, `_check_emerged_norms`, `_update_episode_emerged_norm`, `_append_norm_dict_to_file`, `_analyse_agent_norms`, `_num_agent_norms`, `_mean_*` and the Gini helper are all environment-free. |
| `HarvestModel` grid, berries, seasons, allocations | **Harvest only** | Not touched. |
| `_analyse_agent_norms` | **Small generalisation** | Hard-codes `b["action"] == "throw"` as cooperative. This becomes the environment's cooperative and uncooperative label sets, with Harvest's default `{"throw"}` (R4). |
| `norm_processing._get_cooperative_norms` | **Small generalisation** | Hard-codes `"throw" in norm_name`. Same parameter as above (R4). |
| `run_simulation` | **Reuse unchanged** | Duck-typed on the model's interface (§1). Fruit Market sets `render = False`. |
| `checkpoints.py` | **Reuse; add one naming branch** | `training_checkpoint_name` gets a `fruit_market` branch (§10). `training_checkpoint_path`, `prepare_training_checkpoints`, `finished_episode` and `record_progress` all work as they are. |
| `DataAnalysis` and graphs | **Not reused yet** | Built around Harvest's columns. Fruit Market graphs are a later piece of work. |
| `RenderPygame` | **Not used** | Fruit Market's `WORLD.RGB` frames could be saved instead (optional). |

### 3.2 Proposed refactors
Each is one commit, verified with pytest and `tests/reference_outputs.py`: all 950 hashes must match.

**R1. `SocietyModel` base class** (new `src/society_model.py`). `HarvestModel(SocietyModel)` keeps every public name.
- *What:* move the environment-free members listed above into the base, unchanged.
- *Why:* `FruitMarketModel` needs the same seeding, report continuation, emerged-norm bookkeeping and means. Otherwise it would copy about 200 lines.
- *Risk to existing results:* **very low**. Pure moves: no random draws, file names or orders change. `reseed_for_continuing` stays overridable, since commons overrides it.

**R2. Generic agent core** (`LearningAgent`, with `HarvestAgent(LearningAgent)`).
- *What:* move into `LearningAgent`:
  - decision, ethics and norms module construction;
  - `_ethics_sanction`, `finish_episode`, `get_epsilon`, `get_mean_loss`;
  - the norms update.
  `HarvestAgent` keeps health and berries, `_perform_action` and its reward tables. Hooks: `norm_antecedent_values()`, `norm_label(action)` / `norm_action_index(action)`.
- *Why:* `FruitMarketAgent` reuses the ethics, norms and learning flow without Harvest's body.
- *Risk:* **low**, as long as the order of calls inside `perform_transition` is unchanged. The hashes check that, because a changed order changes rewards.

**R3. Environment-provided observations and actions.**
- *What:*
  - `HarvestModel.observe(agent)` and `observation_size(agent)` hold the code now in `HarvestAgent.observe` and `_calculate_n_features`. The agent's methods delegate to them.
  - `model.agent_actions(agent)` returns `["move", "eat", "throw_X"…]` (now `_generate_actions`).
  - `model.action_mask(agent)` returns `None`.
- *Why:* the brief asks for the DQN's action set and observation to come from the environment.
- *Risk:* **very low**. The same `np.append` sequence gives the same float64 vector, so the network input is unchanged.

**R4. Cooperative norm label sets.**
- *What:* `COOPERATIVE_NORM_ACTIONS` and `UNCOOPERATIVE_NORM_ACTIONS` on the model (Harvest: `{"throw"}`), used by `_analyse_agent_norms`; and an optional `cooperative_labels=("throw",)` argument for `NormProcessing`.
- *Why:* Fruit Market's cooperative and uncooperative norms are different (§9.4).
- *Risk:* **none**. The defaults reproduce the current test exactly.

**R5. Action mask in `DQN`** (`mask=None` default).
- *What:* §5.2.
- *Why:* grappling off.
- *Risk:* **none** with no mask. The mask branch is only taken when a mask is given, so Harvest makes the same `np.random` calls in the same order.

**Not proposed** (considered and rejected):
- **A full abstract `Environment` interface that `HarvestModel` formally implements.** `run_simulation` already relies only on duck typing, and the shared pieces fit naturally in R1–R3. A formal ABC would mean touching every scenario constructor, for documentation value only. I'd describe the interface in a docstring on `SocietyModel` instead (§3.3).
- **A generic runner for sequential and simultaneous stepping.** Each model's `step()` already owns its stepping. Mesa's `RandomActivation` stays in Harvest, and `FruitMarketModel.step()` does choose-all → substrate step → settle-all. `run_simulation` drives both unchanged.
- **Moving Harvest's reward tables into the model.** They are per agent type, and nothing outside the agent uses them. Fruit Market uses the substrate's reward instead.

### 3.3 The interface both models provide (documented on `SocietyModel`)

```
# run_simulation contract (existing)
step(); finish_episode(collect_data); training; epsilon; min_epsilon; episode; max_episodes; _mean_reward()
# agent-facing environment contract (R3 + Fruit Market)
observe(agent) -> np.ndarray            observation_size(agent) -> int
agent_actions(agent) -> list[str]       action_mask(agent) -> np.ndarray[bool] | None
get_society_well_being(observer, norms_observation, ethics_observation) -> np.ndarray
norm_antecedent_specs(agent) / bayesian_feature_specs(agent)    # feature-spec mechanism (§9.3)
norm_label(agent, action) -> str        norm_actions() -> list[str]   # §9
COOPERATIVE_NORM_ACTIONS / UNCOOPERATIVE_NORM_ACTIONS             # R4
get_day(); get_max_days(); get_num_agents(); seed / reseed_for_continuing()
```

---

## 4. Scalar observations

All features are computed in Python from the backend's `FruitMarketState`. The state holds:
- per player: position, orientation, role, inventory, hunger, stamina, ready-to-shoot, public offer and held-by;
- `WORLD.TREES`;
- this step's trade events;
- the river mask, read from the ASCII map once.

Directions are egocentric: (forward, right) components of the offset, rotated by the player's orientation, so they match what a FORWARD or STEP_RIGHT action does. That makes orientation largely redundant. It is kept as a one-hot so the network can tell which way a turn will take it.

"Own fruit" means the player's specialty, the fruit it harvests reliably. Holdings are given **role-relatively** (own fruit, other fruit) rather than as apples and bananas. Both roles then see the same meaning in the same input, and agents of one type can share a replay buffer across roles, as Harvest agents of one type do now. Role is still given as a flag.

| # | Feature | How it's computed | Privileged? | Normalisation |
|---|---|---|---|---|
| 1 | own hunger | `HUNGER` | no | clipped to [0, 1] |
| 1a | time hungry | `(HUNGER − 1)/0.02` steps once hungry (§7.3) | no | ÷ 500, clipped to 1 |
| 2 | own stamina | `STAMINA` | no | already [0, 1] |
| 3 | own fruit held | `INVENTORY[specialty]` | no | ÷ 10, clipped to 1 |
| 4 | other fruit held | `INVENTORY[other]` | no | ÷ 10, clipped to 1 |
| 5 | role | 1 apple farmer, 0 banana farmer | no (own role) | 0/1 |
| 6 | ready to shoot | `READY_TO_SHOOT` | no | 0/1 |
| 7–10 | orientation | one-hot N/E/S/W from `ORIENTATION` | no (own body) | 0/1 |
| 11–12 | own offer | public `MY_OFFER`, role-relative: (own fruit, other fruit) | no | ÷ 3 → [−1, 1] |
| 13–15 | nearest ripe own-fruit tree | Manhattan distance (path length ignoring other players; rivers are walkable), and egocentric (forward, right) of the offset ÷ distance | **yes** beyond the 11 × 11 window | distance ÷ 60 (map diameter); direction ∈ [−1, 1]; none ripe → (1, 0, 0) |
| 16–18 | nearest ripe other-fruit tree | as above | **yes** beyond the window | as above |
| 19–21 | nearest compatible partner | the nearest player whose public offer is compatible with mine both ways (the substrate's `isCompatible` and `hasEnough`, applied in Python); if I have no public offer, the nearest whose offer I could fill from my inventory | **yes**: positions, and offers beyond radius 4 | as 13–15 |
| 22 | compatible offers within radius 4 | count of entries in `OFFERS` compatible with my offer | no (derived from `OFFERS` and `MY_OFFER`) | ÷ 15 |
| 23 | on river | river mask at my position | no | 0/1 |
| 24 | river ahead | river mask at the cell in front | no | 0/1 |
| 25 | held | `HELD_BY ≠ 0` (being grappled this step) | no (the player feels it) | 0/1 |
| 26 | hungry players nearby | count of players within radius 4 with HUNGER = 1 (used by norms, §9.3) | **yes** | ÷ 15 |
| 27… | others' well-being | §7, one value per other player, in player-id order (as in Harvest) | **yes** | ÷ (50 × (1 + 10)), clipped to [−1, 1]. In `need` mode, 550 steps covers a full stomach plus 10 fruit; it is negative only when a player is hungry with no fruit (§7.3). |

That is 27 + (N − 1) inputs: 42 for 16 players, 34 for 8. Feature 1a was added during implementation: HUNGER isn't capped (§2.5), so clipping it to 1 alone would hide how long a player has been hungry. In code the features are `features.FEATURE_NAMES`, in this order.

Notes:
- **Harvest doesn't normalise its inputs** (health, berries, days left and distance are raw). Fruit Market's ranges are wider (distances up to 60, well-being up to hundreds), so I propose the normalisation above. It doesn't affect Harvest.
- **The partner feature (19–21) covers the case of having no offer.** Without that case, a player with no offer would have no signal for whom to trade with.
- **Ablation option (§10): restrict privileged features to the view.** Features 13–21 would be computed only within the player's 11 × 11 window, the information `RGB` gives an upstream agent. Others' well-being stays, since Harvest's ethics observation is equally privileged.
- **Dropped:** distance to the nearest *unripe* tree, time until regrowth, and the number of players nearby. They add inputs with little decision value. Easy to add later.

---

## 5. Actions for decision-making

### 5.0 What stays the same as the original substrate
**Unchanged:**
- **Actions:** the 25 actions, their order and their components. The builder passes `fruit_market.ACTION_SET` to Melting Pot untouched, and an agent's chosen index goes to the substrate as is.
- **The substrate itself:** the map, roles (8 apple and 8 banana farmers), player count (16), episode length (1000), every Lua game component and its parameters, and the substrate's reward.
- **Randomness:** the substrate's random draws come only from the Lua RNG, seeded by `env_seed`.

**Added to the substrate, read-only:**
- `LocationObserver` on each avatar;
- our `TreeStateObserver`, and `HELD_BY`.

These only define observations: no updaters, no state changes, no random draws. **Check:** an integration test runs the substrate twice with the same `env_seed` and the same random action sequence, once as built by Melting Pot and once with our observers. Every step's rewards, inventories, hunger, stamina, `MY_OFFER` and `trade` events must be identical.

**Changed for agents, and nothing else:**
1. Ethics agents learn from the substrate's reward **plus** the ethics sanction (§6).
2. The norms modules record each step's action as a label (§9). Labels are computed after the step, from what happened; they never affect which action is chosen or what it does.

Agents see a scalar feature vector (§4) instead of `RGB`. That changes what the agents perceive, not the substrate.

**The one option that departs from the original:** turning grappling off (§5.2), from your brief. It is **on by default**, which is the original substrate. Turned off, actions 9–11 are never chosen, but the substrate still has them and they keep their indices.

### 5.1 Action set
`FruitMarketModel.agent_actions(agent)` returns the 25 names in §2.2, built from the substrate's own `ACTION_SET` at import time. That checks it has 25 entries in the expected order: if upstream reorders them, a test fails rather than silently relabelling. Agents choose indices 0–24, which go straight to the substrate. There is no macro action and no throw equivalent.

Failed but legal actions are **not** masked: eating with no fruit, an unaffordable offer, holding with no-one in range. The substrate allows them and gives them 0 reward, as the original does.

### 5.2 Disabling grappling (prompt: "Do you want grappling (hold, shove, pull)? (y, n)")
- **Mask:** a static boolean mask of length 25, with indices 9, 10 and 11 false. It's the same for every agent and every step.
- **Exploration:** `np.random.choice` over the allowed actions. Masked actions are never chosen and so never sent to the substrate. If the backend receives one anyway, it is an error.
- **Greedy:** set Q-values of masked actions to `−inf` before `argmax`.
- **Learning target:** `value_next = max` over the **allowed** actions only. Otherwise the target would bootstrap from untrained outputs for actions that are never taken, inflating every target. The mask is static, so it is held by the `DQN`; the replay buffer doesn't need a per-experience mask.
- **Masked outputs get no gradient.** They are never in the replay buffer, so their outputs are never trained.

**The network output stays 25 in both settings.**
- *For:*
  - checkpoints have the same shape;
  - the grappling-off network can be tested with grappling on (the ablation of "what if grappling became available");
  - one `CheckpointMismatchException` rule covers both.
- *Caveat:* a grappling-off network's hold, shove and pull outputs are untrained, so if tested with grappling on it picks them more or less at random. That ablation should be read as "untrained grappling", not "learned grappling". Grappling is part of the checkpoint name (§10), so the two runs never share a folder by accident.

---

## 6. Rewards

### 6.1 Reward (decided)
- **Baseline agents:** r = the substrate's reward. That is 8 for eating the preferred fruit, 1 for eating the other fruit, and 0 for everything else, including harvesting, trading, grappling, failed actions, hunger and water.
- **Ethics agents:** r = the substrate's reward + s. The sanction s is −S, 0 or +S, where **S = 3.5** (decided). This is the shaping term F in r′ = r + F.
  - S is 0.4375 × the substrate's largest reward (8, `mostTastyReward`).
  - In Harvest, the sanction is 0.4 × the largest baseline reward (`sanction` 0.4 / `eat` 1).
  - S is a constant (`FRUIT_MARKET_SANCTION = 3.5`), not computed from the config.

None of Harvest's reward table is used. There is no `forage`, `no_berries` or `survive`, and no reduced `eat` for ethics agents. Harvest does lower ethics agents' `eat` from 1 to 0.8; here the substrate's rewards stay as they are for every agent type, as you asked.

Both parts are recorded each step: `env_reward` (the substrate's) and `ethics_sanction`. So baseline and ethics agents can be compared on the substrate's own reward, and with published Melting Pot results.

**Scale.** A player eats at most about once per hunger cycle (50 steps) without wasting food. Eating gives up to about 20 × 8 = 160 per episode. A sanction of ±3.5 that was non-zero on most of the 1000 steps would outweigh that. That's one reason for the attributed mode below: with it, the sanction is non-zero only on steps where the agent's own action changes someone's well-being, which means harvests, eats and trades.

### 6.2 When the sanction applies
For ethics agents, every step, as in Harvest. EthicsModule compares society's well-being vector before and after, so what matters is which two vectors it compares (§6.3).

### 6.3 Attributed versus joint sanction (decided: attributed)

**Why Harvest doesn't have this problem.** Mesa steps Harvest's agents one at a time. Each agent:
1. records society's well-being;
2. acts;
3. records it again;
4. is sanctioned for the difference;
5. and only then has its health decay (`perform_transition`, then `_update_attributes`).

So the before and after vectors differ only by that agent's own action.

**What's different in Fruit Market.** The substrate steps all 16 players at once. Between the observation at step *t* and the one at step *t*+1:
- **Everyone's** actions have taken effect.
- **Time has passed for everyone.** Each player's hunger rose by 0.02, so `need` well-being fell by 1, or by 0.1 for a player who is already hungry (§7.3).

**The three ways to build the two vectors.** For agent *i* at step *t*, let `before` be every player's well-being at *t* and `after` every player's at *t*+1. Let `passive(before)` be `before` advanced by one step with no actions: hunger +0.02 (so one more hungry step for those already hungry), and inventories unchanged.

| Mode | Compares | So the sanction reflects |
|---|---|---|
| **Joint** (raw) | `before` with `after` | everyone's actions, plus time passing |
| **Joint, time-corrected** | `passive(before)` with `after` | everyone's actions, not time passing |
| **Attributed** | `passive(before)` with `passive(before)` where *i*'s own entry, and its trade partner's if it traded this step, are replaced by their actual `after` values | only *i*'s own action, and its trade |

**Attribution is exact, not an approximation.** A player's well-being depends only on its own hunger, inventory and last fruit eaten. Only three things can change those:
- the player's own eating;
- the player's own harvesting;
- a trade the player was part of.

Another player's move, eat, offer or grapple can't change it directly. So replacing just the agent's own entry, and its partner's, captures everything its action did to well-being in that step.

**Worked example** (`need` mode, 4 players, utilitarian unless stated). At step *t* the well-being is `[120, 60, 20, −3]`. Player 3 has been hungry for 30 steps (0 + 0 − 0.1 × 30). During the step, player 0 walks, player 1 harvests (+100) and players 2 and 3 wait.

- `passive(before)` = `[119, 59, 19, −3.1]`.
- Actual `after` = `[119, 159, 19, −3.1]`.

| Agent's view | Joint | Joint, time-corrected | Attributed |
|---|---|---|---|
| Player 0 (walked) | sum 197 → 293.9: **+3.5**, rewarded for player 1's harvest | 193.9 → 293.9: **+3.5**, same | own entry unchanged: **0** |
| Player 0, if player 1 had *not* harvested | 197 → 193.9: **−3.5**, punished for time passing | 193.9 → 193.9: **0** | **0** |
| Player 1 (harvested) | **+3.5** | **+3.5** | 193.9 → 293.9: **+3.5** |
| Any **maximin** agent, if nobody helps player 3 | min −3 → −3.1: **−3.5 every step** while anyone is hungry | **0** | **0** |
| A maximin agent completes a 2-for-1 trade in favour of the worst-off player | depends on what everyone else did | the minimum rises: **+3.5**, if nobody else lowered it | its own entry −50, the partner's +50, the minimum rises: **+3.5** to both traders |

**What each mode means for learning:**
- **Joint:** every agent's sanction is mostly noise from the other 15 players and from the clock. A maximin agent is punished every step that anyone is hungry, whatever it does. A utilitarian agent is punished every quiet step. With S = 3.5 against an eating reward of at most 8, about every 50 steps, this would swamp the substrate's reward.
- **Joint, time-corrected:** removes the clock, but agents are still rewarded or punished for other players' actions in the same step. Two agents doing the same thing in the same step can get opposite signals.
- **Attributed:** the sanction depends only on what the agent did, as in Harvest. Its limitation is that **indirect** effects are never sanctioned:
  - grappling someone away from a tree;
  - standing on the tree someone was walking to;
  - a trade offer that makes a later trade possible.

  These change well-being only later, through other players' actions. But neither joint mode credits them either: within one step they can't be told apart from what the other players chose.

**Decided: attributed** is the default. Joint is a prompt option, so its effect can be measured as an ablation.

The same three modes apply to `ethics_observation` only. The well-being features in agents' observations (§4) are always the actual current values.

---

## 7. Well-being

Confirmed from the config:
- **How hunger rises:** linearly, by `hunger_rate = 1/delay = 0.02` per step (`PeriodicNeed`).
- **`H_limit = 1`:** HUNGER is reported as a fraction, and costs start when it reaches 1.
- **The cost:** 1 stamina per step at HUNGER ≥ 1. There is no reward cost.
- **Eating resets HUNGER to 0.** The reset doesn't add to what was left.

So `H_limit / hunger_rate = 50`.

### 7.1 `need`
    w_i = (1 − hunger_i)/0.02 + (apples_i + bananas_i) × 50 − 0.1 × hungry_steps_i

The last term is §7.3. Without it, this is the brief's formula unchanged: "steps until hunger costs begin, if the player eats each fruit just as it gets hungry". It is the direct analogue of Harvest's days left to live: resources plus current state, converted to time.
- **Eating at HUNGER = 1** changes w by +50 − 50 = 0, plus the hungry steps it ends (§7.3).
- **Eating earlier** changes w by −(1 − hunger) × 50 < 0: the rest of the current cycle is wasted. Utilitarian and maximin agents are therefore sanctioned for eating early. That is a consequence of the formula, not a bug: early eating really does waste a resource.
- **Harvesting** adds 100, trading changes holdings by the traded counts, and grappling changes nothing directly.

### 7.2 `preference` (corrected; decided)
The brief's form weights each fruit by its value to the holder relative to their less-preferred fruit:

    w_i = (1 − hunger_i)/0.02 + 50 × Σ_f n_f × (v_f / v_min)

For an apple farmer, a banana counts 8 and an apple 1.

**Problem:** eating a banana at HUNGER = 1 changes w by +50 − 400 = −350. Eating the fruit a player values most would then be the worst thing it can do for its own well-being, and utilitarian agents would learn never to eat it.

**Correction:** weight the current hunger cycle by the value of the fruit that started it:

    w_i = (1 − hunger_i)/0.02 × (v_last,i / v_min) + 50 × Σ_f n_f × (v_f / v_min) − 0.1 × hungry_steps_i

Here `v_last,i` is the value to *i* of the fruit it last ate (`v_min` at the start of an episode). The last term is §7.3, unweighted.
- Eating any fruit at HUNGER = 1 is again neutral.
- Holding fruit the player prefers counts more.
- A trade that moves fruit to the player who prefers it raises the total, which is what a preference-based welfare measure should do.
- The value comes from `Taste` (8 or 1), read from the config.

Two cautions:
- **Scale.** Preference well-being is up to 8 times larger than need well-being. The ethics principles only compare before and after, so this doesn't matter to them, but the observation normalisation (feature 27 onwards) uses a scale of 50 × 8 × 11 in this mode.
- **Privileged information.** Like Harvest's well-being, the vector for all players is privileged, because other players' hunger and inventories are not observable in Fruit Market. The model provides it through `get_society_well_being(observer, norms_observation, ethics_observation)` with Harvest's signature. Nobody dies, so the dead-agent branches never apply.

### 7.3 Time spent hungry (decided)
`hungry_steps_i` counts the steps since player *i*'s HUNGER reached 1: the steps on which the substrate takes 1 stamina for hunger. Since HUNGER keeps rising past 1 (§2.5), it is read straight from it: `(HUNGER − 1) / 0.02`. It goes back to 0 when *i* eats. Each hungry step lowers well-being by **0.1**, a tenth of the rate before hunger (1 per step). So well-being keeps falling while a player stays hungry, but slowly.

Why 0.1:
- A player hungry for a whole episode (at most 950 steps) loses about 95, a little under two fruit's worth (2 × 50). So hunger matters, but holding fruit still matters more.
- The rate is one constant (`HUNGRY_WELL_BEING_DECAY`), easy to change.

What it changes:
- Well-being can now go **below 0**. A hungry player with no fruit goes 0, −0.1, −0.2, …. That's fine for all the principles, which only compare values.
- **Eating when hungry is now positive.** It changes w by +0.1 × hungry_steps: ending a long hunger raises well-being more. In Harvest, eating is neutral and death is the floor; here, relieving hunger is the good outcome, and the longer the hunger, the more it's worth.
- **Helping a hungry player is worth more** to egalitarian and maximin agents, the longer that player has been hungry, because the hungry player is further below everyone else.
- **Passive dynamics** (§6.3) include it: `passive(before)` adds one hungry step to every player already at HUNGER = 1.

`hungry_steps` comes from each step's HUNGER, so nothing is added to the substrate. It is reported per agent (§8.1).

---

## 8. Tracking

Two new files, so the existing formats don't change:
- `data/results/current_run/fm_agent_reports_<filepath>.csv`: one row per agent per step, **test runs only**, as in Harvest.
- `fm_model_episode_reports_<filepath>.csv`: one row per episode, in training and testing.

Report continuation for continued training reuses R1's `_start_report` and `_continue_report`.

Writing is buffered: an episode's rows are written once, when it ends. Harvest's `pd.concat` per agent per step would cost about 16,000 concatenations per episode here.

### 8.1 Agent level (per step)

| Column | Why |
|---|---|
| episode, step, agent_id, agent_type, role | Keys. Role is the main covariate. |
| action (index), action_name, norm_label | What was chosen, and how the norms modules saw it (§9). |
| env_reward, ethics_sanction, reward | The substrate's reward, the sanction (blank for baseline agents, as in Harvest), and what the agent learned from (their sum). |
| hunger, hungry_steps, stamina, own_fruit, other_fruit, apples, bananas | State. Both role-relative and absolute holdings, for different analyses. |
| well_being, well_being_other_mode | The mode in use, and the other mode for comparison at no extra cost. |
| ate_own, ate_other | Fruit eaten this step: autarky and Gini. |
| harvested | Fruit harvested this step: specialisation. |
| trade_partner, fruit_given, fruit_received | From `trade` events. Cross-role analysis and gifts in disguise, i.e. generous offers that were filled. |
| offer_posted, offer_cancelled, offer_invalid, offer_public | Offer behaviour, including "offer spam" and offers hidden by the inventory check. |
| grapple_action (hold/shove/pull or blank), grappled_by | Grapples performed, and received (from `HELD_BY`). |
| position_x, position_y, on_river | Cheap. Lets you draw maps and rule out "stayed by the river" artefacts. |
| move_effects | Bit flags: reduced distance to own tree, other tree, partner (§9.2). Lets norms be relabelled offline. |
| num_norms | As in Harvest. |

Dropped from the brief: none. Added: `role`-relative holdings, `harvested`, `offer_public`, position and `move_effects`.

### 8.2 Model level (per episode)

| Column | Why |
|---|---|
| episode, end_step, epsilon, mean_reward, mean_loss, seed | As in Harvest. |
| mean_env_reward, mean_sanction | The substrate's reward, comparable with Melting Pot and across agent types; and the sanction, separately. `mean_reward` is their sum. |
| fruit eaten: total, min, max, Gini | Distribution. Gini uses R1's helper. |
| reward: min, max, Gini; well-being: mean, min, max, Gini (end of episode and mean over steps) | Equality outcomes for the principles. |
| total_trades, cross_role_trades, mean_trade_rate (fruit given / received) | Market activity and its terms. |
| autarky_rate | Share of fruit eaten that was the eater's own specialty. 1 means no trade benefit. |
| preferred_eat_rate | Share of eats that were the preferred fruit. Arguably the substrate's real welfare measure. |
| invalid_offer_rate | Offer steps where the offer was hidden by the inventory check. |
| market_freeze_steps | Steps where some pair of players has compatible public offers but no trade happens. Possible here because each player only accepts non-dominated offers, and because of radius 4 at resolution time. |
| total_grapples (hold, shove, pull separately) | Aggression. |
| mean_hunger, steps_hungry (total over players), mean_stamina | Need outcomes. Stand in for Harvest's `deceased`. |
| num_emerged_norms; plus the existing `agent_norms_<filepath>.csv` for Bayesian norms | As in Harvest. |

---

## 9. Norms: macro actions as labels

### 9.1 Labels
The label is computed after the substrate step, from the action and the state change. Agents never choose labels.

| Action | Label |
|---|---|
| 0 NOOP | `harvest own fruit` or `harvest other fruit` if this step harvested; otherwise `wait at other tree` if standing on a ripe tree of the other fruit; otherwise `wait` |
| 1–6 move or turn | `harvest own fruit` / `harvest other fruit` if this step harvested; otherwise by effect (§9.2) |
| 7, 8 eat | `eat own fruit` / `eat other fruit` (own = specialty). Labelled by intent, as Harvest's "eat" is, even if there was no fruit; whether it succeeded is in the report. |
| 9, 10, 11 | `hold`, `shove`, `pull` |
| 12 | `cancel offer` |
| 13–24 | `offer generous`, `offer fair` or `offer demanding` (§2.2 table) |

I added three labels to the brief's: `harvest own fruit`, `harvest other fruit` and `wait at other tree`. Harvesting is what movement is for, so "IF hungry THEN harvest own fruit" is the norm of interest. Waiting on an other-fruit tree is the only way to harvest it, which is the autarky strategy.

### 9.2 Movement and turning by effect
Distances are path distances (Manhattan; rivers are walkable) to:
- the nearest ripe tree of own fruit;
- the nearest ripe tree of the other fruit;
- the nearest compatible partner (§4, feature 19).

Each is measured before and after the step. For **turns**, the position doesn't change, so "reduces" means the turn reduces the angle between the player's facing and the direction to the target.

Effects: `P` (closer to partner), `O` (closer to own fruit), `X` (closer to other fruit).

Ties are broken by fixed priority: **`approach partner` > `go to own fruit` > `go to other fruit` > `move`**.
- The partner comes first because trade-seeking is rarer and the most norm-relevant.
- Own before other follows the substrate's specialisation.
- A step that reduces none of them, including a blocked move, is `move`.
- `approach partner` counts only when the player has a public offer, or could fill a nearby one. Otherwise wandering towards a player would count as trade-seeking.

All three flags are written to `move_effects` (§8.1), so other tie-breaks can be applied to the reports without rerunning.

### 9.3 Feature specs (through the existing mechanism)

Rule-based antecedents use `boundaries` / `labels` / `zero_label`. The environment supplies them, as `extra_observation_feature_specs` does now.

| Feature | Boundaries | Labels |
|---|---|---|
| hunger | [0.5, 1.0] | sated / peckish / hungry |
| stamina | [0.34, 0.67] | low / medium / high stamina |
| own fruit held | zero_label; [2, 5] | no / few / some / many own fruit |
| other fruit held | zero_label; [2, 5] | no / few / some / many other fruit |
| offer posted | [0.5] (on abs(own offer)) | no offer / offer posted |
| partner nearby | [1] (feature 22 × 15) | no partner nearby / partner nearby |
| ripe own tree | [3/60, 10/60] (normalised distance) | own fruit near / medium / far |
| hungry player nearby | [1] | none hungry nearby / hungry player nearby |

The last row is a derived feature (§4, feature 26): the count of players within radius 4 with HUNGER = 1. It replaces Harvest's per-agent `repeated` well-being in antecedents. With 15 others, a repeated feature makes antecedents 15 buckets long. Rule-based norms (capped at 100) would then never repeat, and the Bayesian candidate set (pairs of predicates) grows with the square of the number of agents. "Hungry player nearby" is the reading the norms are about: whether to help a partner who needs it. It is privileged, so it is reported as such.

**Bayesian:** the same features as `{name, index, thresholds}` on the observation vector (§4 indices). "Hungry player nearby" is feature 26 of the observation (§4), so it has an index. `max_predicates = 2` as now. The norm actions are the 17 labels from §9.1 (`labels.NORM_LABELS`).

### 9.4 Cooperative and uncooperative norms
- **Cooperative actions:** `offer generous`, `offer fair`.
- **Aggressive actions:** `hold`, `shove`, `pull`.
- **Bayesian norms:**
  - *cooperative* = a learned commitment to a cooperative action, or a learned prohibition of an aggressive one;
  - *uncooperative* = a learned commitment to an aggressive action, or a learned prohibition of a cooperative one.
  This is a direct extension of Harvest's (commitment to throw / prohibition of throw).
- **Rule-based emerged norms:** classified by consequent in the same way.
- *Implementation:* `FruitMarketModel._analyse_agent_norms` uses these sets itself, so R4 (changing the shared code) wasn't needed.
- **Specifically interesting:** "IF hungry player nearby … THEN offer generous", "IF … own fruit many THEN offer fair", and "IF … THEN shove". These are what the norm tables should be checked for.
- `offer demanding` is neither cooperative nor uncooperative. It's reported as is.

---

## 10. Options, file names and checkpoints

The options are prompts. They are asked only when the scenario is `fruit_market`, after the existing scenario prompt. Defaults are in brackets.

1. **Steps per episode** (1000, the substrate's own; at most 1000). Replaces "How many days in each episode".
2. **Number of players** from [2, 4, 8, 16] (16, the substrate's own). Half are apple farmers. Replaces the agent-count prompt for this scenario, because the map has 21 spawn points.

   Pressing enter for prompts 1 and 2 gives the original substrate, which is the default for real runs (decided). Other values change the substrate, so they're for pilots and timing only. They appear in the checkpoint and output names, so such runs are never mistaken for full-scale ones.
3. **Grappling**, y/n (y, the original substrate).
4. **Well-being**: need / preference (need).
5. **Ethics sanction from**: attributed / joint (attributed, §6.3). Not asked for baseline agents.
6. **Spatial features**: global / view-limited (global).

When testing, the options above aren't asked: a test uses the settings the checkpoint was trained with, from its `settings.json`. It asks instead how many episodes to test (100, since a 1000-step episode takes minutes), then whether to write norms and which norms module to use, as the harvest scenarios do.
7. Society mix and agent type: the existing prompts, unchanged.
   - In a mixed society, baseline ids are chosen from the seed as now.
   - **Roles follow the substrate's own assignment**: the first half of the players are apple farmers, the rest banana farmers. *(Changed during implementation from id parity, to keep the original substrate.)* The baseline half is chosen at random from the seed, so it almost always has both roles.
   - Roles stay fixed per agent across episodes, by default; reshuffling them each episode is possible later, and the role flag is already in the observation.

Not asked for this scenario: seasons, allocations, render. Choosing `fruit_market` without Melting Pot installed prints how to install it (`requirements-fruit-market.txt`) and exits. The import is lazy, inside the `fruit_market` branch of `create_and_run_model`. Nothing imports `meltingpot` otherwise, so the existing code and tests run without it.

**Checkpoint name.** `training_checkpoint_name` gets a `fruit_market` branch, and every setting that changes what is learned is in the name, e.g.
`fruit_market_concentric_rivers_1000steps_grappling-off_wb-need_sanction-attributed_obs-global_seed42`. The sanction mode only affects ethics types, but it is in every type's name, so all types in one comparison share one folder.
Agents go in `…/<n>_agents/<type>/agent_<i>/`, using the existing path helpers, so continuing, `progress.json`, `settings.json` and the mixed-society folders all work unchanged.

**Output files.** `output_file_string` gives `<mode>_fruit_market[_grappling-off][_preference][_joint][_view][_mixed]_<type>`. Fruit Market's reports have the `fm_` prefix (§8). Emerged norms and Bayesian norm files keep their existing patterns, and the scenario name keeps them apart.

**Existing names are unchanged.** None of the existing branches change, so the 950 reference hashes are unaffected. The only visible change for other scenarios is that `fruit_market` appears in the scenario prompt's list. That's console output only, which isn't hashed, and the reference script answers by name.

---

## 11. Compute

I measured one agent-step (`choose_action` + `train`, batch 64, eager TensorFlow) on this Mac: **11.3 ms** at Harvest's size (7 inputs, 5 actions) and **11.5 ms** at Fruit Market's (39 inputs, 25 actions). Network size barely matters; the number of agent-steps does.

Training stops when epsilon reaches 0.01, after about 4,600 episodes. Estimated training time per agent type:

| Setup | Agent-steps per episode | Time per episode | Per agent type |
|---|---|---|---|
| Harvest (4 agents × 200 days) | 800 | ≈ 9 s | ≈ 12 h |
| **Fruit Market, 16 players × 1000 steps (decided default)** | 16,000 | ≈ 3 min | **≈ 10 days** (≈ 80 days for all 8 types, one after another) |
| Fruit Market, 8 × 1000 (pilot only) | 8,000 | ≈ 1.5 min | ≈ 5 days |

These are DQN time only; the substrate step and feature computation add to them, and need measuring in the pilot.

Ways to bring this down **without changing the substrate**:
- **Compile `DQN.train` and the forward pass with `tf.function`.** Typically several times faster. It could change float results in the last bits, so it would be on for Fruit Market only, never Harvest, which keeps its hashes.
- **Run several agent types at once.** Each type is a separate run, and the M1 Pro has 8 performance cores, while each eager TensorFlow run uses only some of them. Two to four runs side by side, in separate terminals with `run.py train`, would each slow down somewhat, but the total would finish sooner. The pilot can measure how much.
- **Batch the 16 agents' forward passes** in `choose_action` into one call per step. Each agent has its own network, so the gain is in the Python overhead only, and smaller.

The substrate isn't changed by any of these, and the algorithm isn't either. Learning every k-th step would change the algorithm, so I'd avoid it.

The pilot is 20 episodes at the default scale, timing each part, before committing to the main runs.

---

## 12. Testing and verification

1. **After every commit:** `pytest` and `python tests/reference_outputs.py`. All 950 hashes must match, and no existing test may change.
2. **Pure-function unit tests,** which need no Melting Pot. They use a hand-built `FruitMarketState` and cover:
   - the offer table (§2.2) and its fairness labels;
   - compatibility and resolution;
   - egocentric directions for all four orientations;
   - the movement labels and tie-breaks;
   - `need` and `preference` well-being, including eating at the threshold being neutral apart from the hungry steps it ends;
   - the hungry-steps counter and its decay;
   - attributed and joint sanction vectors, using the worked example in §6.3;
   - the reward: the substrate's reward unchanged, plus ±3.5 or 0 for ethics agents only;
   - mask behaviour (masked actions never chosen across 10,000 exploration draws; greedy skips them; the target uses the masked max);
   - observation length and order.
3. **Mask doesn't move Harvest's random draws.** A `DQN` with `mask=None` must make the same `np.random` calls as now. The reference hashes cover this.
4. **Integration tests** marked `pytest.importorskip("meltingpot")` (`tests/test_fruit_market_backend.py`, `tests/test_fruit_market_run.py`):
   - the installed substrate's actions, roles, episode length, hunger delay, taste rewards, trade radius, stamina, regrowth and map are the ones PriENE uses;
   - players start on spawn points, and trees start ripe on potential-tree tiles;
   - `trade` events agree with inventory changes;
   - `HELD_BY` reports holds;
   - **our observers don't change the substrate** (§5.0). The comparison of two runs planned here isn't possible (see the note below). Instead the tests check that our Lua components define nothing but observations (no updaters, no contact or hit handlers, no random draws, no state changes), that Melting Pot's `LocationObserver` likewise only reports, and that the settings differ from Melting Pot's only by the added observers and the level path;
   - `run.py train` then `run.py test` end to end.

   **Found during implementation: Melting Pot's substrate isn't reproducible from its seed.** Building `fruit_market__concentric_rivers` three times with the same `env_seed` gives three different starts (spawn positions, which trees are apple or banana), within one process and across processes. This holds for unmodified Melting Pot, without PriENE's observers. The seeded random number generators (Python's and Lua's) are deterministic, so the variation most likely comes from the order Lua iterates tables of game objects, which decides which object gets which random draw. PriENE's own randomness (network initialisation, exploration, replay sampling, the baseline half, each episode's `env_seed`) is still seeded. But Fruit Market episodes can't be replayed exactly, so results should be compared over many episodes.
5. **Fruit Market reference outputs:** not possible, because the substrate isn't reproducible (above). The existing `hashes.json` is untouched.
6. **Empirical checks before relying on them:**
   - whether stamina recovers on empty ground (§2.6);
   - the 25-action order against the installed version;
   - that resolving a trade in the step it is posted works as read from the updater priorities (offers 250, trades 2).

---

## 13. Implementation plan (one commit each, after approval)

1. R1 `SocietyModel` base.
2. R2 `LearningAgent` core.
3. R3 observations and actions from the model.
4. R4 cooperative label sets.
5. R5 action mask in `DQN`.
6. `requirements-fruit-market.txt` and `constraints-prenv.txt`. Install as in §2.9, with your approval first since it changes prenv. Then the full test suite and the reference-output check.
7. The backend (`src/fruit_market/backend.py`, Lua observer, `FruitMarketState`), with integration tests.
8. Pure functions (`src/fruit_market/features.py`, `labels.py`, `well_being.py`, `sanction.py`), with unit tests.
9. `FruitMarketAgent` and `FruitMarketModel` (simultaneous step, reports).
10. `run.py` prompts, checkpoint name and output names; tests in `test_modes.py` style.
11. Pilot run and timings, Fruit Market reference hashes, then a short results note.

Steps 1–5 change existing code and are checked against the hashes. Step 6 changes prenv and is checked the same way. Steps 7–11 add new code only.

### 13.1 Implementation status (13 Nov 2026)
Steps 1–3, 5–10 are done, one commit each, with all 950 reference hashes matching after every change to existing code. Step 4 (R4) wasn't needed (§9.4). Melting Pot is installed in prenv: `pip freeze` before and after differs only by added packages. The code is in `src/fruit_market/`, with its tests in `tests/test_fruit_market*.py`.

**Timing with the real substrate:** a step with 16 learning agents takes about 0.26 s, so a 1000-step episode takes about 4.5 minutes. That is about 14 days of training per agent type at the default scale. The DQN accounts for about 0.18 s of each step, and the rest is the substrate, features and bookkeeping. This replaces the 10 days estimated in §11.

**Step 11 (pilot and results note) is still to do.** Fruit Market reference hashes aren't possible (§12).

---

## 14. Decisions

Made on 13 Nov 2026:

1. **Backend:** upstream Melting Pot, with our read-only Lua observer.
2. **Install:** into `prenv` with `--no-deps`, after a dry run. The dry run is done: it needs prenv's versions pinned, and with them nothing existing changes (§2.9).
3. **Preference well-being:** the corrected form (§7.2).
4. **Scale:** the substrate's own, 16 players × 1000 steps, as the default.
5. **Rewards:** the substrate's own, with only the ethics sanction added, at ±3.5 (§6.1).
6. **Labels:** the three added labels, with the partner-first tie-break (§9).
7. **Time spent hungry:** slowly reduces well-being, by 0.1 per hungry step (§7.3).
8. **Changes from the original:** only the ethics reward and the norm labels. Checked in §5.0, and tested in §12, item 4.

9. **Sanction mode:** attributed by default, with joint as an ablation option (§6.3).
10. **Sanction size:** ±3.5 (§6.1).
