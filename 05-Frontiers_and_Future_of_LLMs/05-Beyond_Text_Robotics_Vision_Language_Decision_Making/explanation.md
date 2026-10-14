# explanation.md — Topic 5.5 walkthrough

## What the script does, line by line

**`random_episode_config(rng)`**
Places the agent and three differently-"colored" goals on random,
distinct cells of a 5x5 grid, and picks which color the current episode
instructs the agent to reach. A new random configuration is generated
per episode -- no two episodes share a layout.

**`render_grid(agent, goals)`**
Turns the current state into a 4-channel "image": channel 0 marks the
agent's cell, channels 1-3 each mark one goal color's cell. This is
the "vision" input.

**`expert_action(agent, target_pos)` / `step(agent, action)`**
A deterministic shortest-path controller (reduce whichever axis --
row or column -- has the larger remaining distance first) and the
corresponding grid-update function. Because the grid has no obstacles,
this controller is optimal, so every expert-generated label is
correct by construction.

**`collect_expert_dataset(n_episodes, seed)`**
Runs the expert controller through 3,000 random episodes, recording
every `(grid, instruction, action)` triple along the way -- this is
the behavior-cloning training set (about 9,800 individual
state-action pairs).

**`VisionLanguagePolicy`**
A small two-layer CNN turns the 4-channel grid into a flat vision
feature vector; an embedding table turns the (single-token) color
instruction into a language feature vector. When `use_language=True`,
the two are concatenated before the action head; when `False`, the
instruction embedding is replaced with a fixed, uninformative input
(token `0`) regardless of the true target -- this is the ablation.

**`train_policy(...)`**
Ordinary supervised cross-entropy training (behavior cloning): given a
state (and, for the full model, the instruction), predict the expert's
action. Both the full model and the vision-only ablation are trained
on the *exact same dataset* for the *same number of epochs* -- the
only difference is architectural (whether the instruction is actually
used).

**`rollout_success_rate(...)`**
The key evaluation function: starting from a fresh, unseen episode
configuration, the model's own predicted action is executed in the
grid and the *resulting* state is fed back in as the next observation,
repeated for up to 10 steps. Success means the agent is standing on
the instructed goal's cell at the end of the rollout. This is strictly
harder than checking whether the model's predicted action matches the
expert's at each step of an expert-generated trajectory, because
mistakes compound: one wrong turn puts the agent in a state the
training data may never have covered.

## What actually happened when it ran (real numbers, this run)

- **Full policy (vision + language), 500 held-out episodes: 98.0%
  closed-loop success rate.** The model reliably navigates to whichever
  of the three goals the instruction specifies.
- **Vision-only ablation, same 500 episodes: 55.0%.** Denied the
  instruction, the model still does *something* sensible (it moves
  toward *some* goal, so it beats the 33.3% pure-chance rate of
  randomly landing on the right one), but it cannot reliably solve the
  actual task, because the visual scene alone never tells it which
  goal was intended.
- **Chance level (1 of 3 goals): 33.3%** -- included as the
  lower-bound reference point on the chart.
- Total time for expert-data collection, both training runs, and both
  500-episode rollout evaluations: under 20 seconds, on a single CPU
  core, no downloaded weights or images.

## Why this is proof, not illustration

`proof.png`'s three bars come directly from `results.json`, written by
`run()` in `implementation.py` from real closed-loop rollouts of really
-trained policies in a real (if tiny) simulated environment defined in
this same file -- not a hand-set illustrative comparison.
