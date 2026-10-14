"""
Topic 5.5 — Beyond Text: Robotics, Vision-Language, and Decision-Making
==========================================================================

GOAL
----
Demonstrate, with a real trained-and-measured experiment, why fusing a
visual observation with a language instruction is *necessary* -- not
just helpful -- for decision-making in embodied/robotic settings, and
why this is a genuinely different problem from either pure vision or
pure language understanding.

TASK
----
A small grid-world (5x5) contains an agent and three differently
"colored" goal cells at random positions. A language instruction
specifies which color the agent must navigate to. Critically, **the
visual scene alone is ambiguous**: with three goals present and no
instruction, there is no way to know which one is the actual target --
exactly the situation a household robot faces when told "bring me the
red cup" among several cups of different colors, or a self-driving
car's planner faces when told a specific turn to take among several
legal ones.

Two behavior-cloning policies are trained on identical expert
trajectories (a deterministic shortest-path controller supplies expert
actions, so the training labels are correct by construction, not
noisy):

- **Full policy**: a small CNN vision encoder + an instruction
  embedding, fused by concatenation, feeding a small MLP action head.
- **Vision-only ablation**: architecturally identical, but the
  instruction embedding is always the same fixed (uninformative) input
  -- i.e., the policy can see the grid but is never told which color
  matters.

Both policies are evaluated the same way: **closed-loop rollout** in
new, unseen episodes (the model's own predicted action is executed and
fed back in as the next observation, up to 10 steps), measuring the
fraction of episodes where the agent reaches the *instructed* goal.

WHY THIS IS A FAIR MODEL OF VISION-LANGUAGE-ACTION FUSION
-------------------------------------------------------------
This is a genuine instance of the same architecture family used in
real vision-language-action (VLA) robotics models (e.g., RT-2-style
systems): a visual encoder, a language encoder, a fusion step, and a
policy/action head, trained via imitation learning and evaluated via
closed-loop rollout rather than single-step accuracy (which can be
misleadingly high even for a policy that fails over a full episode).
The task is small enough to run on a single CPU core, but the
underlying claim under test -- language conditioning is *necessary*,
not decorative, once the visual scene alone is ambiguous -- is exactly
the claim real household/warehouse robotics language-conditioning
papers make.

SCALING TO PRODUCTION MODELS
------------------------------
Production vision-language-action models replace this toy CNN + toy
instruction embedding with a full vision-language backbone (e.g., a
CLIP-style or ViT vision encoder paired with a pretrained language
model) and are trained on real demonstration data (teleoperated robot
trajectories, or human video), typically with tokenized/discretized
continuous actions rather than 4 discrete moves. The evaluation
methodology used here -- closed-loop success rate under an ambiguous
multi-goal scene, with a vision-only ablation as a control -- is
exactly the kind of ablation real VLA papers report to justify that
language conditioning is doing real work, not just adding parameters.
"""

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GRID = 5
N_GOALS = 3          # three differently-"colored" goals per episode
D_MODEL = 32
MAX_STEPS = 10
N_TRAIN_EPISODES = 3000
BC_EPOCHS = 25
N_EVAL_EPISODES = 500


def random_episode_config(rng):
    cells = [(r, c) for r in range(GRID) for c in range(GRID)]
    rng.shuffle(cells)
    agent = cells[0]
    goals = {color: cells[1 + i] for i, color in enumerate(range(1, N_GOALS + 1))}
    target_color = rng.randint(1, N_GOALS)
    return agent, goals, target_color


def render_grid(agent, goals):
    """4-channel one-hot 'image': channel 0 = agent, channels 1-3 = goal colors."""
    g = torch.zeros(N_GOALS + 1, GRID, GRID)
    g[0, agent[0], agent[1]] = 1.0
    for color, (r, c) in goals.items():
        g[color, r, c] = 1.0
    return g


def expert_action(agent, target_pos):
    """Deterministic shortest-path controller: reduce the larger-magnitude
    axis first. Optimal in this obstacle-free grid."""
    ar, ac = agent
    tr, tc = target_pos
    if ar == tr and ac == tc:
        return None
    if abs(tr - ar) >= abs(tc - ac):
        return 0 if tr < ar else 1  # up / down
    return 2 if tc < ac else 3      # left / right


def step(agent, action):
    r, c = agent
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(GRID - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    elif action == 3:
        c = min(GRID - 1, c + 1)
    return (r, c)


def collect_expert_dataset(n_episodes, seed):
    rng = random.Random(seed)
    states, instrs, actions = [], [], []
    for _ in range(n_episodes):
        agent, goals, target_color = random_episode_config(rng)
        target_pos = goals[target_color]
        for _ in range(MAX_STEPS):
            a = expert_action(agent, target_pos)
            if a is None:
                break
            states.append(render_grid(agent, goals))
            instrs.append(target_color)
            actions.append(a)
            agent = step(agent, a)
    return torch.stack(states), torch.tensor(instrs), torch.tensor(actions)


class VisionLanguagePolicy(nn.Module):
    """Small CNN vision encoder + instruction embedding, fused by
    concatenation -- the same architecture family (vision encoder +
    language encoder + fusion + action head) used in real VLA robotics
    models, just scaled down."""

    def __init__(self, use_language=True):
        super().__init__()
        self.use_language = use_language
        self.conv = nn.Sequential(
            nn.Conv2d(N_GOALS + 1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )
        self.vision_proj = nn.Linear(16 * GRID * GRID, D_MODEL)
        self.instr_emb = nn.Embedding(N_GOALS + 1, D_MODEL)  # 0 = uninformative/pad token
        fusion_in = D_MODEL * 2 if use_language else D_MODEL
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_in, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 4)
        )

    def forward(self, grid, instr):
        v = self.conv(grid)
        v = v.reshape(v.size(0), -1)
        v = self.vision_proj(v)
        if self.use_language:
            l = self.instr_emb(instr)
            h = torch.cat([v, l], dim=-1)
        else:
            h = v
        return self.policy_head(h)


def train_policy(use_language, S, I, A, epochs, batch=64, seed=0):
    model = VisionLanguagePolicy(use_language=use_language)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = S.size(0)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(S[idx], I[idx])
            loss = F.cross_entropy(logits, A[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, time.time() - t0


@torch.no_grad()
def rollout_success_rate(model, use_language, n_episodes, seed, max_steps=MAX_STEPS):
    """Closed-loop evaluation: the model's own action is executed and fed
    back in as the next observation -- not single-step next-action accuracy."""
    rng = random.Random(seed)
    successes = 0
    for _ in range(n_episodes):
        agent, goals, target_color = random_episode_config(rng)
        target_pos = goals[target_color]
        reached = False
        for _ in range(max_steps):
            if agent == target_pos:
                reached = True
                break
            grid = render_grid(agent, goals).unsqueeze(0)
            instr = torch.tensor([target_color if use_language else 0])
            logits = model(grid, instr)
            a = logits.argmax(-1).item()
            agent = step(agent, a)
        if agent == target_pos:
            reached = True
        successes += int(reached)
    return successes / n_episodes


def run():
    t0 = time.time()
    S, I, A = collect_expert_dataset(N_TRAIN_EPISODES, seed=1)

    model_full, t_full = train_policy(True, S, I, A, epochs=BC_EPOCHS, seed=1)
    sr_full = rollout_success_rate(model_full, True, n_episodes=N_EVAL_EPISODES, seed=42)

    model_vis, t_vis = train_policy(False, S, I, A, epochs=BC_EPOCHS, seed=1)
    sr_vis = rollout_success_rate(model_vis, False, n_episodes=N_EVAL_EPISODES, seed=42)

    results = {
        "n_train_transitions": S.size(0),
        "n_eval_episodes": N_EVAL_EPISODES,
        "full_policy_success_rate": sr_full,
        "vision_only_success_rate": sr_vis,
        "chance_level": 1 / N_GOALS,
        "train_time_full": t_full,
        "train_time_vision_only": t_vis,
    }
    print(f"FULL (vision + language) closed-loop success rate:  {sr_full:.3f}")
    print(f"VISION-ONLY (ablation)   closed-loop success rate:  {sr_vis:.3f}")
    print(f"Chance level (1/{N_GOALS} goals):                    {1/N_GOALS:.3f}")
    print(f"Total time: {time.time() - t0:.1f}s on 1 CPU core")
    return results


if __name__ == "__main__":
    run()
