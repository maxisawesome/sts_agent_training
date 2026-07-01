# Plan: RL-training path for the STS LLM agent

## Context

The pivot is done: `sts_lightspeed` is built with combat bindings, `sts_env.py` is a clean
text env (`reset/observe/legal_actions/step/render`), `sts_agent.py` has a working Claude
`LlmPolicy` (reached ~floor 21) + run logging. The question now is **how to train an
open-weights model on this** (Claude can't be fine-tuned by us; it's a teacher + baseline).

Research finding that shapes everything — the ecosystem has converged on an **env / trainer
split**:
- **Env side (ours):** emit observations + a scalar reward. Knows nothing about tokens.
- **Trainer side (the framework):** prompt-building, vLLM rollouts, per-token logprobs,
  masking, GRPO advantages, multi-GPU backprop.

So we do **not** write the RL math. We expose the env + a reward in a standard shape and let
a framework do the heavy lifting. Credit assignment across the ~200-step episode is GRPO's
job: one trajectory-level advantage shared over the agent's tokens, env/observation tokens
masked out — which fits our terminal win/loss reward. SFT-first is mandatory (can't RL a cold
model on a long sparse-reward game).

## Recommended path (two stages)

1. **Prototype on OpenPipe ART** — lowest friction. You keep our agent loop and set
   `trajectory.reward`; our env maps almost 1:1 onto ART's `examples/2048` (multi-step game,
   terminal reward). Small model + LoRA on 1–few GPUs. Validates reward + loop + SFT→RL
   cheaply before committing to heavy infra.
2. **Scale on Prime Intellect (`verifiers` + `prime-rl`)** — you've deployed on Prime Intellect
   before; production-proven (trained INTELLECT-3); multi-GPU/multi-node. Port the env to a
   `verifiers.MultiTurnEnv` (+ `Rubric`). Alternatives if preferred: SkyRL (`BaseTextEnv`,
   long-horizon-native), OpenRLHF (`AgentInstanceBase`), veRL (`AgentLoopBase`, most powerful/
   heaviest). Targeting the **OpenEnv** standard (Gym-style, HF/Meta) keeps us portable across
   most of these.

## What we need to BUILD (framework-agnostic, repo-side — "what you create")

The heavy RL lifting is the framework's; our work is small and reusable across all of them:

1. **`sts_reward.py`** — scalar reward from existing signals (`env.outcome`, `play_game`
   result, per-step floor/hp). Terminal win/loss dominant + light shaping (floor reached,
   act-boss clears, HP fraction). Verifiable, no reward model.
2. **`LocalPolicy` in `sts_agent.py`** — mirror of `LlmPolicy` that generates **text** via an
   OpenAI-compatible vLLM endpoint and parses the action out (instead of Claude tool-use).
   This is both the RL rollout policy and the SFT-inference policy. The text action format
   must be parseable from `legal_actions`.
3. **SFT data generation** — enrich `play_game` logging to capture the full prompt
   (`render_observation`) + completion (today it logs only `action_id` + reasoning), run the
   Claude `LlmPolicy` over many seeds, keep good/winning trajectories → chat-format SFT
   dataset. (Harness is ~80% there; games are seed-replayable so old logs can be reconstructed.)
4. **Trainable-env adapter** — thin wrapper exposing `StsEnv` in the chosen framework's shape:
   an ART `async rollout() -> Trajectory` for stage 1, and/or a `verifiers.MultiTurnEnv`
   (`env_response` + `is_completed` + `Rubric`) / OpenEnv `Environment` for stage 2. Maps
   `render→message`, `step→action`, reward→`trajectory.reward`.
5. **Parallel/batched env runner** — for rollout throughput (the deleted `VectorizedStsEnvironment`
   can be rebuilt, simpler now: many `StsEnv`s stepped together).

## Phased execution

- **Phase 0 — make the env trainable (repo, framework-agnostic):** `sts_reward.py`, enriched
  trajectory logging, `LocalPolicy` (text-gen via vLLM), small parallel runner. Verify offline.
- **Phase 1 — SFT bootstrap:** generate Claude trajectories → SFT a small open model (e.g.
  Qwen2.5-7B) on `render_observation → action` → eval with `play_game` (target: beat the
  heuristic baseline of ~floor 9–11).
- **Phase 2 — RL prototype (ART):** wrap env in an ART rollout, GRPO on the SFT'd model,
  confirm win-rate / floor trends up. Small model + LoRA.
- **Phase 3 — scale (prime-rl/verifiers, or SkyRL/veRL):** port env to `MultiTurnEnv`/OpenEnv,
  full online GRPO on cloud multi-GPU; iterate reward shaping + model size.

## Critical files

- **New:** `sts_reward.py`; adapter(s) (`sts_art_rollout.py` for stage 1; a `verifiers`/OpenEnv
  env package for stage 2); parallel runner (`sts_parallel.py`).
- **Modify:** `sts_agent.py` (add `LocalPolicy`, capture full prompt+completion in logs);
  `requirements.txt` (per stage: `trl`/`vllm`/`transformers` or `openpipe-art` or
  `verifiers`/`prime-rl`); `CLAUDE.md`.

## Verification

- **Reward:** replay a logged game (seed + action_ids) and assert reward matches outcome
  (win > loss; monotone-ish with floor).
- **LocalPolicy:** stand up a small model under vLLM, play a game, confirm legal actions are
  parsed and `play_game` completes.
- **SFT:** eval the fine-tuned model with `play_game` over the baseline seed set; beat
  heuristic avg floor.
- **RL:** reward / floor-reached trend upward across training steps; transcripts under `runs/`.

## Open decisions (need your input before/while executing)

- **Base model + size** (Qwen vs Llama, 3B/7B/…), **LoRA vs full fine-tune**.
- **Stage-2 framework:** Prime Intellect (`verifiers`+`prime-rl`, recommended given your
  history) vs SkyRL / veRL / OpenRLHF.
- **Whether to target OpenEnv now** (portability) vs a framework-native adapter first.
- How much **reward shaping** vs pure terminal win/loss to start with.

## Framework notes / sources (2026)

- **OpenPipe ART** — `github.com/OpenPipe/ART`; bring-your-own-loop, set `trajectory.reward`;
  `examples/2048` is the close analog; LoRA/single-GPU sweet spot; GRPO; SFT in-stack.
- **verifiers + prime-rl** — `github.com/PrimeIntellect-ai`; `MultiTurnEnv` + `Rubric`;
  production-proven (INTELLECT-3, 512 H200s); scales to 1000+ GPUs.
- **SkyRL** — `NovaSky-AI/SkyRL`; `BaseTextEnv.step → {observations, reward, done}`; terminal
  reward by default; `skyrl-agent` purpose-built for long-horizon (50–100+ steps).
- **OpenRLHF** — `AgentInstanceBase` (`reset`/`step`); framework owns loop; lowest absolute
  integration effort; DeepSpeed/Ray/vLLM.
- **veRL** — `volcengine/verl`; `AgentLoopBase.run` (you own the loop); widest algos/backends;
  heaviest config/ops.
- **OpenEnv** — `meta-pytorch/OpenEnv`; emerging Gym-style env standard (HF/Meta + broad
  committee incl. Prime Intellect, vLLM, SkyRL, Unsloth, TRL). Our env is ~90% this shape.
- All are GRPO-family, vLLM/SGLang rollouts, SFT-in-stack. Trajectory data contract is
  `(token_ids, logprobs, finish_reason)` + completion mask + reward — all owned by the trainer.

This planning pass made no code changes.
