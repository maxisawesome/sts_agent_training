# Slay the Spire — LLM Agent Environment

A clean, **text-based environment** over the [`sts_lightspeed`](https://github.com/gamerpuppy/sts_lightspeed)
C++ engine for driving an LLM through Slay the Spire (Ironclad), plus a small agent harness
and evaluation runner.

> **History:** this started as a from-scratch RL project (PPO/DQN policy networks + tree
> search). That stack was a poor fit and has been removed. The genuinely valuable asset —
> the fast, 100% RNG-accurate engine — is now wrapped in a clean environment that a language
> model plays directly. A trained/fine-tuned model can drop in later behind the same
> `Policy` interface.

## What's here

| File | Purpose |
|---|---|
| `sts_lightspeed/` | C++ engine (submodule). Combat (`BattleContext`) and overworld are bound to Python in `bindings/slaythespire.cpp`. |
| `sts_env.py` | `StsEnv` — one `reset / observe / legal_actions / step / render` loop over combat **and** overworld. Reports only observable state (no hidden draw-pile order). |
| `sts_agent.py` | `Policy` interface, `play_game()`, and policies: `HeuristicPolicy`, `RandomPolicy`, `LlmPolicy` (Claude). Per-run logging. |
| `verify_combat.py`, `verify_env.py` | Sanity scripts. |
| `baseline_run.md` | Saved sample output from the stub policies. |

## Setup

Requires **Python 3.9+** and a **C++17 compiler**. No system CMake needed (pip provides one).

```bash
git submodule update --init --recursive
pip install --user cmake anthropic            # cmake builds the engine; anthropic powers the LLM policy

CMAKE=$(python3 -c "import cmake,os;print(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake'))")
cd sts_lightspeed
"$CMAKE" -DPython_EXECUTABLE=$(which python3) -DPYBIND11_FINDPYTHON=ON .
make -j4 slaythespire
cp slaythespire*.so "$(python3 -c 'import site;print(site.getusersitepackages())')"/
cd ..

# verify — combat must be exposed for the game to be playable
python3 -c "import slaythespire as s; print('ok' if hasattr(s,'BattleContext') else 'MISSING combat bindings')"
```

> Pin `-DPython_EXECUTABLE` to the interpreter you'll run with — otherwise pybind may build
> the extension against a different Python (e.g. Homebrew) and the import will fail.

## Run

```bash
# Stub policies — no API key, run free:
python3 verify_env.py 1234                      # play a full game, print sample observations
python3 sts_agent.py                            # heuristic over default seeds
python3 sts_agent.py --policy random 1 2 3

# LLM policy — needs ANTHROPIC_API_KEY:
export ANTHROPIC_API_KEY=sk-ant-...
python3 sts_agent.py --policy llm --verbose 1234                       # one seed, prints reasoning
python3 sts_agent.py --policy llm --model claude-sonnet-4-6 1234 5678  # cheaper for batches
```

A full run is hundreds of model calls, so use `--model claude-sonnet-4-6` for cost-sensitive
batch evaluation; the default is `claude-opus-4-8`.

### Logging

Every run writes to `runs/<policy>-<timestamp>/`:
- `seed_<N>.jsonl` — one record per decision (step, phase, floor, hp, chosen action, and the
  LLM's `reasoning`), ending in a `result` record.
- `summary.json` — batch aggregate (wins, avg floor, per-seed results).

Transcripts are **replayable**: a seed plus its `action_id` sequence deterministically
reproduces the game. `--no-log` disables, `--log-dir` relocates.

## The agent interface

A policy is anything with `choose(observation) -> action_id`:

```python
from sts_agent import play_game, LlmPolicy
result = play_game(LlmPolicy(model="claude-opus-4-8"), seed=1234, log_dir="runs/demo")
```

`observation` carries `phase`, `hp`, `floor`, and `legal_actions` (`[{id, description}]`),
with combat or overworld detail. `render_observation(observation)` is the exact text the LLM
is shown. Your own model swaps in the same way — implement `choose` and pass it to
`play_game`.

## Baseline

Stub policies (offline, in `baseline_run.md`): heuristic ≈ floor 9–11, random ≈ floor 5,
both 0 wins over 10 seeds. That's the floor an LLM should clear comfortably.

## Roadmap

The environment doubles as a **data generator and eval harness** for training your own
agent: collect trajectories from a strong policy, then fine-tune / RL against the same
interface and measure with `play_game`.
