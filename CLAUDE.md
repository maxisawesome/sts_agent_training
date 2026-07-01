# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A **text-based environment over the `sts_lightspeed` C++ engine for driving an LLM agent**
through Slay the Spire (Ironclad). It was originally a from-scratch RL project (PPO/DQN
networks) but was repurposed: the network/training stack was deleted, and the valuable
asset — the fast, RNG-accurate engine — is now wrapped in a clean env that an LLM plays.
See the pivot plan at `~/.claude/plans/this-is-an-old-rosy-pillow.md` for the full story.

## Build / install

There is **no system `cmake`** on the dev machine; use the pip-provided one. pybind also
grabs Homebrew Python 3.13 unless pinned — pin to the interpreter you'll run with.

```bash
git submodule update --init --recursive
pip install --user cmake                      # provides a cmake binary
CMAKE=$(python3 -c "import cmake,os;print(os.path.join(os.path.dirname(cmake.__file__),'data','bin','cmake'))")
cd sts_lightspeed
"$CMAKE" -DPython_EXECUTABLE=$(which python3) -DPYBIND11_FINDPYTHON=ON .
make -j4 slaythespire                         # builds slaythespire.cpython-*.so in this dir
# make the module importable from anywhere:
cp slaythespire*.so "$(python3 -c 'import site;print(site.getusersitepackages())')"/
```

Verify (combat MUST be exposed — `BattleContext` is what makes the game playable):
```bash
python3 -c "import slaythespire as s; print('BattleContext:', hasattr(s,'BattleContext'))"
```

`pip install -e .` builds the same extension via `setup.py`, but the custom build can be
skipped under pip build isolation — the manual build above is the reliable path.

## Run

```bash
python3 verify_combat.py                       # drive a real battle from Python (sanity)
python3 verify_env.py [seed]                    # full run through the unified env (sanity)

python3 sts_agent.py                            # heuristic stub over default seeds
python3 sts_agent.py --policy random 1 2 3
python3 sts_agent.py --policy llm --verbose 1234            # needs ANTHROPIC_API_KEY
python3 sts_agent.py --policy llm --model claude-sonnet-4-6 1234 5678   # cheaper for many seeds
```

Runs log to `runs/<policy>-<timestamp>/` (per-seed `.jsonl` transcripts + `summary.json`);
`--no-log` to disable, `--log-dir` to relocate. A transcript is replayable: seed + the
`action_id` sequence deterministically reproduces the game.

## Architecture

- **`sts_lightspeed/`** — C++ engine (submodule). Out-of-combat play runs through
  `GameContext`; combat runs in a separate `BattleContext`. Both, plus the combat-action
  enumerator and serializers, are bound in `sts_lightspeed/bindings/slaythespire.cpp`
  (the combat bindings + `get_screen_text`/`take_overworld_action` were added here — see
  the binding file's comments).
- **`sts_env.py`** — `StsEnv`: one `reset / observe / legal_actions / step / render` loop
  spanning combat and overworld (auto-enters/exits battles). `observe()` reports only
  observable state (draw-pile order is hidden). `render_observation(obs)` is the text an
  LLM sees.
- **`sts_agent.py`** — `Policy` protocol + `play_game()` + policies: `HeuristicPolicy`,
  `RandomPolicy`, and `LlmPolicy` (Claude via forced tool use; stateless per step).
  Run logging lives here.
- **`verify_combat.py` / `verify_env.py`** — sanity scripts. **`baseline_run.md`** —
  saved sample stub output (heuristic avg ~floor 9–11, random ~5; both 0 wins).

## Conventions / gotchas

- Combat loop in C++ terms: `BattleContext(); bc.init(gc)` → loop `get_battle_actions` →
  `action.execute(bc)` until `bc.is_battle_over` → `bc.exit_battle(gc)`.
- **Never call `get_screen_text` on a `BATTLE` screen** — `BattleSimulator::printActions`
  dumps to stdout. `StsEnv._menu_available` guards this; preserve that guard.
- Combat action descriptions are built in `sts_env._describe_battle_action` (plain
  language). If you change that format, update any policy that string-matches it
  (`HeuristicPolicy._combat` keys on `"play "` / `"Defend"` / `"Strike"`/`"Bash"`).
- A new policy (e.g. a future fine-tuned model) is a drop-in: implement
  `Policy.choose(observation) -> action_id`.

## Dependencies

`pybind11` + `cmake` (build), `anthropic` (the `LlmPolicy`), optional `wandb`. No torch.
The model default is `claude-opus-4-8`; a full run is hundreds of calls, so
`claude-sonnet-4-6` is much cheaper for batch eval.
