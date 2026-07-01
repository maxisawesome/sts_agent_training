#!/usr/bin/env python3
"""
Agent harness for StsEnv — model-agnostic.

A `Policy` maps an observation to a legal action id. The harness (`play_game`) runs the
env to completion against any policy. This is the seam an LLM plugs into: implement
`Policy.choose(observation)` by feeding `render_observation(observation)` to the model and
parsing back an action id. Until then, the heuristic/random stubs let the whole pipeline
run for free.

    from sts_agent import play_game, HeuristicPolicy
    result = play_game(HeuristicPolicy(), seed=1234)
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Protocol

from sts_env import StsEnv, render_observation


class Policy(Protocol):
    def choose(self, observation: dict) -> int:
        """Return the `id` of one of observation['legal_actions']."""
        ...


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def choose(self, observation: dict) -> int:
        return self._rng.choice(observation["legal_actions"])["id"]


class HeuristicPolicy:
    """A deliberately simple stub: block when threatened, otherwise attack; sensible
    overworld defaults. Good enough to exercise the harness, not to play well."""

    def choose(self, observation: dict) -> int:
        actions = observation["legal_actions"]
        if observation["phase"] != "combat":
            return self._overworld(observation, actions)
        return self._combat(observation, actions)

    def _combat(self, obs: dict, actions: list[dict]) -> int:
        combat = obs["combat"]
        incoming = sum(
            m.get("intent_damage", 0) * m.get("intent_hits", 0)
            for m in combat["monsters"]
            if m["alive"] and m.get("is_attacking")
        )
        threatened = incoming > combat["player"]["block"]

        def is_card(a):
            return a["description"].startswith("play ")

        defends = [a for a in actions if is_card(a) and "Defend" in a["description"]]
        attacks = [
            a for a in actions if is_card(a) and ("Strike" in a["description"] or "Bash" in a["description"])
        ]
        other_cards = [a for a in actions if is_card(a)]

        if threatened and defends:
            return defends[0]["id"]
        if attacks:
            return attacks[0]["id"]
        if other_cards:
            return other_cards[0]["id"]
        return actions[-1]["id"]  # end turn (last action)

    def _overworld(self, obs: dict, actions: list[dict]) -> int:
        # No rich event text yet; take the first legal action.
        return actions[0]["id"]


STS_SYSTEM_PROMPT = """You are an expert Slay the Spire player controlling the Ironclad. \
Slay the Spire is a roguelike deckbuilder: you climb a spire over three acts, fighting \
monsters and bosses, building a deck of cards, and collecting relics and potions. The goal \
is to survive and win — reach and defeat the act bosses without dropping to 0 HP.

You play by choosing one action per turn from a numbered list of legal actions. The game \
state you are shown is everything you can observe (a human sees the same — you do NOT know \
the order of your draw pile or unrevealed information).

Strategy guidance:
- In combat: read each monster's intent. Block when incoming damage threatens you; otherwise \
develop your board and deal damage. Kill dangerous or low-HP enemies first. Don't waste energy. \
Account for vulnerable (you deal +50% to it) and weak (it deals -25%). End the turn when you \
have nothing better to do.
- Card rewards: prefer cards that improve your deck's plan; skipping is fine to keep the deck lean.
- Map: weigh elites (risky, good rewards), rest sites (heal or upgrade), shops, and unknowns.
- Events/shops/rest: pick what most improves your run given your HP, gold, and deck.

Always choose by calling the choose_action tool with the integer id of one of the listed \
legal actions. Think briefly, then commit."""

CHOOSE_ACTION_TOOL = {
    "name": "choose_action",
    "description": "Commit to your next action by its id from the legal actions list.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the choice.",
            },
            "action_id": {
                "type": "integer",
                "description": "The id of the chosen action from the legal actions list.",
            },
        },
        "required": ["reasoning", "action_id"],
        "additionalProperties": False,
    },
}


class LlmPolicy:
    """Drives the env with a Claude model via forced tool use. Stateless per step:
    each decision is a single Messages API call over the rendered observation.

    Requires ANTHROPIC_API_KEY in the environment. Swap `model` for cost/quality;
    a full run is hundreds of calls, so Sonnet (`claude-sonnet-4-6`) is much cheaper
    than the Opus default if you're running many seeds."""

    def __init__(self, model: str = "claude-opus-4-8", max_retries: int = 2,
                 max_tokens: int = 2048, verbose: bool = False):
        import anthropic  # imported lazily so the stub policies need no dependency

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens  # generous: the reasoning field must finish before action_id
        self.verbose = verbose
        self.last_reasoning = ""

    def choose(self, observation: dict) -> int:
        valid_ids = [a["id"] for a in observation["legal_actions"]]
        if len(valid_ids) == 1:
            return valid_ids[0]  # no decision to make; skip the API call

        prompt = render_observation(observation)
        messages = [{"role": "user", "content": prompt}]
        max_tokens = self.max_tokens

        for attempt in range(self.max_retries + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=STS_SYSTEM_PROMPT,
                tools=[CHOOSE_ACTION_TOOL],
                tool_choice={"type": "tool", "name": "choose_action"},
                messages=messages,
            )
            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            action_id = tool_use.input.get("action_id") if tool_use else None
            if tool_use is not None and tool_use.input.get("reasoning"):
                self.last_reasoning = tool_use.input["reasoning"]

            if action_id in valid_ids:
                if self.verbose:
                    print(f"    LLM[{action_id}]: {self.last_reasoning}")
                return action_id

            # Failed. The common cause is truncation: a long reasoning field hit max_tokens
            # before action_id was emitted (stop_reason == "max_tokens", incomplete tool JSON).
            truncated = response.stop_reason == "max_tokens"
            if self.verbose:
                print(f"    LLM retry (stop={response.stop_reason}, action_id={action_id})")
            if truncated:
                max_tokens = min(max_tokens * 2, 8192)
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": "Be decisive: at most ONE short sentence of "
                     "reasoning, then call choose_action with a valid action_id."},
                ]
            elif tool_use is not None:
                messages = messages + [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Invalid action_id {action_id}. Valid ids are {valid_ids}. Choose again.",
                        "is_error": True,
                    }]},
                ]
            else:
                messages = messages + [
                    {"role": "user", "content": f"You must call choose_action with one of {valid_ids}."},
                ]

        # Fell through retries — take the first legal action rather than crash.
        if self.verbose:
            print(f"    LLM gave no valid action after {self.max_retries + 1} tries; defaulting to {valid_ids[0]}")
        return valid_ids[0]


def play_game(policy: Policy, seed: int, character: str = "IRONCLAD", ascension: int = 0,
              max_steps: int = 10000, log_dir: str | None = None) -> dict:
    """Play one game. If log_dir is set, write a per-step JSONL transcript to
    <log_dir>/seed_<seed>.jsonl. The transcript is replayable: seed + the action_id
    sequence deterministically reproduces the game."""
    env = StsEnv(character=character, ascension=ascension)
    obs = env.reset(seed=seed)
    steps = 0
    max_floor = obs["floor"]

    log_file = None
    log_path = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"seed_{seed}.jsonl")
        log_file = open(log_path, "w")

    try:
        while not env.done and steps < max_steps:
            actions = obs["legal_actions"]
            action_id = policy.choose(obs)
            if log_file is not None:
                rec = {
                    "step": steps,
                    "phase": obs["phase"],
                    "floor": obs["floor"],
                    "hp": obs["hp"],
                    "action_id": action_id,
                    "action": next((a["description"] for a in actions if a["id"] == action_id), None),
                }
                if obs["phase"] == "combat":
                    rec["turn"] = obs["combat"]["turn"]
                # Only attach reasoning for genuine decisions (a single-option step is auto-taken)
                reasoning = getattr(policy, "last_reasoning", None)
                if reasoning and len(actions) > 1:
                    rec["reasoning"] = reasoning
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
            obs = env.step(action_id)
            steps += 1
            max_floor = max(max_floor, obs["floor"])

        result = {
            "seed": seed,
            "outcome": env.outcome,
            "won": env.outcome == "PLAYER_VICTORY",
            "floor_reached": max_floor,
            "final_floor": obs["floor"],
            "hp": obs["hp"],
            "max_hp": obs["max_hp"],
            "steps": steps,
            "truncated": steps >= max_steps,
        }
        if log_file is not None:
            log_file.write(json.dumps({"type": "result", **result}) + "\n")
            result["log_path"] = log_path
        return result
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Slay the Spire with a policy.")
    parser.add_argument("seeds", type=int, nargs="*", default=[1234, 5678, 99, 42, 7],
                        help="game seeds to play")
    parser.add_argument("--policy", choices=["heuristic", "random", "llm"], default="heuristic")
    parser.add_argument("--model", default="claude-opus-4-8",
                        help="Claude model for --policy llm (e.g. claude-sonnet-4-6 for cheaper runs)")
    parser.add_argument("--verbose", action="store_true", help="print the LLM's per-step reasoning")
    parser.add_argument("--log-dir", default=None,
                        help="where to write transcripts (default: runs/<policy>-<timestamp>)")
    parser.add_argument("--no-log", action="store_true", help="disable run logging")
    args = parser.parse_args()

    if args.policy == "llm":
        policy: Policy = LlmPolicy(model=args.model, verbose=args.verbose)
        label = f"LlmPolicy ({args.model})"
    elif args.policy == "random":
        policy, label = RandomPolicy(), "RandomPolicy"
    else:
        policy, label = HeuristicPolicy(), "HeuristicPolicy"

    if args.no_log:
        log_dir = None
    else:
        log_dir = args.log_dir or os.path.join(
            "runs", f"{args.policy}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    print(f"Playing {len(args.seeds)} game(s) with {label}...\n")
    results = []
    for s in args.seeds:
        r = play_game(policy, seed=s, log_dir=log_dir)
        results.append(r)
        print(
            f"  seed {r['seed']:>6}: {r['outcome']:<14} floor {r['floor_reached']:>2}"
            f"  hp {r['hp']}/{r['max_hp']}  ({r['steps']} steps)"
        )
    wins = sum(r["won"] for r in results)
    avg_floor = sum(r["floor_reached"] for r in results) / len(results)
    print(f"\n  wins {wins}/{len(results)} | avg floor {avg_floor:.1f}")

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        summary = {
            "policy": label,
            "wins": wins,
            "games": len(results),
            "avg_floor": round(avg_floor, 2),
            "results": results,
        }
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  logs: {log_dir}/  (per-seed transcripts + summary.json)")
