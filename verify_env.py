#!/usr/bin/env python3
"""
Phase 1 verification: drive the unified StsEnv through a full run with a trivial
policy, exercising both combat and overworld via one interface.

Policy: in combat, prefer the first CARD action (greedy), else end turn; out of
combat, take the first legal action. Just proves the loop runs end to end.
"""
import random

from sts_env import StsEnv, render_observation


def pick(env, obs):
    actions = obs["legal_actions"]
    if env.in_combat:
        # prefer playing a card; the description of an end-turn action says "end turn"
        cards = [a for a in actions if a["description"].startswith("play ")]
        if cards:
            return cards[0]["id"]
    return actions[0]["id"]


def main():
    seed = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 1234
    env = StsEnv()
    obs = env.reset(seed=seed)

    print("===== INITIAL OBSERVATION =====")
    print(render_observation(obs))
    print()

    steps = 0
    shown_combat = False
    while not env.done and steps < 5000:
        if env.in_combat and not shown_combat:
            print("===== FIRST COMBAT OBSERVATION (rendered for LLM) =====")
            print(render_observation(obs))
            print()
            shown_combat = True
        obs = env.step(pick(env, obs))
        steps += 1

    print(f"===== GAME OVER after {steps} steps =====")
    print(f"outcome={env.outcome}  floor={obs['floor']}  hp={obs['hp']}/{obs['max_hp']}")


if __name__ == "__main__":
    main()
