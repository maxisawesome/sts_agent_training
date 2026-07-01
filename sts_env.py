#!/usr/bin/env python3
"""
Slay the Spire environment — a clean, text-first wrapper over the sts_lightspeed
C++ engine, designed for an LLM agent (not a fixed-size tensor policy).

One interface spans the whole run. Out of combat the engine is driven through the
`GameContext` action API; in combat it is driven through a `BattleContext` (see the
combat bindings in sts_lightspeed). This wrapper hides that seam: it auto-enters and
auto-exits battles, and exposes a single `observe()` / `legal_actions()` / `step()` loop.

Hidden information is respected — `observe()` only reports what a human player can see
(revealed monster intents, current hand, deck contents but not draw-pile order).

    env = StsEnv()
    obs = env.reset(seed=1234)
    while not env.done:
        actions = env.legal_actions()          # [{id, description}, ...]
        obs = env.step(actions[0]["id"])
    print(env.outcome)
"""
from __future__ import annotations

import re
from typing import Any, Optional

import slaythespire as sts

# A menu line is an actionable choice iff its label (text before the first ':') is a
# console command: a bare integer, a "<kind> <i> [<j>]" reward/shop selector, or a keyword.
_COMMAND_RE = re.compile(
    r"^(?:\d+"
    r"|(?:card|relic|potion|gold|maxhp)\s+\d+(?:\s+\d+)?"
    r"|skip|proceed|remove|emeraldKey|sapphireKey)$"
)


def _parse_menu(text: str) -> list[dict]:
    """Parse ConsoleSimulator menu text into [{command, description}] choices.

    Header/info lines (e.g. 'Map Screen: ...', 'CurX: 3 CurY: 2') are skipped because
    their label is not a valid command."""
    choices: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        label, _, rest = line.partition(":")
        command = " ".join(label.split())  # normalize internal whitespace ("0 " -> "0")
        if not _COMMAND_RE.match(command):
            continue
        choices.append({"command": command, "description": rest.strip() or command})
    return choices


class StsEnv:
    def __init__(self, character: str = "IRONCLAD", ascension: int = 0):
        self.character_class = getattr(sts.CharacterClass, character)
        self.ascension = ascension
        self.gc: Optional[sts.GameContext] = None
        self._bc: Optional[sts.BattleContext] = None
        self.seed: Optional[int] = None

    # ---- lifecycle ----------------------------------------------------------
    def reset(self, seed: int) -> dict:
        self.seed = seed
        self.gc = sts.GameContext(self.character_class, seed, self.ascension)
        self._bc = None
        self._sync_battle()
        return self.observe()

    def _sync_battle(self) -> None:
        """Enter a battle when the engine lands on a BATTLE screen; exit when it ends."""
        if self._bc is not None and self._bc.is_battle_over:
            self._bc.exit_battle(self.gc)
            self._bc = None
        if (
            self._bc is None
            and self.gc.outcome == sts.GameOutcome.UNDECIDED
            and self.gc.screen_state == sts.ScreenState.BATTLE
        ):
            self._bc = sts.BattleContext()
            self._bc.init(self.gc)

    # ---- state --------------------------------------------------------------
    @property
    def in_combat(self) -> bool:
        return self._bc is not None

    @property
    def done(self) -> bool:
        return self.gc.outcome != sts.GameOutcome.UNDECIDED

    @property
    def outcome(self) -> str:
        return {
            sts.GameOutcome.UNDECIDED: "UNDECIDED",
            sts.GameOutcome.PLAYER_VICTORY: "PLAYER_VICTORY",
            sts.GameOutcome.PLAYER_LOSS: "PLAYER_LOSS",
        }[self.gc.outcome]

    def _battle_actions(self):
        return list(sts.get_battle_actions(self._bc))

    def _menu_available(self) -> bool:
        # get_screen_text routes through ConsoleSimulator, which dumps battle state to
        # stdout on a BATTLE screen. Only read it on a genuine, live overworld screen.
        return not self.done and self.gc.screen_state != sts.ScreenState.BATTLE

    def _overworld_choices(self) -> list[dict]:
        """Parsed, human-readable menu for the current out-of-combat screen.

        Falls back to the raw GameAction enumeration if the menu can't be parsed, so the
        loop never stalls on a screen we don't recognize."""
        if not self._menu_available():
            return []
        choices = _parse_menu(sts.get_screen_text(self.gc))
        if not choices and self.gc.get_valid_actions():
            choices = [
                {"command": None, "_bits": b, "description": f"action {i}"}
                for i, b in enumerate(self.gc.get_valid_actions())
            ]
        return choices

    def legal_actions(self) -> list[dict]:
        """Stable-id, human-readable list of currently legal actions."""
        if self.done:
            return []
        if self.in_combat:
            bstate = sts.get_battle_state(self._bc)
            return [
                {"id": i, "description": _describe_battle_action(a, bstate)}
                for i, a in enumerate(self._battle_actions())
            ]
        return [
            {"id": i, "description": c["description"]}
            for i, c in enumerate(self._overworld_choices())
        ]

    def step(self, action_id: int) -> dict:
        if self.done:
            raise RuntimeError("step() called on a finished game")
        if self.in_combat:
            actions = self._battle_actions()
            if not 0 <= action_id < len(actions):
                raise ValueError(f"illegal combat action id {action_id} (have {len(actions)})")
            actions[action_id].execute(self._bc)
        else:
            choices = self._overworld_choices()
            if not 0 <= action_id < len(choices):
                raise ValueError(f"illegal action id {action_id} (have {len(choices)})")
            choice = choices[action_id]
            if choice["command"] is None:
                self.gc.execute_action(choice["_bits"])
            else:
                sts.take_overworld_action(self.gc, choice["command"])
        self._sync_battle()
        return self.observe()

    # ---- observation --------------------------------------------------------
    def observe(self) -> dict:
        """Observable game state as a structured dict (combat or overworld)."""
        gc = self.gc
        base: dict[str, Any] = {
            "phase": "combat" if self.in_combat else "overworld",
            "floor": gc.floor_num,
            "act": gc.act,
            "hp": gc.cur_hp,
            "max_hp": gc.max_hp,
            "gold": gc.gold,
            "done": self.done,
            "outcome": self.outcome,
        }
        if self.in_combat:
            base["combat"] = sts.get_battle_state(self._bc)
        else:
            base["screen"] = str(gc.screen_state).split(".")[-1]
            base["deck"] = sorted(_card_name(c) for c in gc.deck)
            base["relics"] = [str(r.id).split(".")[-1] for r in gc.relics]
            if self._menu_available():
                base["menu_text"] = sts.get_screen_text(gc)
        base["legal_actions"] = self.legal_actions()
        return base

    def render(self) -> str:
        """Human-readable rendering of observe() — the basis for an LLM prompt."""
        return render_observation(self.observe())


def _card_name(card) -> str:
    name = str(card.id).split(".")[-1]
    return name + ("+" if getattr(card, "upgraded", False) else "")


def _describe_battle_action(action, bstate: dict) -> str:
    """Plain-language description of a combat action, built from the battle state +
    the action's indices (instead of the engine's terse debug string)."""
    at = action.get_action_type()
    if at == sts.ActionType.END_TURN:
        return "end turn"

    if at == sts.ActionType.CARD:
        src = action.get_source_idx()
        hand = bstate["hand"]
        if 0 <= src < len(hand):
            card = hand[src]
            name = card["name"] + ("+" if card["upgraded"] else "")
            desc = f"play {name} (cost {card['cost']})"
            if card.get("requires_target"):
                tgt = action.get_target_idx()
                mon = next((m for m in bstate["monsters"] if m["idx"] == tgt), None)
                if mon is not None:
                    desc += f" on {mon['name']}"
            return desc
        return f"play card {src}"

    if at == sts.ActionType.POTION:
        src = action.get_source_idx()
        name = dict(bstate["potions"]).get(src, f"potion {src}")
        tgt = action.get_target_idx()
        if tgt > 5:  # discard sentinel (engine encodes -1)
            return f"discard potion {name}"
        desc = f"use potion {name}"
        if len(bstate["monsters"]) > 1:  # only worth naming a target with multiple enemies
            mon = next((m for m in bstate["monsters"] if m["idx"] == tgt), None)
            if mon is not None:
                desc += f" on {mon['name']}"
        return desc

    if at in (sts.ActionType.SINGLE_CARD_SELECT, sts.ActionType.MULTI_CARD_SELECT):
        task = bstate.get("card_select_task", "select")
        return f"{task}: choose option {action.get_source_idx()}"

    return f"action {action.get_source_idx()}"


# ---- text rendering ---------------------------------------------------------
def render_observation(obs: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"Floor {obs['floor']} (Act {obs['act']}) | HP {obs['hp']}/{obs['max_hp']} | Gold {obs['gold']}"
    )
    if obs["phase"] == "combat":
        c = obs["combat"]
        p = c["player"]
        pw = f" | powers: {_fmt_powers(p['powers'])}" if p["powers"] else ""
        lines.append(f"-- COMBAT (turn {c['turn']}, {c['input_state']}) --")
        lines.append(f"You: {p['hp']}/{p['max_hp']} HP, {p['block']} block, {p['energy']} energy{pw}")
        for m in c["monsters"]:
            if not m["alive"]:
                continue
            intent = m["intent"]
            if m.get("is_attacking"):
                intent = f"attack {m.get('intent_damage')}x{m.get('intent_hits')}"
            mp = f" | {_fmt_powers(m['powers'])}" if m["powers"] else ""
            blk = f", {m['block']} block" if m["block"] else ""
            lines.append(
                f"  [{m['idx']}] {m['name']}: {m['hp']}/{m['max_hp']} HP{blk} | intent: {intent}{mp}"
            )
        hand = ", ".join(
            f"{cd['name']}{'+' if cd['upgraded'] else ''}({cd['cost']})" for cd in c["hand"]
        )
        lines.append(f"Hand: [{hand}]")
        lines.append(
            f"Piles: draw {c['draw_pile_count']}, discard {c['discard_pile_count']},"
            f" exhaust {c['exhaust_pile_count']}"
        )
        if c["potions"]:
            lines.append("Potions: " + ", ".join(f"{n}" for _, n in c["potions"]))
    else:
        lines.append(f"-- {obs['screen']} --")
        lines.append(f"Deck ({len(obs['deck'])}): {', '.join(obs['deck'])}")
        if obs["relics"]:
            lines.append(f"Relics: {', '.join(obs['relics'])}")
    lines.append("Legal actions:")
    for a in obs["legal_actions"]:
        lines.append(f"  {a['id']}: {a['description']}")
    return "\n".join(lines)


def _fmt_powers(powers: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in powers.items())
