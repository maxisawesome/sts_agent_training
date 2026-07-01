#!/usr/bin/env python3
"""
Phase 0.5 verification: prove that combat can be driven end-to-end from Python
through the newly-bound BattleContext.

Plays a real Ironclad seed: steps the out-of-combat GameContext until the first
BATTLE, then drives the fight to completion with a trivial greedy policy
(play any affordable attack at the lowest-HP monster, otherwise end turn),
then writes the result back to the GameContext.
"""
import slaythespire as sts


def describe_state(state):
    p = state["player"]
    lines = [
        f"  turn {state['turn']} | input={state['input_state']} | outcome={state['outcome']}",
        f"  player: {p['hp']}/{p['max_hp']} hp, {p['block']} block, {p['energy']} energy"
        + (f", powers={dict(p['powers'])}" if p["powers"] else ""),
    ]
    for m in state["monsters"]:
        intent = m["intent"]
        if m.get("is_attacking"):
            intent += f" (atk {m.get('intent_damage')}x{m.get('intent_hits')})"
        pw = f" powers={dict(m['powers'])}" if m["powers"] else ""
        lines.append(
            f"  monster[{m['idx']}] {m['name']}: {m['hp']}/{m['max_hp']} hp,"
            f" {m['block']} block | intent={intent}{pw}"
        )
    hand = ", ".join(f"{c['name']}({c['cost']})" for c in state["hand"])
    lines.append(f"  hand: [{hand}]")
    lines.append(
        f"  piles: draw={state['draw_pile_count']} discard={state['discard_pile_count']}"
        f" exhaust={state['exhaust_pile_count']}"
    )
    return "\n".join(lines)


def greedy_pick(bc, actions):
    """Trivial policy: prefer playing a CARD action; else end turn / first action."""
    card_actions = [a for a in actions if a.get_action_type() == sts.ActionType.CARD]
    if card_actions:
        return card_actions[0]
    return actions[0]


def play_battle(gc, verbose=True):
    bc = sts.BattleContext()
    bc.init(gc)
    step = 0
    while not bc.is_battle_over:
        actions = sts.get_battle_actions(bc)
        if not actions:
            print("  !! no legal actions but battle not over — stuck")
            break
        if verbose and step < 6:
            print(describe_state(sts.get_battle_state(bc)))
            print("  legal:", [a.print_desc(bc) for a in actions])
        a = greedy_pick(bc, actions)
        if verbose and step < 6:
            print("  -> chose:", a.print_desc(bc), "\n")
        a.execute(bc)
        step += 1
        if step > 500:
            print("  !! step limit hit")
            break
    print(f"  battle finished in {step} actions: outcome={bc.outcome}")
    bc.exit_battle(gc)
    return bc.outcome


def main():
    seed = 1234
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    print(f"seed={seed}  start screen={gc.screen_state}  hp={gc.cur_hp}/{gc.max_hp}")

    battles = 0
    for _ in range(2000):
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        if gc.screen_state == sts.ScreenState.BATTLE:
            battles += 1
            print(f"\n=== BATTLE #{battles} on floor {gc.floor_num} ===")
            play_battle(gc, verbose=(battles == 1))
            print(f"  after battle: hp={gc.cur_hp}/{gc.max_hp}, floor={gc.floor_num}")
            if battles >= 3:
                print("\nReached 3 battles — combat is drivable from Python. Stopping.")
                break
            continue
        acts = gc.get_valid_actions()
        if not acts:
            print(f"no valid out-of-combat actions at screen {gc.screen_state}; stopping")
            break
        gc.execute_action(acts[0])

    print(f"\nDONE. battles played={battles}, final hp={gc.cur_hp}/{gc.max_hp}, "
          f"floor={gc.floor_num}, outcome={gc.outcome}")


if __name__ == "__main__":
    main()
