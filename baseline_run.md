# Baseline run — stub policies (no LLM)

These are the throwaway stub policies that exercise the harness. They make deliberately
dumb choices (the heuristic always takes the first overworld option and only crudely
blocks/attacks in combat). They exist to prove the pipeline runs end to end and to give
the LLM something to beat — they are NOT meant to play well.

## HeuristicPolicy — 10 seeds

| seed | outcome | floor reached | final HP | steps |
|---|---|---|---|---|
| 1234 | PLAYER_LOSS | 12 | 0/80 | 218 |
| 5678 | PLAYER_LOSS | 11 | 0/80 | 152 |
| 99 | PLAYER_LOSS | 12 | 0/80 | 253 |
| 42 | PLAYER_LOSS | 16 | 0/80 | 229 |
| 7 | PLAYER_LOSS | 6 | 0/80 | 123 |
| 13 | PLAYER_LOSS | 6 | 0/80 | 96 |
| 21 | PLAYER_LOSS | 4 | 0/80 | 109 |
| 100 | PLAYER_LOSS | 13 | 0/80 | 243 |
| 200 | PLAYER_LOSS | 5 | 0/80 | 147 |
| 777 | PLAYER_LOSS | 6 | 0/80 | 150 |

**wins 0/10 · avg floor 9.1**

## RandomPolicy — 10 seeds

| seed | outcome | floor reached | final HP | steps |
|---|---|---|---|---|
| 1234 | PLAYER_LOSS | 5 | 0/80 | 87 |
| 5678 | PLAYER_LOSS | 4 | 0/77 | 56 |
| 99 | PLAYER_LOSS | 2 | 0/80 | 47 |
| 42 | PLAYER_LOSS | 6 | 0/85 | 74 |
| 7 | PLAYER_LOSS | 6 | 0/80 | 77 |
| 13 | PLAYER_LOSS | 5 | 0/80 | 54 |
| 21 | PLAYER_LOSS | 5 | 0/80 | 82 |
| 100 | PLAYER_LOSS | 6 | 0/80 | 75 |
| 200 | PLAYER_LOSS | 6 | 0/80 | 86 |
| 777 | PLAYER_LOSS | 3 | 0/85 | 65 |

**wins 0/10 · avg floor 4.8**

## Sample play-by-play (HeuristicPolicy, seed 1234)

Each block is exactly what a policy (or LLM) sees, plus the action the stub chose.
Truncated to the first 24 decision points.

```
Floor 0 (Act 1) | HP 80/80 | Gold 99
-- EVENT_SCREEN --
Deck (10): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED
Relics: BURNING_BLOOD
Legal actions:
  0: Choose a colorless card to obtain.
  1: Max Hp +10%.
  2: Transform two cards in your cards.	Obtain a curse.
  3: Obtain a random boss relic.	Lose your starter relic.

>>> CHOSE 0: Choose a colorless card to obtain.
```

```
Floor 0 (Act 1) | HP 80/80 | Gold 99
-- REWARDS --
Deck (10): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED
Relics: BURNING_BLOOD
Legal actions:
  0: take Forethought
  1: take Finesse
  2: take Enlightenment
  3: skip rewards

>>> CHOSE 0: take Forethought
```

```
Floor 0 (Act 1) | HP 80/80 | Gold 99
-- MAP_SCREEN --
Deck (11): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, FORETHOUGHT, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED
Relics: BURNING_BLOOD
Legal actions:
  0: MONSTER
  1: MONSTER

>>> CHOSE 0: MONSTER
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 3 energy
  [0] JAW_WORM: 40/40 HP | intent: attack 11x1
Hand: [Strike(1), Defend(1), Strike(1), Strike(1), Forethought(0)]
Piles: draw 6, discard 0, exhaust 0
Legal actions:
  0: { use card (0) (Strike,4,1,1) -> (0) JAW_WORM }
  1: { use card (1) (Defend,8,1,1) }
  2: { use card (2) (Strike,0,1,1) -> (0) JAW_WORM }
  3: { use card (3) (Strike,3,1,1) -> (0) JAW_WORM }
  4: { use card (4) (Forethought,10,0,0) }
  5: { end turn }

>>> CHOSE 1: { use card (1) (Defend,8,1,1) }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 5 block, 2 energy
  [0] JAW_WORM: 40/40 HP | intent: attack 11x1
Hand: [Strike(1), Strike(1), Strike(1), Forethought(0)]
Piles: draw 6, discard 1, exhaust 0
Legal actions:
  0: { use card (0) (Strike,4,1,1) -> (0) JAW_WORM }
  1: { use card (1) (Strike,0,1,1) -> (0) JAW_WORM }
  2: { use card (2) (Strike,3,1,1) -> (0) JAW_WORM }
  3: { use card (3) (Forethought,10,0,0) }
  4: { end turn }

>>> CHOSE 0: { use card (0) (Strike,4,1,1) -> (0) JAW_WORM }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 5 block, 1 energy
  [0] JAW_WORM: 34/40 HP | intent: attack 11x1
Hand: [Strike(1), Strike(1), Forethought(0)]
Piles: draw 6, discard 2, exhaust 0
Legal actions:
  0: { use card (0) (Strike,0,1,1) -> (0) JAW_WORM }
  1: { use card (1) (Strike,3,1,1) -> (0) JAW_WORM }
  2: { use card (2) (Forethought,10,0,0) }
  3: { end turn }

>>> CHOSE 0: { use card (0) (Strike,0,1,1) -> (0) JAW_WORM }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 5 block, 0 energy
  [0] JAW_WORM: 28/40 HP | intent: attack 11x1
Hand: [Strike(1), Forethought(0)]
Piles: draw 6, discard 3, exhaust 0
Legal actions:
  0: { use card (1) (Forethought,10,0,0) }
  1: { end turn }

>>> CHOSE 0: { use card (1) (Forethought,10,0,0) }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 1, PLAYER_NORMAL) --
You: 74/80 HP, 0 block, 3 energy
  [0] JAW_WORM: 28/40 HP | intent: JAW_WORM_BELLOW
Hand: [Defend(1), Bash(2), Strike(1), Strike(1), Defend(1)]
Piles: draw 2, discard 4, exhaust 0
Legal actions:
  0: { use card (0) (Defend,5,1,1) }
  1: { use card (1) (Bash,9,2,2) -> (0) JAW_WORM }
  2: { use card (2) (Strike,1,1,1) -> (0) JAW_WORM }
  3: { use card (3) (Strike,2,1,1) -> (0) JAW_WORM }
  4: { use card (4) (Defend,7,1,1) }
  5: { end turn }

>>> CHOSE 1: { use card (1) (Bash,9,2,2) -> (0) JAW_WORM }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 1, PLAYER_NORMAL) --
You: 74/80 HP, 0 block, 1 energy
  [0] JAW_WORM: 20/40 HP | intent: JAW_WORM_BELLOW | vulnerable 2
Hand: [Defend(1), Strike(1), Strike(1), Defend(1)]
Piles: draw 2, discard 5, exhaust 0
Legal actions:
  0: { use card (0) (Defend,5,1,1) }
  1: { use card (1) (Strike,1,1,1) -> (0) JAW_WORM }
  2: { use card (2) (Strike,2,1,1) -> (0) JAW_WORM }
  3: { use card (3) (Defend,7,1,1) }
  4: { end turn }

>>> CHOSE 1: { use card (1) (Strike,1,1,1) -> (0) JAW_WORM }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 2, PLAYER_NORMAL) --
You: 74/80 HP, 0 block, 3 energy
  [0] JAW_WORM: 11/40 HP, 6 block | intent: attack 14x1 | strength 3, vulnerable 1
Hand: [Defend(1), Strike(1), Strike(1), Strike(1), Defend(1)]
Piles: draw 6, discard 0, exhaust 0
Legal actions:
  0: { use card (0) (Defend,6,1,1) }
  1: { use card (1) (Strike,3,1,1,f) -> (0) JAW_WORM }
  2: { use card (2) (Strike,4,1,1) -> (0) JAW_WORM }
  3: { use card (3) (Strike,1,1,1) -> (0) JAW_WORM }
  4: { use card (4) (Defend,8,1,1) }
  5: { end turn }

>>> CHOSE 0: { use card (0) (Defend,6,1,1) }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 2, PLAYER_NORMAL) --
You: 74/80 HP, 5 block, 2 energy
  [0] JAW_WORM: 11/40 HP, 6 block | intent: attack 14x1 | strength 3, vulnerable 1
Hand: [Strike(1), Strike(1), Strike(1), Defend(1)]
Piles: draw 6, discard 1, exhaust 0
Legal actions:
  0: { use card (0) (Strike,3,1,1,f) -> (0) JAW_WORM }
  1: { use card (1) (Strike,4,1,1) -> (0) JAW_WORM }
  2: { use card (2) (Strike,1,1,1) -> (0) JAW_WORM }
  3: { use card (3) (Defend,8,1,1) }
  4: { end turn }

>>> CHOSE 3: { use card (3) (Defend,8,1,1) }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 2, PLAYER_NORMAL) --
You: 74/80 HP, 10 block, 1 energy
  [0] JAW_WORM: 11/40 HP, 6 block | intent: attack 14x1 | strength 3, vulnerable 1
Hand: [Strike(1), Strike(1), Strike(1)]
Piles: draw 6, discard 2, exhaust 0
Legal actions:
  0: { use card (0) (Strike,3,1,1,f) -> (0) JAW_WORM }
  1: { use card (1) (Strike,4,1,1) -> (0) JAW_WORM }
  2: { use card (2) (Strike,1,1,1) -> (0) JAW_WORM }
  3: { end turn }

>>> CHOSE 0: { use card (0) (Strike,3,1,1,f) -> (0) JAW_WORM }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- COMBAT (turn 2, PLAYER_NORMAL) --
You: 74/80 HP, 10 block, 1 energy
  [0] JAW_WORM: 8/40 HP | intent: attack 14x1 | strength 3, vulnerable 1
Hand: [Strike(1), Strike(1)]
Piles: draw 6, discard 3, exhaust 0
Legal actions:
  0: { use card (0) (Strike,4,1,1) -> (0) JAW_WORM }
  1: { use card (1) (Strike,1,1,1) -> (0) JAW_WORM }
  2: { end turn }

>>> CHOSE 0: { use card (0) (Strike,4,1,1) -> (0) JAW_WORM }
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 99
-- REWARDS --
Deck (11): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, FORETHOUGHT, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED
Relics: BURNING_BLOOD
Legal actions:
  0: take 15g
  1: take Sword Boomerang
  2: take Cleave
  3: take Whirlwind
  4: skip rewards

>>> CHOSE 0: take 15g
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 114
-- REWARDS --
Deck (11): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, FORETHOUGHT, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED
Relics: BURNING_BLOOD
Legal actions:
  0: take Sword Boomerang
  1: take Cleave
  2: take Whirlwind
  3: skip rewards

>>> CHOSE 0: take Sword Boomerang
```

```
Floor 1 (Act 1) | HP 80/80 | Gold 114
-- MAP_SCREEN --
Deck (12): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, FORETHOUGHT, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, SWORD_BOOMERANG
Relics: BURNING_BLOOD
Legal actions:
  0: EVENT
  1: MONSTER

>>> CHOSE 0: EVENT
```

```
Floor 2 (Act 1) | HP 80/80 | Gold 114
-- EVENT_SCREEN --
Deck (12): BASH, DEFEND_RED, DEFEND_RED, DEFEND_RED, DEFEND_RED, FORETHOUGHT, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, STRIKE_RED, SWORD_BOOMERANG
Relics: BURNING_BLOOD
Legal actions:
  0: [Agree] Gain 175 Gold. Become Cursed - Doubt.
  1: [Disagree] Nothing happens.

>>> CHOSE 0: [Agree] Gain 175 Gold. Become Cursed - Doubt.
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 3 energy
  [0] ACID_SLIME_S: 9/9 HP | intent: ACID_SLIME_S_LICK
  [1] SPIKE_SLIME_M: 31/31 HP | intent: SPIKE_SLIME_M_LICK
Hand: [Strike(1), Defend(1), Defend(1), Forethought(0), Strike(1)]
Piles: draw 8, discard 0, exhaust 0
Legal actions:
  0: { use card (0) (Strike,1,1,1) -> (0) ACID_SLIME_S }
  1: { use card (0) (Strike,1,1,1) -> (1) SPIKE_SLIME_M }
  2: { use card (1) (Defend,8,1,1) }
  3: { use card (2) (Defend,6,1,1) }
  4: { use card (3) (Forethought,10,0,0) }
  5: { use card (4) (Strike,4,1,1) -> (0) ACID_SLIME_S }
  6: { use card (4) (Strike,4,1,1) -> (1) SPIKE_SLIME_M }
  7: { end turn }

>>> CHOSE 0: { use card (0) (Strike,1,1,1) -> (0) ACID_SLIME_S }
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 2 energy
  [0] ACID_SLIME_S: 3/9 HP | intent: ACID_SLIME_S_LICK
  [1] SPIKE_SLIME_M: 31/31 HP | intent: SPIKE_SLIME_M_LICK
Hand: [Defend(1), Defend(1), Forethought(0), Strike(1)]
Piles: draw 8, discard 1, exhaust 0
Legal actions:
  0: { use card (0) (Defend,8,1,1) }
  1: { use card (1) (Defend,6,1,1) }
  2: { use card (2) (Forethought,10,0,0) }
  3: { use card (3) (Strike,4,1,1) -> (0) ACID_SLIME_S }
  4: { use card (3) (Strike,4,1,1) -> (1) SPIKE_SLIME_M }
  5: { end turn }

>>> CHOSE 3: { use card (3) (Strike,4,1,1) -> (0) ACID_SLIME_S }
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 1 energy
  [1] SPIKE_SLIME_M: 31/31 HP | intent: SPIKE_SLIME_M_LICK
Hand: [Defend(1), Defend(1), Forethought(0)]
Piles: draw 8, discard 2, exhaust 0
Legal actions:
  0: { use card (0) (Defend,8,1,1) }
  1: { use card (1) (Defend,6,1,1) }
  2: { use card (2) (Forethought,10,0,0) }
  3: { end turn }

>>> CHOSE 0: { use card (0) (Defend,8,1,1) }
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 0, PLAYER_NORMAL) --
You: 80/80 HP, 5 block, 0 energy
  [1] SPIKE_SLIME_M: 31/31 HP | intent: SPIKE_SLIME_M_LICK
Hand: [Defend(1), Forethought(0)]
Piles: draw 8, discard 3, exhaust 0
Legal actions:
  0: { use card (1) (Forethought,10,0,0) }
  1: { end turn }

>>> CHOSE 0: { use card (1) (Forethought,10,0,0) }
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 1, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 3 energy | powers: Frail 1
  [1] SPIKE_SLIME_M: 31/31 HP | intent: SPIKE_SLIME_M_LICK
Hand: [Defend(1), Defend(1), Sword Boomerang(1), Bash(2), Strike(1)]
Piles: draw 4, discard 4, exhaust 0
Legal actions:
  0: { use card (0) (Defend,5,1,1) }
  1: { use card (1) (Defend,7,1,1) }
  2: { use card (2) (Sword Boomerang,11,1,1) }
  3: { use card (3) (Bash,9,2,2) -> (1) SPIKE_SLIME_M }
  4: { use card (4) (Strike,2,1,1) -> (1) SPIKE_SLIME_M }
  5: { end turn }

>>> CHOSE 3: { use card (3) (Bash,9,2,2) -> (1) SPIKE_SLIME_M }
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 1, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 1 energy | powers: Frail 1
  [1] SPIKE_SLIME_M: 23/31 HP | intent: SPIKE_SLIME_M_LICK | vulnerable 2
Hand: [Defend(1), Defend(1), Sword Boomerang(1), Strike(1)]
Piles: draw 4, discard 5, exhaust 0
Legal actions:
  0: { use card (0) (Defend,5,1,1) }
  1: { use card (1) (Defend,7,1,1) }
  2: { use card (2) (Sword Boomerang,11,1,1) }
  3: { use card (3) (Strike,2,1,1) -> (1) SPIKE_SLIME_M }
  4: { end turn }

>>> CHOSE 3: { use card (3) (Strike,2,1,1) -> (1) SPIKE_SLIME_M }
```

```
Floor 3 (Act 1) | HP 80/80 | Gold 289
-- COMBAT (turn 2, PLAYER_NORMAL) --
You: 80/80 HP, 0 block, 3 energy | powers: Frail 1
  [1] SPIKE_SLIME_M: 14/31 HP | intent: attack 8x1 | vulnerable 1
Hand: [Strike(1), Strike(1), Doubt(-3), Defend(1), Bash(2)]
Piles: draw 8, discard 0, exhaust 0
Legal actions:
  0: { use card (0) (Strike,0,1,1) -> (1) SPIKE_SLIME_M }
  1: { use card (1) (Strike,3,1,1) -> (1) SPIKE_SLIME_M }
  2: { use card (3) (Defend,6,1,1,f) }
  3: { use card (4) (Bash,9,2,2) -> (1) SPIKE_SLIME_M }
  4: { end turn }

>>> CHOSE 2: { use card (3) (Defend,6,1,1,f) }
```

... game ended: outcome=PLAYER_LOSS, floor 12, hp 0/80, 218 steps
