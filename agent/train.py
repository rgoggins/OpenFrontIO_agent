"""
openfront_alpha.py
===================
Dynamic, per‑turn action costs
-----------------------------
The cost of every build/weapon action now lives **only** in the game state, so
it can change over time (tech upgrades, scarcity, diplomacy penalties, etc.).

* `_can_afford()` and `apply_action()` look up `getattr(self, f"cost_of_{action}")`.
* The static constant was renamed to `BASE_ACTION_COST` and is used **only** to
  seed the initial state; you can mutate the `cost_of_*` fields whenever game
  rules demand.

Everything else (network, encoding, MCTS) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. CONSTANTS (latest spec)
# ---------------------------------------------------------------------------
BOARD_H, BOARD_W = 1000, 1000
MAX_PLAYERS_AND_NATIONS = 200

NON_SPATIAL_INPUT_STATE = [
    "gold",
    "troops",
    "population",
    "population_growth",
    "max_population",
    "troops_extended",
    "cost_of_atom_bomb",
    "cost_of_mrv",
    "cost_of_hydrogen_bomb",
    "cost_of_build_warship",
    "cost_of_build_port",
    "cost_of_build_missile_silo",
    "cost_of_build_sam_launcher",
    "cost_of_build_defense_post",
    "cost_of_build_city",
]

SPATIAL_ACTION_TYPES = [
    "atom_bomb",
    "mrv",
    "hydrogen_bomb",
    "build_warship",
    "build_port",
    "build_missile_silo",
    "build_sam_launcher",
    "build_defense_post",
    "build_city",
    "invade_ground",
    "invade_sea",
    "ally",
]

NON_SPATIAL_ACTIONS = ["set_troop_ratio", "set_attack_ratio"]

SPATIAL_ACTION_SIZE = BOARD_H * BOARD_W * len(SPATIAL_ACTION_TYPES)
ACTION_SIZE = SPATIAL_ACTION_SIZE + len(NON_SPATIAL_ACTIONS)

# ---------------------------------------------------------------------------
# Base costs used to initialise state. *After that, costs live in state*
# ---------------------------------------------------------------------------
BASE_ACTION_COST: Dict[str, int] = {
    "atom_bomb": 500,
    "mrv": 400,
    "hydrogen_bomb": 600,
    "build_warship": 300,
    "build_port": 150,
    "build_missile_silo": 200,
    "build_sam_launcher": 100,
    "build_defense_post": 75,
    "build_city": 250,
    "invade_ground": 10,
    "invade_sea": 15,
    "ally": 0,
}

# ---------------------------------------------------------------------------
# 2. ENVIRONMENT
# ---------------------------------------------------------------------------
class OpenFrontEnv:
    """Simultaneous move wrapper with dynamic action costs."""

    def __init__(self):
        self.turns = 0
        # TODO: board representation

        # --- economy & demo
        self.gold: int = 500
        self.troops: int = 0
        self.population: int = 100
        self.population_growth: int = 2
        self.max_population: int = 200
        self.troops_extended: int = 0

        # Initialise *dynamic* cost fields from base table
        for action, cost in BASE_ACTION_COST.items():
            setattr(self, f"cost_of_{action}", cost)

    # ---------------- legality ----------------------------------------
    def _action_cost(self, action_name: str) -> int:
        attr = f"cost_of_{action_name}"
        if hasattr(self, attr):
            return getattr(self, attr)
        return BASE_ACTION_COST.get(action_name, 0)

    def _can_afford(self, action_idx: int) -> bool:
        if action_idx >= SPATIAL_ACTION_SIZE:
            return True  # sliders free
        typ = SPATIAL_ACTION_TYPES[action_idx // (BOARD_H * BOARD_W)]
        return self.gold >= self._action_cost(typ)

    def legal_actions(self) -> List[int]:
        return [a for a in range(ACTION_SIZE) if self._can_afford(a)]

    # ---------------- state transition --------------------------------
    def apply_action(self, action: int) -> "OpenFrontEnv":
        ns = self.__class__()
        # copy all scalars
        for name in NON_SPATIAL_INPUT_STATE:
            setattr(ns, name, getattr(self, name))
        ns.turns = self.turns + 1

        if action < SPATIAL_ACTION_SIZE:
            typ = SPATIAL_ACTION_TYPES[action // (BOARD_H * BOARD_W)]
            cost = self._action_cost(typ)
            ns.gold -= cost
            # TODO: mutate board & secondary effects
        else:
            # TODO: implement ratio adjustments
            pass

        ns.population = min(ns.population + ns.population_growth, ns.max_population)
        # Example dynamic price inflation per turn
        # for action in SPATIAL_ACTION_TYPES:
        #     setattr(ns, f"cost_of_{action}", getattr(ns, f"cost_of_{action}") + 1)
        return ns

    # ---------------- termination & reward ---------------------------
    def is_terminal(self) -> bool:
        return self.turns >= 500

    def result(self) -> float:
        return 0.0

# ---------------------------------------------------------------------------
# 3. ENCODING & DECODING (unchanged)
# ---------------------------------------------------------------------------

def encode_state(state: OpenFrontEnv) -> torch.Tensor:
    spatial_planes = MAX_PLAYERS_AND_NATIONS + 12  # placeholder
    planes = torch.zeros(spatial_planes + len(NON_SPATIAL_INPUT_STATE), BOARD_H, BOARD_W)
    for i, name in enumerate(NON_SPATIAL_INPUT_STATE):
        planes[spatial_planes + i].fill_(float(getattr(state, name)))
    return planes


def decode_action(action_idx: int) -> Tuple[int, int, str]:
    if action_idx < SPATIAL_ACTION_SIZE:
        typ = action_idx // (BOARD_H * BOARD_W)
        cell = action_idx % (BOARD_H * BOARD_W)
        y, x = divmod(cell, BOARD_W)
        return x, y, SPATIAL_ACTION_TYPES[typ]
    return -1, -1, NON_SPATIAL_ACTIONS[action_idx - SPATIAL_ACTION_SIZE]

# ---------------------------------------------------------------------------
# 4. NETWORK (unchanged)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)

class PolicyValueNet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        k = 128
        self.stem = nn.Sequential(nn.Conv2d(in_c, k, 3, 1, 1), nn.BatchNorm2d(k), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(k) for _ in range(8)])
        self.pi_head = nn.Sequential(nn.Conv2d(k, 2, 1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2 * BOARD_H * BOARD_W, ACTION_SIZE))
        self.v_head = nn.Sequential(nn.Conv2d(k, 1, 1), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(), nn.Linear(BOARD_H * BOARD_W, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        x = self.body(self.stem(x))
        return self.pi_head(x), self.v_head(x).squeeze(-1)

# ---------------------------------------------------------------------------
# 5. MCTS & TRAINING – unchanged
# ---------------------------------------------------------------------------
