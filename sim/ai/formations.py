"""
Formation home zones for off-ball steering.

Each player has a "home" position derived from:
  1. Their role and slot within that role
  2. The team's current attack direction
  3. The ball's lateral (y) position — home zones shift toward the ball side

The formation data is expressed in a team-local reference frame:
  x_frac = 0 → own goal line,  1 → opponent goal line
  y_frac = 0 → bottom touchline, 1 → top touchline

Utility AI (Step 3/later) imposes a distance-from-home penalty, so this
module also exposes home_distance_penalty() for that use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sim.core.entities import Ball, Pitch, Player, Role, Vec2
from sim.core.match_state import MatchState

if TYPE_CHECKING:
    from sim.core.entities import Team


# ─── Formation definition ─────────────────────────────────────────────────────

@dataclass
class Formation:
    name: str
    # Per-role: ordered list of (x_frac, y_frac) slots.
    # Players in the same role fill slots in the order they appear in team.players.
    slots: dict[Role, list[tuple[float, float]]]

    def get_slot(self, role: Role, slot_idx: int) -> tuple[float, float]:
        role_slots = self.slots[role]
        return role_slots[min(slot_idx, len(role_slots) - 1)]


# 6v6 formation: 1 GK — 2 DEF — 2 MID — 1 FWD
FORMATION_1221 = Formation(
    name="1-2-2-1",
    slots={
        Role.GOALKEEPER: [(0.05, 0.50)],
        Role.DEFENDER:   [(0.20, 0.35), (0.20, 0.65)],
        Role.MIDFIELDER: [(0.48, 0.35), (0.48, 0.65)],
        Role.FORWARD:    [(0.72, 0.50)],
    },
)

# How strongly each role tracks the ball's lateral (y) position.
# 0 = fixed, 1 = follows ball perfectly.
_BALL_TRACK_Y: dict[Role, float] = {
    Role.GOALKEEPER: 0.06,   # barely drifts — stays near post
    Role.DEFENDER:   0.40,   # shifts strongly to ball side
    Role.MIDFIELDER: 0.50,   # mirrors ball closely
    Role.FORWARD:    0.55,   # aggressive lateral tracking
}

# Maximum lateral displacement from base y position (fraction of pitch width).
_MAX_LATERAL_FRAC: dict[Role, float] = {
    Role.GOALKEEPER: 0.07,   # ≈ 2.3 m — goal area only
    Role.DEFENDER:   0.38,   # can cover most of the width
    Role.MIDFIELDER: 0.44,
    Role.FORWARD:    0.46,
}

# Possession-based x bias (team-local fraction, 0 = own goal, 1 = opp goal).
# When team HAS the ball: push attacking line forward.
# When team is DEFENDING: compress and drop deeper.
_X_BIAS_ATTACKING: dict[Role, float] = {
    Role.GOALKEEPER:  0.04,   # slight advance to ~9% when team attacks
    Role.DEFENDER:    0.30,   # DEF to ~50% (midfield line)
    Role.MIDFIELDER:  0.18,   # MID to ~66% (attacking third)
    Role.FORWARD:     0.08,   # FWD to ~80%
}
_X_BIAS_DEFENDING: dict[Role, float] = {
    Role.GOALKEEPER:  0.00,
    Role.DEFENDER:   -0.07,   # DEF drops to ~13% (deep own half)
    Role.MIDFIELDER: -0.22,   # MID drops to ~26% (own defensive third)
    Role.FORWARD:    -0.30,   # FWD drops to ~42% (own midfield — press from midfield)
}
# Extra forward push when ball is in the attacking half (on top of possession bias).
_X_BIAS_BALL_IN_ATTACK: dict[Role, float] = {
    Role.GOALKEEPER:  0.02,   # slight additional advance when ball in attacking half
    Role.DEFENDER:    0.12,   # DEF pushed to ~62% when ball is in attack half
    Role.MIDFIELDER:  0.06,   # MID to ~72%
    Role.FORWARD:     0.00,
}


# ─── Home position computation ────────────────────────────────────────────────

def home_position(
    player: Player,
    slot_idx: int,
    pitch: Pitch,
    state: MatchState,
    ball: Ball,
    formation: Formation = FORMATION_1221,
) -> Vec2:
    """
    Compute the current home-zone position for one player.

    slot_idx: the player's ordinal within their role on their team
    (0 = first DEF in team.players, 1 = second DEF, etc.).
    """
    base_x_frac, base_y_frac = formation.get_slot(player.role, slot_idx)

    # Possession-based line height: push up when attacking, compress when defending.
    x_bias = 0.0
    if state.possession_team == player.team_id:
        x_bias = _X_BIAS_ATTACKING[player.role]
        # Additional push when ball is already in the attacking half.
        mid_x = pitch.length / 2
        ball_in_attack = (
            ball.position.x > mid_x
            if state.attack_direction[player.team_id] == 1
            else ball.position.x < mid_x
        )
        if ball_in_attack:
            x_bias += _X_BIAS_BALL_IN_ATTACK[player.role]
    elif state.possession_team is not None:
        x_bias = _X_BIAS_DEFENDING[player.role]
    adjusted_x_frac = max(0.02, min(0.95, base_x_frac + x_bias))

    # Map team-local x to world x based on attack direction
    if state.attack_direction[player.team_id] == 1:   # attacking toward high x
        world_x = pitch.length * adjusted_x_frac
    else:                                               # attacking toward low x
        world_x = pitch.length * (1.0 - adjusted_x_frac)

    # Base y in world coords
    base_y = pitch.width * base_y_frac

    # Lateral tracking: shift home zone toward the ball's y position
    mid_y = pitch.width / 2
    ball_offset = ball.position.y - mid_y       # positive = ball is in top half
    track = _BALL_TRACK_Y[player.role]
    max_shift = pitch.width * _MAX_LATERAL_FRAC[player.role]
    shift = max(-max_shift, min(max_shift, ball_offset * track))

    world_y = max(1.0, min(pitch.width - 1.0, base_y + shift))
    return Vec2(world_x, world_y)


def build_home_positions(
    team: Team,
    pitch: Pitch,
    state: MatchState,
    ball: Ball,
    formation: Formation = FORMATION_1221,
) -> dict[Player, Vec2]:
    """Returns a {player: home_position} dict for every player in the team."""
    result: dict[Player, Vec2] = {}
    for role in Role:
        for slot_idx, player in enumerate(team.get_by_role(role)):
            result[player] = home_position(player, slot_idx, pitch, state, ball, formation)
    return result


# ─── Zone penalty for utility AI ─────────────────────────────────────────────

def home_distance_penalty(
    player: Player,
    player_home: Vec2,
    max_penalty_dist: float = 25.0,
) -> float:
    """
    Returns a [0, 1] multiplier that penalises actions pulling a player
    far from their home zone.  1.0 = at home, 0.0 = max_penalty_dist away.
    Used by the utility AI to bias against e.g. defenders bombing forward.
    """
    dist = player.position.distance_to(player_home)
    return max(0.0, 1.0 - dist / max_penalty_dist)
