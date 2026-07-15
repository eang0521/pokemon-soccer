"""
Consideration functions for utility AI.
All public functions return a float in [0.0, 1.0].  Higher = more desirable.

Design rule: each function scores ONE facet of an action.  Callers combine
them via geometric_mean(), which lets a single near-zero score tank the whole
action — the intentional "fatal-flaw" behaviour from the spec.
"""
from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.entities import Player, Vec2


# ─── Response curve primitives ───────────────────────────────────────────────

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def linear(x: float, lo: float, hi: float) -> float:
    """x in [lo, hi] → [0, 1] linearly."""
    if hi == lo:
        return 0.5
    return clamp01((x - lo) / (hi - lo))

def inverse_linear(x: float, lo: float, hi: float) -> float:
    """x in [lo, hi] → [1, 0] linearly (smaller x is better)."""
    return 1.0 - linear(x, lo, hi)

def sigmoid(x: float, mid: float, k: float = 1.0) -> float:
    """Smooth S-curve centred on `mid`; `k` controls steepness."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - mid)))
    except OverflowError:
        return 0.0 if x < mid else 1.0


# ─── Score combiner ──────────────────────────────────────────────────────────

def geometric_mean(scores: list[float]) -> float:
    """
    Geometric mean of consideration scores.
    A single near-zero value tanks the whole result — an action with one fatal
    flaw cannot be rescued by high scores elsewhere.
    """
    if not scores:
        return 0.0
    product = 1.0
    for s in scores:
        product *= max(s, 1e-6)     # 1e-6 floor keeps log maths valid
    return product ** (1.0 / len(scores))


# ─── Considerations ──────────────────────────────────────────────────────────

def openness(target_pos: Vec2, opponents: list[Player]) -> float:
    """
    How uncontested is a position?
    1.0 = nearest opponent ≥ 10 m away (free).
    0.0 = opponent standing on the spot.
    """
    if not opponents:
        return 1.0
    min_dist = min(opp.position.distance_to(target_pos) for opp in opponents)
    return clamp01(min_dist / 10.0)


def forward_progress(from_pos: Vec2, to_pos: Vec2, goal_pos: Vec2) -> float:
    """
    How much closer to goal does this move bring the ball?
    0.5 = neutral (lateral pass); > 0.5 = forward; < 0.5 = backward.
    Floored at 0.15 so backward passes are low-scored but not impossible.
    """
    from_dist = from_pos.distance_to(goal_pos)
    to_dist = to_pos.distance_to(goal_pos)
    gain_m = from_dist - to_dist    # positive = toward goal
    raw = clamp01(0.5 + gain_m / 22.0)  # 22 m ≈ half the 6v6 pitch length
    return max(0.20, raw)           # 0.20 floor: backward pass is suboptimal, not impossible


def pass_lane_safety(from_pos: Vec2, to_pos: Vec2, opponents: list[Player]) -> float:
    """
    How clear is the passing lane?
    1.0 = no opponent within 3 m of the line segment.
    0.0 = opponent directly in the path.
    Uses perpendicular distance from each opponent to the pass segment.
    """
    pass_vec = to_pos - from_pos
    pass_len = pass_vec.magnitude
    if pass_len < 0.1:
        return 1.0

    direction = pass_vec / pass_len
    min_perp = float("inf")

    for opp in opponents:
        rel = opp.position - from_pos
        along = rel.dot(direction)
        if along < 0.0 or along > pass_len:
            continue    # not between passer and target
        # 2D cross-product magnitude = perpendicular distance
        perp = abs(rel.x * direction.y - rel.y * direction.x)
        min_perp = min(min_perp, perp)

    if math.isinf(min_perp):
        return 1.0
    return clamp01(min_perp / 3.0)     # fully blocked within 3 m of line


def pass_accuracy(passer: Player, target_pos: Vec2) -> float:
    """
    Likelihood of a clean delivery based on Sp. Attack vs distance.
    sp_attack 100 → comfortable up to 50 m; floored at 0.20 so even low-SpA
    players have a viable fallback pass.
    """
    dist = passer.position.distance_to(target_pos)
    max_comfortable = passer.effective_sp_attack * 0.5   # sp100 → 50 m
    raw = clamp01(1.0 - dist / max(max_comfortable, 8.0))
    return max(0.20, raw)


def shot_success_odds(
    attacker: Player,
    goalkeeper: Optional[Player],
    distance: float,
) -> float:
    """
    Combined shot quality: attacker's shot stats vs goalkeeper's save stats,
    modulated by distance.
    Attack + Sp.Attack → shot power; Def + Sp.Def → save quality.
    Distance factor: 0 m = full power, 60 m = half power.
    """
    if goalkeeper is None:
        matchup = 0.60      # no keeper present, attacker-favoured default
    else:
        shot = (attacker.effective_attack + attacker.effective_sp_attack) / 2.0
        save = (goalkeeper.effective_defense + goalkeeper.effective_sp_defense) / 2.0
        matchup = shot / (shot + save)

    dist_factor = clamp01(1.0 - distance / 60.0)
    return clamp01(matchup * (0.5 + 0.5 * dist_factor))


def shot_angle_quality(ball_pos: Vec2, goal_center: Vec2, pitch_width: float) -> float:
    """
    How central is the shooting angle?
    1.0 = dead centre; 0.0 = on the touchline (0° angle).
    """
    lateral_offset = abs(ball_pos.y - goal_center.y)
    return clamp01(1.0 - lateral_offset / (pitch_width / 2.0))


def shot_composure(carrier: Player, opponents: list[Player]) -> float:
    """
    Reduces shot quality when the carrier is under immediate pressure.
    Full composure ≥ 5 m away; drops to 0.30 when an opponent is within 0.5 m.
    """
    if not opponents:
        return 1.0
    nearest = min(o.position.distance_to(carrier.position) for o in opponents)
    if nearest >= 5.0:
        return 1.0
    if nearest <= 0.5:
        return 0.30
    return 0.30 + 0.70 * ((nearest - 0.5) / 4.5)


def shot_lane_pressure(
    shooter_pos: "Vec2",
    goal_pos: "Vec2",
    opponents: "list[Player]",
) -> float:
    """
    How clear is the shooting lane from defenders?
    1.0 = no defender in the path to goal.
    Drops toward 0.05 when a defender stands directly between shooter and goal.
    Acts as a fatal-flaw factor in the geometric mean: a body in the lane
    makes passing almost always the better choice.
    """
    if not opponents:
        return 1.0
    shoot_vec = goal_pos - shooter_pos
    shoot_len = shoot_vec.magnitude
    if shoot_len < 0.1:
        return 1.0
    direction = shoot_vec / shoot_len
    min_perp = float("inf")
    for opp in opponents:
        rel = opp.position - shooter_pos
        along = rel.dot(direction)
        if along < 0.5 or along > shoot_len:   # only between shooter and goal
            continue
        perp = abs(rel.x * direction.y - rel.y * direction.x)
        min_perp = min(min_perp, perp)
    if math.isinf(min_perp):
        return 1.0
    # 0.05 when defender is directly in line; 1.0 at ≥ 5 m lateral clearance
    return max(0.05, clamp01(min_perp / 5.0))


def dribble_matchup(dribbler: Player, nearest_defender: Optional[Player]) -> float:
    """
    Dribbler's chance of getting past the nearest defender.
    Speed is the primary factor; defender's Defense adds resistance.
    0.8 if no defender (open field — good but not certain).
    """
    if nearest_defender is None:
        return 0.80
    atk = dribbler.effective_speed
    # Defender resists with a blend of speed (tracking) and defense (tackling)
    dfs = nearest_defender.effective_speed * 0.6 + nearest_defender.effective_defense * 0.4
    return clamp01(atk / (atk + dfs))
