"""
Steering behaviours for player movement.

All pure functions return a desired velocity vector; apply_steering() commits
the result to player.velocity and player.position.

Based on Craig Reynolds' steering behaviours:
  seek       — full-speed straight toward target
  arrive     — seek with deceleration near target (no overshooting)
  separation — push away from nearby teammates to prevent clumping
"""
from __future__ import annotations

from typing import Optional

from sim.core.entities import Pitch, Player, Vec2


# ─── Pure steering functions ──────────────────────────────────────────────────

def seek(position: Vec2, target: Vec2, max_speed: float) -> Vec2:
    """Full-speed desired velocity toward target."""
    delta = target - position
    dist = delta.magnitude
    if dist < 1e-6:
        return Vec2(0.0, 0.0)
    return delta.normalized() * max_speed


def arrive(
    position: Vec2,
    target: Vec2,
    max_speed: float,
    slow_radius: float = 5.0,
    stop_radius: float = 0.3,
) -> Vec2:
    """
    Seek with linear deceleration inside slow_radius.
    Returns zero velocity within stop_radius to prevent jitter.
    """
    delta = target - position
    dist = delta.magnitude
    if dist <= stop_radius:
        return Vec2(0.0, 0.0)
    speed = max_speed * min(1.0, dist / slow_radius)
    return delta.normalized() * speed


def separation(
    position: Vec2,
    neighbor_positions: list[Vec2],
    min_dist: float = 3.0,
) -> Vec2:
    """
    Repulsion force away from too-close neighbours.
    Returns a vector in roughly [0, 1] magnitude per neighbour; caller scales
    by their speed budget via separation_weight in apply_steering().
    """
    force = Vec2(0.0, 0.0)
    for npos in neighbor_positions:
        delta = position - npos
        dist = delta.magnitude
        if 1e-6 < dist < min_dist:
            # Strength ramps from 0 (at min_dist) to 1 (at overlap)
            strength = (min_dist - dist) / min_dist
            force = force + delta.normalized() * strength
    return force


# ─── Integration step ─────────────────────────────────────────────────────────

def apply_steering(
    player: Player,
    desired_velocity: Vec2,
    teammates: list[Player],
    dt: float,
    separation_min_dist: float = 3.0,
    separation_weight: float = 0.35,
    pitch: Optional[Pitch] = None,
) -> None:
    """
    Combine desired_velocity with a separation correction, clamp to max speed,
    then commit to player.velocity and player.position.

    separation_weight: fraction of max_speed_mps budgeted for the correction.
    pitch: if provided, clamps the resulting position to within the field.
    """
    neighbor_positions = [tm.position for tm in teammates if tm is not player]
    sep = separation(player.position, neighbor_positions, separation_min_dist)
    sep_velocity = sep * (player.max_speed_mps * separation_weight)

    combined = desired_velocity + sep_velocity

    speed = combined.magnitude
    if speed > player.max_speed_mps:
        combined = combined * (player.max_speed_mps / speed)

    player.velocity = combined
    new_pos = player.position + combined * dt
    player.position = pitch.clamp(new_pos) if pitch else new_pos
