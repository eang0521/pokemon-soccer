"""
Ball physics: trajectory, friction deceleration, goal-line prediction.

Functions
---------
pass_speed(passer)        → initial m/s for a pass
shot_speed(attacker)      → initial m/s for a shot
kick(ball, target, speed) → set ball velocity toward target
step(ball, dt)            → advance ball; apply friction
goal_crossing(ball, x)    → predict (y, t) where ball crosses x = goal_x, or None
can_reach(pos, spd, target, budget, reach) → bool
"""
from __future__ import annotations

import math
from typing import Optional

from sim.core.entities import Ball, Player, Vec2

FRICTION_DECEL = 2.5   # m/s² — 5.0 stalled passes; 2.5 lets a 13 m/s pass reach ~34 m
MIN_SPEED = 0.1        # m/s; below this the ball is treated as stationary


def pass_speed(passer: Player) -> float:
    """Initial ball speed for a pass: [10, 20] m/s based on passer's sp_attack."""
    return 10.0 + (passer.stats.sp_attack / 150.0) * 10.0


def shot_speed(attacker: Player) -> float:
    """Initial ball speed for a shot: [15, 30] m/s based on attacker's attack stat."""
    return 15.0 + (attacker.stats.attack / 150.0) * 15.0


def kick(ball: Ball, target_pos: Vec2, speed: float) -> None:
    """Set ball velocity directed toward target_pos at the given speed."""
    dx = target_pos.x - ball.position.x
    dy = target_pos.y - ball.position.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        ball.velocity = Vec2(speed, 0.0)
    else:
        ball.velocity = Vec2(dx / dist * speed, dy / dist * speed)


def step(ball: Ball, dt: float) -> None:
    """Advance ball position and apply constant-magnitude friction deceleration."""
    spd = math.hypot(ball.velocity.x, ball.velocity.y)
    if spd < MIN_SPEED:
        ball.velocity = Vec2()
        return

    ball.position = Vec2(
        ball.position.x + ball.velocity.x * dt,
        ball.position.y + ball.velocity.y * dt,
    )
    new_spd = max(0.0, spd - FRICTION_DECEL * dt)
    if new_spd < MIN_SPEED:
        ball.velocity = Vec2()
    else:
        scale = new_spd / spd
        ball.velocity = Vec2(ball.velocity.x * scale, ball.velocity.y * scale)


def goal_crossing(ball: Ball, goal_x: float) -> Optional[tuple[float, float]]:
    """
    Predict where/when the ball's current trajectory will cross x = goal_x.

    Uses the continuous friction model matching step():
      x(t) = x0 + vx*t - (vx/spd)*(FRICTION_DECEL/2)*t²

    Returns (y_at_crossing, time_seconds) or None if the ball won't reach goal_x.
    """
    vx = ball.velocity.x
    vy = ball.velocity.y
    spd = math.hypot(vx, vy)
    if spd < MIN_SPEED:
        return None

    dx = goal_x - ball.position.x
    if abs(dx) < 1e-6 or dx * vx <= 0:
        return None  # already there, or moving away

    # Quadratic: a*t² + b*t + c = 0  where
    #   a = (vx/spd) * FRICTION_DECEL / 2
    #   b = -vx
    #   c = dx
    a = (vx / spd) * (FRICTION_DECEL / 2.0)
    b = -vx
    c = dx
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None  # friction kills ball before it reaches goal_x

    t_stop = spd / FRICTION_DECEL
    sqrt_disc = math.sqrt(disc)
    t_cross: Optional[float] = None
    for t_cand in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
        if 1e-9 < t_cand <= t_stop + 1e-6:
            if t_cross is None or t_cand < t_cross:
                t_cross = t_cand

    if t_cross is None:
        return None

    ay = (vy / spd) * (FRICTION_DECEL / 2.0)
    y_cross = ball.position.y + vy * t_cross - ay * t_cross * t_cross

    return (y_cross, t_cross)


def can_reach(
    from_pos: Vec2,
    speed: float,
    target_pos: Vec2,
    time_budget: float,
    reach: float = 1.5,
) -> bool:
    """True if a player at from_pos (moving at speed m/s) can reach within reach m of target_pos."""
    needed = from_pos.distance_to(target_pos) - reach
    return needed <= speed * time_budget
