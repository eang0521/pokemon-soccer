"""
Utility AI decision-making for the ball carrier.

Pipeline per carrier tick:
  1. generate_candidates()  — what actions are available?
  2. score_candidate()      — score each via geometric_mean of considerations
  3. weighted_select()      — weighted-random pick from top-N (avoids argmax robotics)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from sim.core.entities import Ball, Pitch, Player, Role, Vec2
from sim.core.match_state import MatchState
from sim.ai.considerations import (
    clamp01,
    dribble_matchup,
    forward_progress,
    geometric_mean,
    openness,
    pass_accuracy,
    pass_lane_safety,
    shot_angle_quality,
    shot_composure,
    shot_lane_pressure,
    shot_success_odds,
)


# ─── Action candidate ────────────────────────────────────────────────────────

@dataclass
class ActionCandidate:
    action_type: str                   # "shoot" | "pass" | "dribble"
    target: Optional[Player] = None   # populated for "pass"
    score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.target:
            return f"pass → {self.target.name}"
        return self.action_type

    def __repr__(self) -> str:
        return f"ActionCandidate({self.label}, score={self.score:.3f})"


# ─── Decider ─────────────────────────────────────────────────────────────────

class Decider:
    """
    Scores and selects one action for the ball carrier.
    Stateless between calls; instantiate once and call decide() per frame.
    """

    TOP_N = 3                   # candidates in the weighted-random pool
    DRIBBLE_LOOKAHEAD_M = 5.0   # metres to look ahead when rating dribble space

    def __init__(self, pitch: Pitch):
        self.pitch = pitch

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(
        self,
        carrier: Player,
        ball: Ball,
        teammates: list[Player],
        opponents: list[Player],
        state: MatchState,
        forbidden: frozenset[str] = frozenset(),
    ) -> ActionCandidate:
        """
        Full pipeline: generate → score → weighted select.
        forbidden: action_type strings to exclude from consideration (e.g. "shoot" on cooldown).
        Returns the chosen ActionCandidate with breakdown populated.
        """
        candidates = self._generate_candidates(carrier, teammates)
        if forbidden:
            candidates = [c for c in candidates if c.action_type not in forbidden]
        for c in candidates:
            self._score_candidate(c, carrier, ball, teammates, opponents, state)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return self._weighted_select(candidates)

    def all_scored(
        self,
        carrier: Player,
        ball: Ball,
        teammates: list[Player],
        opponents: list[Player],
        state: MatchState,
    ) -> list[ActionCandidate]:
        """
        Returns all candidates sorted by score descending — useful for tracing.
        Does not perform selection.
        """
        candidates = self._generate_candidates(carrier, teammates)
        for c in candidates:
            self._score_candidate(c, carrier, ball, teammates, opponents, state)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ── Candidate generation ──────────────────────────────────────────────────

    def _generate_candidates(
        self, carrier: Player, teammates: list[Player]
    ) -> list[ActionCandidate]:
        candidates: list[ActionCandidate] = [ActionCandidate("shoot")]
        for tm in teammates:
            if tm is not carrier:
                candidates.append(ActionCandidate("pass", target=tm))
        candidates.append(ActionCandidate("dribble"))
        return candidates

    # ── Scoring dispatch ──────────────────────────────────────────────────────

    def _score_candidate(
        self,
        c: ActionCandidate,
        carrier: Player,
        ball: Ball,
        teammates: list[Player],
        opponents: list[Player],
        state: MatchState,
    ) -> None:
        if c.action_type == "shoot":
            self._score_shoot(c, carrier, opponents, state)
        elif c.action_type == "pass":
            self._score_pass(c, carrier, opponents, state)
        elif c.action_type == "dribble":
            self._score_dribble(c, carrier, opponents, state)

    # ── Shoot ─────────────────────────────────────────────────────────────────

    # Per-role shooting range (m): GK never shoots; DEF only close-in;
    # MID and FWD share the full 10m range — the bias difference drives selection.
    _ROLE_SHOOT_ZONE_M: dict[Role, float] = {
        Role.GOALKEEPER: 0.0,
        Role.DEFENDER:   12.0,
        Role.MIDFIELDER: 20.0,
        Role.FORWARD:    25.0,
    }

    # Role-based shot bias: reflects real positional shooting responsibility.
    _ROLE_SHOT_BIAS: dict[Role, float] = {
        Role.GOALKEEPER: 0.0,
        Role.DEFENDER:   0.1,
        Role.MIDFIELDER: 0.65,
        Role.FORWARD:    1.0,
    }

    def _score_shoot(
        self,
        c: ActionCandidate,
        carrier: Player,
        opponents: list[Player],
        state: MatchState,
    ) -> None:
        goal = Vec2(state.attacking_goal_x(carrier.team_id), self.pitch.width / 2)
        dist = carrier.position.distance_to(goal)
        gk = self._find_goalkeeper(opponents)

        odds     = shot_success_odds(carrier, gk, dist)
        angle    = shot_angle_quality(carrier.position, goal, self.pitch.width)
        shoot_zone = self._ROLE_SHOOT_ZONE_M.get(carrier.role, 10.0)
        zone     = clamp01(1.0 - dist / shoot_zone) if shoot_zone > 0.0 else 0.0
        composure = shot_composure(carrier, opponents)
        lane     = shot_lane_pressure(carrier.position, goal, opponents)
        role_bias = self._ROLE_SHOT_BIAS.get(carrier.role, 1.0)
        c.breakdown = {"shot_odds": odds, "angle": angle, "zone": zone,
                       "composure": composure, "lane": lane, "role_bias": role_bias}
        c.score = geometric_mean([odds, angle, zone, composure, lane]) * role_bias

    # ── Pass ──────────────────────────────────────────────────────────────────

    def _score_pass(
        self,
        c: ActionCandidate,
        carrier: Player,
        opponents: list[Player],
        state: MatchState,
    ) -> None:
        target = c.target
        goal = Vec2(state.attacking_goal_x(carrier.team_id), self.pitch.width / 2)

        open_q = openness(target.position, opponents)
        prog = forward_progress(carrier.position, target.position, goal)
        safety = pass_lane_safety(carrier.position, target.position, opponents)
        accuracy = pass_accuracy(carrier, target.position)

        c.breakdown = {
            "openness": open_q,
            "progress": prog,
            "lane_safety": safety,
            "accuracy": accuracy,
        }
        c.score = geometric_mean([open_q, prog, safety, accuracy])

    # ── Dribble ───────────────────────────────────────────────────────────────

    def _score_dribble(
        self,
        c: ActionCandidate,
        carrier: Player,
        opponents: list[Player],
        state: MatchState,
    ) -> None:
        goal = Vec2(state.attacking_goal_x(carrier.team_id), self.pitch.width / 2)
        toward_goal = (goal - carrier.position).normalized()
        lookahead = carrier.position + toward_goal * self.DRIBBLE_LOOKAHEAD_M

        space = openness(lookahead, opponents)
        prog = forward_progress(carrier.position, lookahead, goal)
        matchup = dribble_matchup(carrier, self._nearest_opponent(carrier, opponents))

        c.breakdown = {"space_ahead": space, "progress": prog, "matchup": matchup}
        c.score = geometric_mean([space, prog, matchup])

    # ── Selection ─────────────────────────────────────────────────────────────

    def _weighted_select(self, ranked: list[ActionCandidate]) -> ActionCandidate:
        """Weighted random among top-N to avoid fully deterministic play."""
        pool = ranked[: self.TOP_N]
        if len(pool) == 1:
            return pool[0]
        weights = [max(c.score, 1e-6) for c in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_goalkeeper(opponents: list[Player]) -> Optional[Player]:
        return next((p for p in opponents if p.role == Role.GOALKEEPER), None)

    @staticmethod
    def _nearest_opponent(player: Player, opponents: list[Player]) -> Optional[Player]:
        if not opponents:
            return None
        return min(opponents, key=lambda o: o.position.distance_to(player.position))
