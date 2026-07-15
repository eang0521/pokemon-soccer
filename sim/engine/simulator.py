"""
Main match simulation tick-loop.

Simulator owns the game clock and coordinates:
  - Phase management (kickoff, restarts, half-time, full-time)
  - Decision-making (carrier every ~0.3 s, off-ball every ~0.8 s)
  - Movement (steering toward formation home zone or carrier)
  - Ball physics (passes and shots travel physically; friction deceleration)
  - Event logging

Usage:
    sim = Simulator(teams=[team0, team1], pitch=Pitch(), seed=42)
    event_log = sim.run(verbose=True)
"""
from __future__ import annotations

import math
import random
from typing import Optional

from sim.core.entities import Ball, Pitch, Player, Role, Team, Vec2
from sim.core.match_state import GamePhase, MatchConfig, MatchState
from sim.core.rules import Rules
from sim.ai.decision import Decider
from sim.ai.formations import build_home_positions
from sim.ai.steering import apply_steering, arrive, seek
from sim.engine import physics
from sim.engine.events import EventLog, EventType


# ─── Timing / physics constants ──────────────────────────────────────────────

DT = 0.1                      # seconds per tick (tick_rate = 10 Hz)
CARRIER_DECIDE_TICKS = 10     # re-decide every 1.0 sim-seconds when carrying
OFFBALL_DECIDE_TICKS = 10     # off-ball players re-evaluate home zone every 1.0 s
TACKLE_RANGE_M = 1.2          # metres within which a tackle can be attempted
TACKLE_INTERVAL_TICKS = 5     # cooldown ticks between tackle attempts per player
PICKUP_RANGE_M = 1.5          # swept-sphere radius for loose-ball pickup
# Pressing count is now dynamic — see _n_pressers()
SHOT_COOLDOWN_TICKS = 3000    # ticks a player must wait between shots (~5 sim-minutes)
GK_CLAIM_RANGE_M = 2.5        # GK can claim any loose ball within this range in their box
GK_PA_DEPTH_M = 9.5           # depth (m from goal line) of the GK's priority zone (≈10 yd)
GK_SHOT_SAVE_RANGE_M = 1.85  # extended dive range for GK when a shot is in flight
PASS_TARGET_RANGE_M = 3.5     # intended receiver's extended pickup radius


class Simulator:
    """Runs a complete 90-minute Pokémon soccer match, tick by tick."""

    def __init__(
        self,
        teams: list[Team],
        pitch: Optional[Pitch] = None,
        config: Optional[MatchConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)

        self.teams = teams          # teams[0] = home, teams[1] = away
        self.pitch = pitch or Pitch()
        self.config = config or MatchConfig()
        self.config.pitch_length = self.pitch.length   # keep match state in sync with pitch

        self.state = MatchState(config=self.config)
        self.ball = Ball()
        self.rules = Rules(self.pitch)
        self.decider = Decider(self.pitch)
        self.events = EventLog()

        all_players = self._all_players()
        self._decide_timer:   dict[Player, int] = {p: 0 for p in all_players}
        self._tackle_timer:   dict[Player, int] = {p: 0 for p in all_players}
        self._shot_timer:     dict[Player, int] = {p: 0 for p in all_players}
        self._gk_hold_timer:  dict[Player, int] = {p: 0 for p in all_players}
        self._prev_carrier:   Optional[Player]  = None
        self._move_target: dict[Player, Vec2] = {
            p: Vec2() for p in all_players
        }

        # Ball-kick tracking: used by _check_loose_ball to classify events
        self._last_kicked_by: Optional[Player] = None
        self._kick_target: Optional[Player] = None   # pass target; None for shots
        self._kick_is_shot: bool = False
        self._shot_beat_gk: bool = False   # True after GK dives and misses a save roll
        self._ball_prev_pos: Vec2 = Vec2()  # position before physics.step; used for swept pickup

        # Counter-press: team that just lost possession presses hard for N ticks
        self._counter_press_ticks: int = 0
        self._counter_press_team: int = -1

        # Possession and goal tracking
        self._possession_ticks: list[int] = [0, 0]
        self._last_passer: dict[Player, Optional[Player]] = {p: None for p in all_players}
        self._goals: list[dict] = []
        self._pending_frame_goals: list[dict] = []

        # Penalty shootout results (populated if the match goes to penalties)
        self._penalty_kicks: list[dict] = []
        self._penalty_winner: Optional[int] = None

        # Counterattack window: team that just won possession attacks at pace
        self._counterattack_team: int = -1
        self._counterattack_ticks: int = 0
        self._prev_possession_team: Optional[int] = None

    # ─── Public interface ─────────────────────────────────────────────────────

    def run(self, verbose: bool = False) -> EventLog:
        """Simulate the full match and return the populated event log."""
        self._do_kickoff(kicking_team=0)

        while not self.state.is_over:
            self._tick()
            if verbose and self.state.tick % (self.state.config.tick_rate * 60) == 0:
                print(f"  {self.state.match_minute:2d}' — {self.state.score_str}")

        if verbose:
            print(f"  Full time  — {self.state.score_str}")

        return self.events

    def run_capturing(self, sample_every: int = 3) -> tuple[EventLog, list[dict]]:
        """
        Simulate the full match and capture a frame snapshot every `sample_every` ticks.
        Returns (event_log, frames) where each frame is a compact dict.
        """
        frames: list[dict] = []
        self._do_kickoff(kicking_team=0)
        prev_logged = 0

        while not self.state.is_over:
            self._tick()
            if self.state.tick % sample_every == 0:
                new_events = self.events.events[prev_logged:]
                prev_logged = len(self.events.events)
                frames.append(self._snapshot([e.event_type.value for e in new_events]))

        # Always capture the final frame
        new_events = self.events.events[prev_logged:]
        frames.append(self._snapshot([e.event_type.value for e in new_events]))

        return self.events, frames

    def _snapshot(self, events: list[str]) -> dict:
        total_secs = self.state.simulated_seconds
        goal = self._pending_frame_goals[0] if self._pending_frame_goals else None
        self._pending_frame_goals.clear()
        return {
            "t":      self.state.tick,
            "min":    self.state.match_minute,
            "sec":    int(total_secs % 60),
            "score":  list(self.state.score),
            "phase":  self.state.phase.value,
            "ball":   [round(self.ball.position.x, 2), round(self.ball.position.y, 2)],
            "players": [
                [round(p.position.x, 2), round(p.position.y, 2), 1 if p.has_ball else 0,
                 round(p.stamina / p.max_stamina, 2) if p.max_stamina > 0 else 1.0]
                for team in self.teams
                for p in team.players
            ],
            "events": events,
            "poss":   list(self._possession_ticks),
            "goal":   goal,
        }

    # ─── Main tick ────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._handle_phase()

        if self.state.phase == GamePhase.IN_PLAY:
            self._update_decisions()
            self._update_movement()
            self._update_ball()
            self._resolve_events()
            self.rules.drain_stamina(self.teams, self.state)
            self._update_possession()
            self._check_counterattack_trigger()

        self.state.tick += 1

        if self._counter_press_ticks > 0:
            self._counter_press_ticks -= 1
        if self._counterattack_ticks > 0:
            self._counterattack_ticks -= 1

        for p in self._all_players():
            if self._decide_timer[p] > 0:
                self._decide_timer[p] -= 1
            if self._tackle_timer[p] > 0:
                self._tackle_timer[p] -= 1
            if self._shot_timer[p] > 0:
                self._shot_timer[p] -= 1
            if self._gk_hold_timer[p] > 0:
                self._gk_hold_timer[p] -= 1

    # ─── Phase management ─────────────────────────────────────────────────────

    def _handle_phase(self) -> None:
        phase = self.state.phase

        if phase == GamePhase.PRE_KICKOFF:
            return

        if phase == GamePhase.GOAL_SCORED:
            self.state.phase_ticks_remaining -= 1
            if self.state.phase_ticks_remaining <= 0:
                self._do_kickoff(self.state.restart_team or 0)
            return

        if phase == GamePhase.HALF_TIME:
            self._do_kickoff(self.state.restart_team or 1)
            return

        if phase in (GamePhase.THROW_IN, GamePhase.GOAL_KICK, GamePhase.CORNER_KICK):
            self._do_restart()
            return

        if phase == GamePhase.EXTRA_TIME_BREAK:
            self._do_kickoff(self.state.restart_team or 0)
            return

        if phase == GamePhase.IN_PLAY:
            time_phase = self.rules.check_time(self.state)
            if time_phase == GamePhase.HALF_TIME:
                self._do_half_time()
            elif time_phase == GamePhase.FULL_TIME:
                self._do_full_time()
            elif time_phase == GamePhase.EXTRA_TIME_BREAK:
                self._do_et_break()
            elif time_phase == GamePhase.PENALTY_SHOOTOUT:
                self._do_penalty_shootout()
            return

        if phase == GamePhase.KICKOFF:
            self.state.phase = GamePhase.IN_PLAY
            return

    # ─── Kickoff and restart setup ────────────────────────────────────────────

    def _do_kickoff(self, kicking_team: int) -> None:
        """Place all players in formation; give ball to forward at centre circle."""
        for team in self.teams:
            home = build_home_positions(team, self.pitch, self.state, self.ball)
            for p in team.players:
                p.position = home[p]
                p.velocity = Vec2()
                p.has_ball = False

        kicking = self.teams[kicking_team]
        fwds = kicking.get_by_role(Role.FORWARD)
        kicker = fwds[0] if fwds else min(
            kicking.players,
            key=lambda p: p.position.distance_to(self.pitch.center),
        )
        kicker.position = Vec2(self.pitch.center.x, self.pitch.center.y)
        self._give_ball(kicker)

        self.events.log(
            self.state.tick, EventType.KICKOFF, self.pitch.center, player=kicker,
        )
        self.state.phase = GamePhase.IN_PLAY

    def _do_restart(self) -> None:
        """Restart play for throw-ins, goal kicks, and corners."""
        pos = self.state.restart_position or self.pitch.center
        team_id = self.state.restart_team or 0
        phase = self.state.phase

        event_type = {
            GamePhase.THROW_IN:    EventType.THROW_IN,
            GamePhase.GOAL_KICK:   EventType.GOAL_KICK,
            GamePhase.CORNER_KICK: EventType.CORNER_KICK,
        }.get(phase, EventType.KICKOFF)

        if phase == GamePhase.CORNER_KICK:
            self._do_corner_kick(pos, team_id, event_type)
            return
        if phase == GamePhase.GOAL_KICK:
            self._do_goal_kick(pos, team_id, event_type)
            return

        # Throw-in: award to nearest restart-team player
        team = self.teams[team_id]
        taker = min(team.players, key=lambda p: p.position.distance_to(pos))
        taker.position = Vec2(pos.x, pos.y)
        self._give_ball(taker)
        self.events.log(self.state.tick, event_type, pos, player=taker)
        self.state.phase = GamePhase.IN_PLAY

    def _do_corner_kick(self, pos: Vec2, team_id: int, event_type: EventType) -> None:
        """Cross into the box from a corner flag; attackers make box runs."""
        team = self.teams[team_id]
        taker = min(team.players, key=lambda p: p.position.distance_to(pos))
        taker.position = Vec2(pos.x, pos.y)
        self.ball.position = Vec2(pos.x, pos.y)

        atk_goal_x = self.state.attacking_goal_x(team_id)
        box_depth = 11.0
        cross_x = (atk_goal_x - box_depth) if atk_goal_x >= self.pitch.length else (atk_goal_x + box_depth)
        cross_target = Vec2(
            max(0.0, min(self.pitch.length, cross_x)),
            max(0.0, min(self.pitch.width, self.pitch.width / 2 + random.uniform(-5.0, 5.0))),
        )

        self._last_kicked_by = taker
        self._kick_target = None
        self._kick_is_shot = False
        physics.kick(self.ball, cross_target, 14.0 + random.uniform(-2.0, 2.0))
        taker.has_ball = False
        self.ball.carrier = None

        # Send attackers into the box to contest the cross
        for p in team.players:
            if p is not taker and p.role in (Role.FORWARD, Role.MIDFIELDER):
                run_x = cross_target.x + random.uniform(-3.0, 4.0)
                run_y = cross_target.y + random.uniform(-5.0, 5.0)
                self._move_target[p] = Vec2(
                    max(1.0, min(self.pitch.length - 1.0, run_x)),
                    max(1.0, min(self.pitch.width - 1.0, run_y)),
                )
                self._decide_timer[p] = 0

        self.events.log(self.state.tick, event_type, pos, player=taker)
        self.state.phase = GamePhase.IN_PLAY

    def _do_goal_kick(self, pos: Vec2, team_id: int, event_type: EventType) -> None:
        """GK takes a long distribution toward the most advanced outfield teammate."""
        team = self.teams[team_id]
        gk = team.goalkeeper
        taker = gk if gk is not None else min(team.players, key=lambda p: p.position.distance_to(pos))
        taker.position = Vec2(pos.x, pos.y)
        self.ball.position = Vec2(pos.x, pos.y)

        atk_goal_x = self.state.attacking_goal_x(team_id)
        candidates = [p for p in team.players if p is not taker and p.role != Role.GOALKEEPER]

        if candidates:
            best = max(
                candidates,
                key=lambda p: p.position.x if atk_goal_x >= self.pitch.length else -p.position.x,
            )
            kick_x = best.position.x + (4.0 if atk_goal_x >= self.pitch.length else -4.0) + random.uniform(-3.0, 3.0)
            kick_y = best.position.y + random.uniform(-4.0, 4.0)
        else:
            kick_x = self.pitch.length / 2 + random.uniform(-5.0, 5.0)
            kick_y = self.pitch.width / 2 + random.uniform(-5.0, 5.0)

        kick_target = Vec2(
            max(0.0, min(self.pitch.length, kick_x)),
            max(0.0, min(self.pitch.width, kick_y)),
        )

        self._last_kicked_by = taker
        self._kick_target = None
        self._kick_is_shot = False
        physics.kick(self.ball, kick_target, 18.0 + random.uniform(-2.0, 2.0))
        taker.has_ball = False
        self.ball.carrier = None

        # Reset all outfield players' timers so they immediately reposition to receive
        for p in team.players:
            if p is not taker:
                self._decide_timer[p] = 0

        self.events.log(self.state.tick, event_type, pos, player=taker)
        self.state.phase = GamePhase.IN_PLAY

    def _do_half_time(self) -> None:
        for p in self._all_players():
            p.stamina = p.max_stamina
        self.rules.apply_half_time(self.state)
        self.events.log(self.state.tick, EventType.HALF_TIME, self.pitch.center)

    def _do_full_time(self) -> None:
        self.rules.apply_full_time(self.state)
        self.events.log(self.state.tick, EventType.FULL_TIME, self.pitch.center)

    def _do_et_break(self) -> None:
        """Transition into the next ET period: increment counter, log, set up kickoff."""
        self.state.et_period += 1
        self.state.phase = GamePhase.EXTRA_TIME_BREAK
        # Alternate kickoff teams: team 1 starts ET1, team 0 starts ET2
        self.state.restart_team = 1 if self.state.et_period == 1 else 0
        self.events.log(self.state.tick, EventType.EXTRA_TIME, self.pitch.center)

    def _do_penalty_shootout(self) -> None:
        """Simulate a full penalty shootout and resolve the match."""
        self.state.phase = GamePhase.PENALTY_SHOOTOUT
        scores = [0, 0]
        kicks: list[dict] = []
        MAX_ROUNDS = 5

        kicker_pools = [
            [p for p in t.players if p.role != Role.GOALKEEPER]
            for t in self.teams
        ]
        gks = [t.goalkeeper for t in self.teams]

        # Ensure kicker pools are never empty (fall back to all players)
        for i, pool in enumerate(kicker_pools):
            if not pool:
                kicker_pools[i] = list(self.teams[i].players)

        kicks_taken = [0, 0]
        done = False
        for rnd in range(MAX_ROUNDS):
            if done:
                break
            for shooting_team in (0, 1):
                defending_team = 1 - shooting_team
                kicker = kicker_pools[shooting_team][rnd % len(kicker_pools[shooting_team])]
                gk = gks[defending_team]
                scored = self._penalty_roll(kicker, gk)
                if scored:
                    scores[shooting_team] += 1
                kicks_taken[shooting_team] += 1
                kicks.append({
                    "round":   rnd + 1,
                    "team":    shooting_team,
                    "shooter": kicker.name,
                    "gk":      gk.name if gk else None,
                    "scored":  scored,
                    "score":   list(scores),
                })
                self.events.log(self.state.tick, EventType.PENALTY_KICK, self.pitch.center, player=kicker)
                # Early termination: can the trailing team still catch up?
                for t in (0, 1):
                    ot = 1 - t
                    if scores[t] > scores[ot] + (MAX_ROUNDS - kicks_taken[ot]):
                        done = True
                        break

        # Sudden death until one team leads after both have kicked
        sd = 0
        while scores[0] == scores[1] and sd < 20:
            for shooting_team in (0, 1):
                defending_team = 1 - shooting_team
                kicker = kicker_pools[shooting_team][(MAX_ROUNDS + sd) % len(kicker_pools[shooting_team])]
                gk = gks[defending_team]
                scored = self._penalty_roll(kicker, gk)
                if scored:
                    scores[shooting_team] += 1
                kicks.append({
                    "round":   f"SD{sd + 1}",
                    "team":    shooting_team,
                    "shooter": kicker.name,
                    "gk":      gk.name if gk else None,
                    "scored":  scored,
                    "score":   list(scores),
                })
                self.events.log(self.state.tick, EventType.PENALTY_KICK, self.pitch.center, player=kicker)
            sd += 1

        self._penalty_kicks = kicks
        self._penalty_winner = 0 if scores[0] >= scores[1] else 1
        self._do_full_time()

    @staticmethod
    def _penalty_roll(shooter: Player, gk: Optional[Player]) -> bool:
        """Returns True if the penalty kick results in a goal."""
        shot_q = (shooter.stats.attack + shooter.stats.sp_attack) / 2.0
        if gk is None:
            return random.random() < 0.74
        save_q = (gk.stats.defense + gk.stats.sp_defense) / 2.0
        goal_prob = shot_q / (shot_q + save_q * 0.35)
        goal_prob = max(0.40, min(0.92, goal_prob))
        return random.random() < goal_prob

    # ─── Decision update ──────────────────────────────────────────────────────

    def _update_decisions(self) -> None:
        carrier = self.ball.carrier

        # GK just picked up the ball → impose a hold before distributing,
        # but skip it for back-passes from teammates (back-pass rule).
        if (carrier is not None
                and carrier is not self._prev_carrier
                and carrier.role == Role.GOALKEEPER
                and self._gk_hold_timer[carrier] == 0):
            back_pass = (
                self._last_kicked_by is not None
                and self._last_kicked_by.team_id == carrier.team_id
            )
            if not back_pass:
                self._gk_hold_timer[carrier] = 25   # 2.5 sim-seconds
                self._decide_timer[carrier]  = 25

        if carrier is not None and self._decide_timer[carrier] <= 0:
            # GK still in hold window — don't distribute yet
            if carrier.role == Role.GOALKEEPER and self._gk_hold_timer[carrier] > 0:
                self._decide_timer[carrier] = CARRIER_DECIDE_TICKS
            else:
                self._carrier_decide(carrier)
                if self.ball.carrier is carrier:
                    self._decide_timer[carrier] = CARRIER_DECIDE_TICKS

        self._prev_carrier = carrier

        for p in self._all_players():
            if p is not carrier and self._decide_timer[p] <= 0:
                is_counter = (
                    self._counterattack_team == p.team_id
                    and self._counterattack_ticks > 0
                )

                # Forward run: sprint toward goal when attacking OR on counterattack
                if (
                    carrier is not None
                    and carrier.team_id == p.team_id
                    and p.role == Role.FORWARD
                ):
                    atk_goal_x = self.state.attacking_goal_x(p.team_id)
                    mid_x = self.pitch.length / 2
                    ball_in_attack = (
                        (atk_goal_x >= self.pitch.length and self.ball.position.x > mid_x)
                        or (atk_goal_x <= 0.0 and self.ball.position.x < mid_x)
                    )
                    if ball_in_attack or is_counter:
                        run_dist = 12.0 if is_counter else 8.0
                        goal = Vec2(atk_goal_x, self.pitch.width / 2)
                        toward_goal = (goal - self.ball.position).normalized()
                        raw = self.ball.position + toward_goal * run_dist
                        self._move_target[p] = Vec2(
                            max(1.0, min(self.pitch.length - 1.0, raw.x)),
                            max(1.0, min(self.pitch.width - 1.0, raw.y)),
                        )
                        self._decide_timer[p] = OFFBALL_DECIDE_TICKS
                        continue

                # Attacking MID run: both MIDs overlap into the box, offset to each side
                if (
                    carrier is not None
                    and carrier.team_id == p.team_id
                    and p.role == Role.MIDFIELDER
                ):
                    atk_goal_x = self.state.attacking_goal_x(p.team_id)
                    mid_x = self.pitch.length / 2
                    ball_in_attack = (
                        (atk_goal_x >= self.pitch.length and self.ball.position.x > mid_x)
                        or (atk_goal_x <= 0.0 and self.ball.position.x < mid_x)
                    )
                    if ball_in_attack or is_counter:
                        goal = Vec2(atk_goal_x, self.pitch.width / 2)
                        toward_goal = (goal - self.ball.position).normalized()
                        # Offset opposite sides so the two MIDs don't overlap
                        side = 1.0 if p.position.y < self.pitch.width / 2 else -1.0
                        perp = Vec2(-toward_goal.y * side, toward_goal.x * side)
                        raw = self.ball.position + toward_goal * 9.0 + perp * 3.0
                        self._move_target[p] = Vec2(
                            max(1.0, min(self.pitch.length - 1.0, raw.x)),
                            max(1.0, min(self.pitch.width - 1.0, raw.y)),
                        )
                        self._decide_timer[p] = OFFBALL_DECIDE_TICKS
                        continue

                # Man-marking: DEFs track the nearest dangerous attacker in their zone
                if (
                    p.role == Role.DEFENDER
                    and (carrier is None or carrier.team_id != p.team_id)
                ):
                    def_goal_x = self.state.attacking_goal_x(1 - p.team_id)
                    def_goal = Vec2(def_goal_x, self.pitch.width / 2)
                    opp_team = self.teams[1 - p.team_id]
                    threats = [
                        q for q in opp_team.players
                        if q.role != Role.GOALKEEPER
                        and q.position.distance_to(def_goal) < 18.0
                    ]
                    if threats:
                        threat = min(threats, key=lambda q: q.position.distance_to(p.position))
                        to_goal = (def_goal - threat.position).normalized()
                        mark_pos = Vec2(
                            max(1.0, min(self.pitch.length - 1.0,
                                         threat.position.x + to_goal.x * 2.0)),
                            max(1.0, min(self.pitch.width - 1.0,
                                         threat.position.y + to_goal.y * 2.0)),
                        )
                        self._move_target[p] = mark_pos
                    else:
                        self._move_target[p] = self._defensive_compact_position(p)
                    self._decide_timer[p] = OFFBALL_DECIDE_TICKS
                    continue

                # Tracking-back: FWD drops just over center when ball is in own defensive third
                if p.role == Role.FORWARD and (carrier is None or carrier.team_id != p.team_id):
                    atk_goal_x = self.state.attacking_goal_x(p.team_id)
                    L = self.pitch.length
                    ball_in_def_third = (
                        (atk_goal_x >= L and self.ball.position.x < L / 3)
                        or (atk_goal_x <= 0.0 and self.ball.position.x > 2 * L / 3)
                    )
                    if ball_in_def_third:
                        drop_x = L * 0.52 if atk_goal_x >= L else L * 0.48
                        self._move_target[p] = Vec2(
                            max(1.0, min(L - 1.0, drop_x)),
                            max(1.0, min(self.pitch.width - 1.0, p.position.y)),
                        )
                        self._decide_timer[p] = OFFBALL_DECIDE_TICKS
                        continue

                # Tracking-back: MID drops toward defensive third boundary when ball is deep
                if p.role == Role.MIDFIELDER and (carrier is None or carrier.team_id != p.team_id):
                    atk_goal_x = self.state.attacking_goal_x(p.team_id)
                    L = self.pitch.length
                    ball_in_def_third = (
                        (atk_goal_x >= L and self.ball.position.x < L / 3)
                        or (atk_goal_x <= 0.0 and self.ball.position.x > 2 * L / 3)
                    )
                    if ball_in_def_third:
                        drop_x = L * 0.38 if atk_goal_x >= L else L * 0.62
                        self._move_target[p] = Vec2(
                            max(1.0, min(L - 1.0, drop_x)),
                            max(1.0, min(self.pitch.width - 1.0, p.position.y)),
                        )
                        self._decide_timer[p] = OFFBALL_DECIDE_TICKS
                        continue

                team = self.teams[p.team_id]
                home = build_home_positions(team, self.pitch, self.state, self.ball)
                self._move_target[p] = home[p]
                self._decide_timer[p] = OFFBALL_DECIDE_TICKS

    def _carrier_decide(self, carrier: Player) -> None:
        team = self.teams[carrier.team_id]
        opp_team = self.teams[1 - carrier.team_id]
        teammates = [p for p in team.players if p is not carrier]
        opponents = list(opp_team.players)

        if carrier.role == Role.GOALKEEPER:
            forbidden: frozenset[str] = frozenset({"shoot", "dribble"})
        elif self._shot_timer[carrier] > 0:
            forbidden = frozenset({"shoot"})
        else:
            forbidden = frozenset()

        # Counterattack: suppress shooting from own half — drive forward instead
        if self._counterattack_team == carrier.team_id and self._counterattack_ticks > 0:
            atk_goal_x = self.state.attacking_goal_x(carrier.team_id)
            mid_x = self.pitch.length / 2
            in_own_half = (
                (atk_goal_x >= self.pitch.length and carrier.position.x < mid_x)
                or (atk_goal_x <= 0.0 and carrier.position.x > mid_x)
            )
            if in_own_half:
                forbidden = forbidden | frozenset({"shoot"})

        action = self.decider.decide(
            carrier, self.ball, teammates, opponents, self.state, forbidden=forbidden
        )

        if action.action_type == "shoot":
            self._execute_shot(carrier)
        elif action.action_type == "pass" and action.target is not None:
            self._execute_pass(carrier, action.target)
        else:
            # Dribble: move toward goal while evading the nearest pressing defender.
            goal = Vec2(
                self.state.attacking_goal_x(carrier.team_id),
                self.pitch.width / 2,
            )
            goal_dir = (goal - carrier.position).normalized()

            nearest_opp = min(
                opponents,
                key=lambda o: o.position.distance_to(carrier.position),
                default=None,
            )
            if nearest_opp is not None and nearest_opp.position.distance_to(carrier.position) < 8.0:
                away = (carrier.position - nearest_opp.position).normalized()
                # 70 % toward goal, 30 % away from nearest defender
                raw = Vec2(goal_dir.x * 0.7 + away.x * 0.3, goal_dir.y * 0.7 + away.y * 0.3)
                dribble_dir = raw.normalized()
            else:
                dribble_dir = goal_dir

            raw_target = carrier.position + dribble_dir * 6.0
            self._move_target[carrier] = Vec2(
                max(1.0, min(self.pitch.length - 1.0, raw_target.x)),
                max(1.0, min(self.pitch.width - 1.0, raw_target.y)),
            )

    # ─── Action execution ─────────────────────────────────────────────────────

    def _execute_pass(self, carrier: Player, target: Player) -> None:
        """Kick the ball physically toward target; outcome resolved when ball is collected."""
        if self._is_offside(target):
            self.events.log(self.state.tick, EventType.OFFSIDE, target.position,
                            player=carrier, target=target)
            self.state.restart_position = Vec2(target.position.x, target.position.y)
            self.state.restart_team = 1 - carrier.team_id
            self.state.phase = GamePhase.THROW_IN
            return
        pass_spd = physics.pass_speed(carrier)
        dist = carrier.position.distance_to(target.position)
        travel_time = dist / pass_spd if pass_spd > 0.0 else 0.0
        # Lead target: aim ahead of where they'll be (0.3 blending keeps it realistic)
        lead = Vec2(
            target.position.x + target.velocity.x * travel_time * 0.3,
            target.position.y + target.velocity.y * travel_time * 0.3,
        )
        # Accuracy scatter: sp_attack governs precision; longer passes magnify error
        accuracy = min(1.0, carrier.effective_sp_attack / 100.0)
        spread = dist * 0.05 * (1.8 - accuracy)
        aim = Vec2(
            max(0.0, min(self.pitch.length, lead.x + random.gauss(0, spread * 0.5))),
            max(0.0, min(self.pitch.width,  lead.y + random.gauss(0, spread))),
        )
        self._last_kicked_by = carrier
        self._kick_target = target
        self._kick_is_shot = False
        physics.kick(self.ball, aim, pass_spd)
        carrier.has_ball = False
        self.ball.carrier = None

    def _execute_shot(self, carrier: Player) -> None:
        """Kick the ball physically toward goal; outcome resolved when ball stops or is collected."""
        goal_x = self.state.attacking_goal_x(carrier.team_id)
        goal_center_y = self.pitch.width / 2
        dist = carrier.position.distance_to(Vec2(goal_x, goal_center_y))

        # Shot placement: Attack (power) + Sp.Atk (technique) → accuracy 0-1.
        # sigma_y is the std-dev of lateral scatter; when |y_offset| > half_gw
        # the ball misses wide or over the bar → goal kick for the defence.
        # Target miss rate: ~5% close in for elite shooters, ~20% from box edge
        # for average stats, ~40%+ for weak shooters from long range.
        half_gw     = self.pitch.goal_width / 2         # 2.82 m
        accuracy    = min(1.0, (carrier.effective_attack + carrier.effective_sp_attack) / 240.0)
        dist_factor = min(1.0, dist / 25.0)
        sigma_y     = half_gw * (0.5 + 1.0 * dist_factor) * (1.2 - 0.7 * accuracy)
        y_offset    = random.gauss(0, max(0.3, sigma_y))
        shot_target = Vec2(goal_x, goal_center_y + y_offset)

        self.events.log(self.state.tick, EventType.SHOT, carrier.position, player=carrier)
        self._last_kicked_by = carrier
        self._kick_target = None
        self._kick_is_shot = True
        self._shot_beat_gk = False
        self._shot_timer[carrier] = SHOT_COOLDOWN_TICKS
        physics.kick(self.ball, shot_target, physics.shot_speed(carrier))
        carrier.has_ball = False
        self.ball.carrier = None

    # ─── Movement ─────────────────────────────────────────────────────────────

    def _update_movement(self) -> None:
        carrier = self.ball.carrier
        ball_spd = math.hypot(self.ball.velocity.x, self.ball.velocity.y)

        if carrier is None:
            for team in self.teams:
                home = build_home_positions(team, self.pitch, self.state, self.ball)
                # GK intercept heading: defending goal x for this team
                def_goal_x = self.state.attacking_goal_x(1 - team.team_id)
                ball_toward_goal = (
                    ball_spd > physics.MIN_SPEED and (
                        (def_goal_x >= self.pitch.length and self.ball.velocity.x > 0.5) or
                        (def_goal_x <= 0.0 and self.ball.velocity.x < -0.5)
                    )
                )
                crossing = physics.goal_crossing(self.ball, def_goal_x) if ball_toward_goal else None

                chaser = min(
                    team.players,
                    key=lambda p: p.position.distance_to(self.ball.position),
                )
                for p in team.players:
                    if p.role == Role.GOALKEEPER and crossing is not None:
                        y_cross, _ = crossing
                        gk_y = max(1.0, min(self.pitch.width - 1.0, y_cross))
                        gk_target = Vec2(def_goal_x, gk_y)
                        desired = seek(p.position, gk_target, p.max_speed_mps)
                    elif p.role == Role.GOALKEEPER:
                        desired = arrive(p.position, self._gk_angle_position(p), p.max_speed_mps)
                    elif p is chaser:
                        desired = seek(p.position, self.ball.position, p.max_speed_mps)
                    else:
                        desired = arrive(p.position, home[p], p.max_speed_mps)
                    apply_steering(p, desired, team.players, DT, pitch=self.pitch)
            return

        for team in self.teams:
            pressers: set[Player] = set()
            if carrier.team_id != team.team_id and carrier.role != Role.GOALKEEPER:
                sorted_by_dist = sorted(
                    team.players,
                    key=lambda p: p.position.distance_to(carrier.position),
                )
                pressers = set(sorted_by_dist[:self._n_pressers(team.team_id)])

            home = build_home_positions(team, self.pitch, self.state, self.ball)

            for p in team.players:
                if p is carrier:
                    target = self._move_target.get(p, home[p])
                    desired = arrive(p.position, target, p.max_speed_mps)
                elif p.role == Role.GOALKEEPER and p.team_id != carrier.team_id:
                    desired = arrive(p.position, self._gk_angle_position(p), p.max_speed_mps)
                elif p in pressers:
                    desired = seek(p.position, carrier.position, p.max_speed_mps)
                else:
                    target = self._move_target.get(p, home[p])
                    desired = arrive(p.position, target, p.max_speed_mps)

                apply_steering(p, desired, team.players, DT, pitch=self.pitch)

    # ─── Ball sync and in-play event resolution ───────────────────────────────

    def _update_ball(self) -> None:
        """Sync ball to carrier; advance loose ball with friction physics."""
        if self.ball.carrier is not None:
            self.ball.position = self.ball.carrier.position
        else:
            self._ball_prev_pos = Vec2(self.ball.position.x, self.ball.position.y)
            physics.step(self.ball, DT)

    def _resolve_events(self) -> None:
        """Check goal, out-of-bounds, tackles, loose-ball pickup in priority order."""
        if self.state.phase != GamePhase.IN_PLAY:
            return

        self.rules.sync_possession(self.ball, self.state)
        carrier = self.ball.carrier

        if carrier is not None:
            scored_by = self.rules.check_goal(self.ball, self.state)
            if scored_by is not None:
                assist_p = self._last_passer.get(carrier)
                if assist_p is not None and assist_p.team_id != scored_by:
                    assist_p = None
                goal_record = {
                    "team":   scored_by,
                    "scorer": carrier.name,
                    "assist": assist_p.name if assist_p else None,
                    "minute": self.state.match_minute,
                    "tick":   self.state.tick,
                    "score":  [
                        self.state.score[0] + (1 if scored_by == 0 else 0),
                        self.state.score[1] + (1 if scored_by == 1 else 0),
                    ],
                }
                self._goals.append(goal_record)
                self._pending_frame_goals.append(goal_record)
                self.events.log(
                    self.state.tick, EventType.GOAL, self.ball.position, player=carrier,
                )
                self.rules.apply_goal(scored_by, self.state)
                self._lose_ball(carrier, self.ball.position)
                return

            self._check_tackles(carrier)
        else:
            # Loose ball: give players (especially GK) a chance to collect first
            if self._check_loose_ball():
                return

            # Ball out of play
            if not self.pitch.is_in_bounds(self.ball.position):
                scored_by = self.rules.check_goal(self.ball, self.state)
                if scored_by is not None:
                    scorer = self._last_kicked_by
                    assist_p = self._last_passer.get(scorer) if scorer else None
                    if assist_p is not None and assist_p.team_id != scored_by:
                        assist_p = None
                    goal_record = {
                        "team":   scored_by,
                        "scorer": scorer.name if scorer else "Unknown",
                        "assist": assist_p.name if assist_p else None,
                        "minute": self.state.match_minute,
                        "tick":   self.state.tick,
                        "score":  [
                            self.state.score[0] + (1 if scored_by == 0 else 0),
                            self.state.score[1] + (1 if scored_by == 1 else 0),
                        ],
                    }
                    self._goals.append(goal_record)
                    self._pending_frame_goals.append(goal_record)
                    self.events.log(
                        self.state.tick, EventType.GOAL, self.ball.position, player=scorer,
                    )
                    self.rules.apply_goal(scored_by, self.state)
                    self.ball.velocity = Vec2()
                    return

                oob_phase = self.rules.check_out_of_bounds(self.ball, self.state)
                if oob_phase is not None:
                    self.rules.apply_restart(oob_phase, self.ball.position, self.state)
                self.ball.velocity = Vec2()

    def _check_tackles(self, carrier: Player) -> None:
        # GK holding the ball cannot be challenged — opposing players back off
        if carrier.role == Role.GOALKEEPER:
            return
        opp_team = self.teams[1 - carrier.team_id]
        for opp in opp_team.players:
            if self._tackle_timer[opp] > 0:
                continue
            if opp.position.distance_to(carrier.position) > TACKLE_RANGE_M:
                continue

            atk_speed = carrier.effective_speed
            def_stat = opp.effective_defense * 0.6 + opp.effective_speed * 0.4
            prob = def_stat / (def_stat + atk_speed)
            self._tackle_timer[opp] = TACKLE_INTERVAL_TICKS

            mid = Vec2(
                (opp.position.x + carrier.position.x) / 2,
                (opp.position.y + carrier.position.y) / 2,
            )

            if random.random() < prob:
                squirt_vel = carrier.velocity * 0.4
                self._lose_ball(carrier, mid)
                self.ball.velocity = squirt_vel
                # Reset kick tracking: loose ball from tackle has no kick intent
                self._last_kicked_by = None
                self._kick_target = None
                self._kick_is_shot = False
                self.events.log(
                    self.state.tick, EventType.TACKLE_WON, mid,
                    player=opp, target=carrier,
                )
                self.events.log(
                    self.state.tick, EventType.TACKLE_LOST, mid,
                    player=carrier, target=opp,
                )
                return
            else:
                self.events.log(
                    self.state.tick, EventType.TACKLE_LOST, carrier.position,
                    player=opp, target=carrier,
                )

    def _ball_near_defending_goal(self, team_id: int) -> bool:
        """True if the ball is within GK_PA_DEPTH_M of team_id's defending goal line."""
        bx = self.ball.position.x
        def_goal_x = self.state.attacking_goal_x(1 - team_id)
        if def_goal_x >= self.pitch.length:
            return bx >= self.pitch.length - GK_PA_DEPTH_M
        return bx <= GK_PA_DEPTH_M

    def _check_loose_ball(self) -> bool:
        """
        Three-phase loose-ball resolution. Returns True if a pickup occurred.

        Phase 1 — GK priority (tackle-drops and rebounds near own goal).
        Phase 2 — Directed pass (arrival / interception logic).
        Phase 3 — Nearest player scrambles for a truly loose ball.
        """
        ball_spd = math.hypot(self.ball.velocity.x, self.ball.velocity.y)
        if self._phase1_gk_claim():
            return True
        if not self._kick_is_shot and self._kick_target is not None:
            in_flight, pickup = self._phase2_directed_pass(ball_spd)
            if in_flight:
                return False   # ball still flying; no pickup this tick
            if pickup:
                return True    # clean uncontested arrival
            # contested arrival: fall through to scramble
        return self._phase3_loose_scramble(ball_spd)

    def _phase1_gk_claim(self) -> bool:
        """GK claims any truly-loose ball (tackle drop / rebound) near their goal."""
        if self._kick_is_shot or self._kick_target is not None:
            return False
        for team in self.teams:
            gk = team.goalkeeper
            if gk is None:
                continue
            if not self._ball_near_defending_goal(team.team_id):
                continue
            if self._swept_dist(gk.position, self._ball_prev_pos, self.ball.position) <= GK_CLAIM_RANGE_M:
                self._resolve_pickup(gk)
                return True
        return False

    def _phase2_directed_pass(self, ball_spd: float) -> tuple[bool, bool]:
        """
        Resolves an intentional pass.

        Returns (in_flight, pickup_occurred):
          (True,  False) — ball still flying, no interception; skip phase 3
          (False, True)  — clean arrival; pickup done
          (False, False) — contested arrival; fall to phase 3
        """
        target = self._kick_target
        kt_dist = target.position.distance_to(self.ball.position)
        passer_team = self._last_kicked_by.team_id if self._last_kicked_by else -1

        ball_near_target = kt_dist <= PASS_TARGET_RANGE_M
        ball_slowed = ball_spd < 3.0

        if ball_near_target or ball_slowed:
            contested = any(
                self._swept_dist(p.position, self._ball_prev_pos, self.ball.position) <= PICKUP_RANGE_M
                for p in self._all_players()
                if p.team_id != passer_team
            )
            if not contested:
                self._resolve_pickup(target)
                return False, True
            return False, False   # contested → phase 3
        else:
            # Ball in flight: mid-flight interception only
            for p in self._all_players():
                if p.team_id == passer_team:
                    continue
                if self._swept_dist(p.position, self._ball_prev_pos, self.ball.position) <= PICKUP_RANGE_M:
                    self._resolve_pickup(p)
                    return False, True
            return True, False    # still in flight

    def _phase3_loose_scramble(self, ball_spd: float) -> bool:
        """Nearest player within PICKUP_RANGE_M wins the loose ball.
        GK gets priority within GK_SHOT_SAVE_RANGE_M when a shot is in flight."""
        best: Optional[Player] = None
        best_dist = PICKUP_RANGE_M

        # On live shots: defending GK gets extended dive range and wins any competition.
        # This models the GK diving to cover near-post/far-post shots that would otherwise
        # slip past the 1.5 m pickup radius when the GK is a step away from the crossing.
        if (
            self._kick_is_shot
            and not self._shot_beat_gk
            and self._last_kicked_by is not None
        ):
            for team in self.teams:
                if team.team_id == self._last_kicked_by.team_id:
                    continue
                gk = team.goalkeeper
                if gk is None:
                    continue
                dist = gk.position.distance_to(self.ball.position)
                if dist <= GK_SHOT_SAVE_RANGE_M:
                    best = gk
                    best_dist = dist
                break  # only one defending team

        if best is None:
            for p in self._all_players():
                if (
                    self._shot_beat_gk
                    and p.role == Role.GOALKEEPER
                    and self._last_kicked_by is not None
                    and p.team_id != self._last_kicked_by.team_id
                ):
                    continue
                dist = p.position.distance_to(self.ball.position)
                if dist < best_dist:
                    best_dist = dist
                    best = p

        if best is None:
            return False

        # Stat-based save check: GK must earn saves against live shots
        if (
            self._kick_is_shot
            and not self._shot_beat_gk
            and best.role == Role.GOALKEEPER
            and self._last_kicked_by is not None
            and best.team_id != self._last_kicked_by.team_id
            and ball_spd >= 5.0
        ):
            shooter = self._last_kicked_by
            shot_q = (shooter.effective_attack + shooter.effective_sp_attack) / 2.0
            save_q = (best.effective_defense + best.effective_sp_defense) / 2.0
            save_prob = min(0.90, (save_q * 1.5) / (shot_q + save_q * 1.5))
            if random.random() > save_prob:
                self._shot_beat_gk = True
                return False  # GK dives but ball beats them

            # 15% chance ball spills loose (rebound)
            if random.random() < 0.15:
                self._shot_beat_gk = True
                self._kick_target = None
                spill_spd = max(ball_spd * 0.4, 3.0)
                scatter = random.uniform(-math.pi / 3, math.pi / 3)
                ref_x = -self.ball.velocity.x / ball_spd if ball_spd > 1e-6 else -1.0
                ref_y =  self.ball.velocity.y / ball_spd if ball_spd > 1e-6 else 0.0
                cos_s, sin_s = math.cos(scatter), math.sin(scatter)
                sx = ref_x * cos_s - ref_y * sin_s
                sy = ref_x * sin_s + ref_y * cos_s
                mag = math.hypot(sx, sy)
                if mag > 1e-6:
                    self.ball.velocity = Vec2(sx / mag * spill_spd, sy / mag * spill_spd)
                self.events.log(self.state.tick, EventType.SAVE, self.ball.position, player=best)
                return False  # ball stays loose; attackers scramble

        self._resolve_pickup(best)
        return True

    def _resolve_pickup(self, best: Player) -> None:
        """Log the appropriate event and give the ball to `best`."""
        lkb = self._last_kicked_by
        kt = self._kick_target

        if self._kick_is_shot:
            if (
                best.role == Role.GOALKEEPER
                and lkb is not None
                and best.team_id != lkb.team_id
            ):
                self.events.log(
                    self.state.tick, EventType.SAVE, self.ball.position, player=best,
                )
            else:
                self.events.log(
                    self.state.tick, EventType.POSSESSION_CHANGE, self.ball.position,
                    player=best,
                )
        elif kt is not None:
            if best is kt:
                self.events.log(
                    self.state.tick, EventType.PASS_COMPLETE, self.ball.position,
                    player=lkb, target=best,
                )
            elif lkb is not None and best.team_id != lkb.team_id:
                self.events.log(
                    self.state.tick, EventType.PASS_INTERCEPTED, self.ball.position,
                    player=lkb, target=best,
                )
            else:
                self.events.log(
                    self.state.tick, EventType.POSSESSION_CHANGE, self.ball.position,
                    player=best,
                )
        else:
            self.events.log(
                self.state.tick, EventType.POSSESSION_CHANGE, self.ball.position,
                player=best,
            )

        # Track assist: set last_passer for the receiver if this was a successful pass
        if kt is not None and best is kt and lkb is not None and lkb.team_id == best.team_id:
            self._last_passer[best] = lkb
        else:
            self._last_passer[best] = None

        self._give_ball(best)

    # ─── Ball transfer helpers ────────────────────────────────────────────────

    def _give_ball(self, player: Player) -> None:
        if self.ball.carrier is not None:
            old_team_id = self.ball.carrier.team_id
            self.ball.carrier.has_ball = False
            if old_team_id != player.team_id:
                self._counter_press_ticks = 15
                self._counter_press_team = old_team_id

        self.ball.carrier = player
        self.ball.position = player.position
        self.ball.velocity = Vec2()
        player.has_ball = True
        self._decide_timer[player] = 0
        # Clear kick tracking now that someone has the ball
        self._last_kicked_by = None
        self._kick_target = None
        self._kick_is_shot = False
        self._shot_beat_gk = False

    def _lose_ball(self, carrier: Player, position: Vec2) -> None:
        carrier.has_ball = False
        self.ball.carrier = None
        self.ball.position = position

    def _n_pressers(self, defending_team_id: int) -> int:
        """Defending team presses harder when losing; extra presser during counter-press window."""
        diff = self.state.score[defending_team_id] - self.state.score[1 - defending_team_id]
        if diff >= 2:
            n = 1
        elif diff <= -1:
            n = 3
        else:
            n = 2
        if self._counter_press_ticks > 0 and self._counter_press_team == defending_team_id:
            n = min(3, n + 1)
        return n

    def _gk_angle_position(self, gk: Player) -> Vec2:
        """
        Position the GK on the ball-to-goal line to narrow the shooting angle.
        Stands up to 3m off the goal line; closer ball → smaller standoff.
        """
        def_goal_x = self.state.attacking_goal_x(1 - gk.team_id)
        goal_center = Vec2(def_goal_x, self.pitch.width / 2)
        ball_dist = self.ball.position.distance_to(goal_center)
        if ball_dist < 1e-6:
            return goal_center
        to_ball = (self.ball.position - goal_center).normalized()
        standoff = min(3.0, ball_dist * 0.15)
        target = goal_center + to_ball * standoff
        return Vec2(
            max(0.3, min(self.pitch.length - 0.3, target.x)),
            max(1.0, min(self.pitch.width - 1.0, target.y)),
        )

    def _check_counterattack_trigger(self) -> None:
        """Open a 3-second counterattack window when the last-touch team changes."""
        # last_touch_team persists through loose-ball phases; possession_team goes None.
        cur = self.state.last_touch_team
        prev = self._prev_possession_team
        if cur is not None and prev is not None and cur != prev:
            self._counterattack_team = cur
            self._counterattack_ticks = 30
            for q in self._all_players():
                if q.team_id == cur and not q.has_ball:
                    self._decide_timer[q] = 0
        self._prev_possession_team = cur

    def _update_possession(self) -> None:
        """Credit one tick of possession to the team currently controlling the ball."""
        if self.ball.carrier is not None:
            self._possession_ticks[self.ball.carrier.team_id] += 1
        elif (
            not self._kick_is_shot
            and self._kick_target is not None
            and self._last_kicked_by is not None
        ):
            # Pass in transit: credit to the passing team
            self._possession_ticks[self._last_kicked_by.team_id] += 1

    # ─── Utility ──────────────────────────────────────────────────────────────

    def _all_players(self) -> list[Player]:
        return [p for team in self.teams for p in team.players]

    def _is_offside(self, target: Player) -> bool:
        """True if target is in an offside position at the moment of the pass."""
        atk_goal_x = self.state.attacking_goal_x(target.team_id)
        half_x = self.pitch.length / 2
        opponents = self.teams[1 - target.team_id].players

        if atk_goal_x >= self.pitch.length:  # attacking right
            if target.position.x <= half_x:
                return False
            opp_x = sorted((p.position.x for p in opponents), reverse=True)
            if len(opp_x) < 2:
                return False
            return target.position.x > opp_x[1]
        else:  # attacking left
            if target.position.x >= half_x:
                return False
            opp_x = sorted(p.position.x for p in opponents)
            if len(opp_x) < 2:
                return False
            return target.position.x < opp_x[1]

    def _defensive_compact_position(self, p: Player) -> Vec2:
        """Compact block position for a defending player when there is no direct man-mark threat."""
        def_goal_x = self.state.attacking_goal_x(1 - p.team_id)
        ball_x = self.ball.position.x
        ball_y = self.ball.position.y
        mid_y = self.pitch.width / 2

        if def_goal_x >= self.pitch.length:  # defending right goal
            line_x = min(def_goal_x - 7.0, max(def_goal_x - 18.0, ball_x + 3.0))
        else:  # defending left goal
            line_x = max(def_goal_x + 7.0, min(def_goal_x + 18.0, ball_x - 3.0))

        # Lateral position: preserve current y — only depth (x) is compacted
        return Vec2(
            max(1.0, min(self.pitch.length - 1.0, line_x)),
            max(1.0, min(self.pitch.width - 1.0, p.position.y)),
        )

    @staticmethod
    def _swept_dist(player_pos: Vec2, ball_from: Vec2, ball_to: Vec2) -> float:
        """
        Minimum distance from player_pos to the ball's trajectory segment ball_from→ball_to.
        Used for swept-sphere pickup detection at low tick rates where the ball
        can travel many metres per tick and skip over a player's point radius.
        """
        dx = ball_to.x - ball_from.x
        dy = ball_to.y - ball_from.y
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-10:
            return player_pos.distance_to(ball_from)
        # Project player onto segment, clamp to [0, 1]
        t = ((player_pos.x - ball_from.x) * dx + (player_pos.y - ball_from.y) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        closest_x = ball_from.x + t * dx
        closest_y = ball_from.y + t * dy
        return math.hypot(player_pos.x - closest_x, player_pos.y - closest_y)
