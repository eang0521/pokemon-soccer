from __future__ import annotations

from typing import Optional

from sim.core.entities import Ball, Pitch, Vec2
from sim.core.match_state import GamePhase, MatchState


class Rules:
    """
    Stateless rule evaluator. Answers "what just happened?" and "what should
    happen next?" given current ball position and match state.
    Does not mutate state directly — callers apply results via the apply_* methods.
    """

    def __init__(self, pitch: Pitch):
        self.pitch = pitch

    # ── Possession ────────────────────────────────────────────────────────────

    def sync_possession(self, ball: Ball, state: MatchState) -> None:
        """
        Copy carrier info into match state and update last_touch.
        Call once per tick after movement resolves.
        """
        if ball.carrier is not None:
            state.possession_team = ball.carrier.team_id
            state.possession_player = ball.carrier.name
            state.last_touch_team = ball.carrier.team_id
            state.last_touch_player = ball.carrier.name
        else:
            state.possession_team = None
            state.possession_player = None
            # last_touch_* intentionally preserved — it's still the most recent touch

    # ── Goal detection ────────────────────────────────────────────────────────

    def check_goal(self, ball: Ball, state: MatchState) -> Optional[int]:
        """
        Returns the team_id that scored, or None.
        Correctly handles half-time direction flips via state.attacking_goal_x().
        """
        pos = ball.position
        p = self.pitch
        half_gw = p.goal_width / 2
        mid_y = p.width / 2

        for team_id in (0, 1):
            goal_x = state.attacking_goal_x(team_id)
            crossed = (goal_x >= p.length and pos.x >= p.length) or \
                      (goal_x <= 0.0 and pos.x <= 0.0)
            if crossed and abs(pos.y - mid_y) <= half_gw:
                return team_id
        return None

    # ── Out-of-bounds ─────────────────────────────────────────────────────────

    def check_out_of_bounds(self, ball: Ball, state: MatchState) -> Optional[GamePhase]:
        """
        Returns the restart phase if the ball is out of play, else None.
        Does NOT handle the case where the ball is a goal — call check_goal first.
        """
        pos = ball.position
        p = self.pitch

        if p.is_in_bounds(pos):
            return None

        # Over a touchline
        if pos.y < 0.0 or pos.y > p.width:
            return GamePhase.THROW_IN

        # Over a goal line (but not a goal)
        return self._goal_line_restart(pos, state)

    def _goal_line_restart(self, pos: Vec2, state: MatchState) -> GamePhase:
        """
        Ball out over the goal line (not in the goal).
        Corner kick if the defending team touched it last;
        goal kick if the attacking team touched it last.
        """
        last_team = state.last_touch_team

        # Find which team is ATTACKING toward the side the ball went out on
        if pos.x >= self.pitch.length:
            attacker = next(t for t in (0, 1) if state.attacking_goal_x(t) >= self.pitch.length)
        else:
            attacker = next(t for t in (0, 1) if state.attacking_goal_x(t) <= 0.0)
        defender = 1 - attacker

        if last_team is None or last_team == attacker:
            return GamePhase.GOAL_KICK    # attacker put it out → defender restarts
        else:
            return GamePhase.CORNER_KICK  # defender put it out → attacker gets corner

    # ── Restart position ──────────────────────────────────────────────────────

    def restart_info(
        self, phase: GamePhase, ball_pos: Vec2, state: MatchState
    ) -> tuple[Vec2, int]:
        """
        Returns (restart_position, team_awarded_restart) for the given phase.
        """
        p = self.pitch
        mid_y = p.width / 2

        if phase == GamePhase.THROW_IN:
            # Keep x where ball exited; y is on the touchline
            x = max(0.0, min(p.length, ball_pos.x))
            y = 0.0 if ball_pos.y <= 0.0 else p.width
            awarded_to = 1 - (state.last_touch_team or 0)
            return Vec2(x, y), awarded_to

        if phase == GamePhase.GOAL_KICK:
            # Defending team places the ball on their 6-yard line
            if ball_pos.x >= p.length / 2:
                defender = next(t for t in (0, 1) if state.attacking_goal_x(t) < p.length / 2)
                kick_x = p.length - 5.5
            else:
                defender = next(t for t in (0, 1) if state.attacking_goal_x(t) >= p.length / 2)
                kick_x = 5.5
            return Vec2(kick_x, mid_y), defender

        if phase == GamePhase.CORNER_KICK:
            # Attacking team takes corner from the nearest corner flag
            corner_x = p.length if ball_pos.x >= p.length / 2 else 0.0
            corner_y = p.width if ball_pos.y >= p.width / 2 else 0.0
            if corner_x >= p.length:
                attacker = next(t for t in (0, 1) if state.attacking_goal_x(t) >= p.length)
            else:
                attacker = next(t for t in (0, 1) if state.attacking_goal_x(t) <= 0.0)
            return Vec2(corner_x, corner_y), attacker

        if phase == GamePhase.KICKOFF:
            return p.center, state.restart_team or 0

        return p.center, 0

    # ── Time ─────────────────────────────────────────────────────────────────

    def check_time(self, state: MatchState) -> Optional[GamePhase]:
        """
        Checks whether a time-triggered phase change should occur.
        Returns the new phase or None. Does not mutate state.
        """
        if state.phase == GamePhase.FULL_TIME:
            return None
        cfg = state.config
        tick = state.tick
        et = state.et_period

        if et == 0:
            # Half-time fires at half_ticks during normal play (direction hasn't flipped yet)
            if (
                tick >= cfg.half_ticks
                and tick < cfg.total_ticks
                and state.phase == GamePhase.IN_PLAY
                and state.attack_direction[0] == 1
            ):
                return GamePhase.HALF_TIME
            if tick >= cfg.total_ticks:
                if cfg.extra_time and state.score[0] == state.score[1]:
                    return GamePhase.EXTRA_TIME_BREAK
                return GamePhase.FULL_TIME
        elif et == 1:
            if tick >= cfg.total_ticks + cfg.et_period_ticks:
                return GamePhase.EXTRA_TIME_BREAK
        elif et == 2:
            if tick >= cfg.total_ticks + 2 * cfg.et_period_ticks:
                if state.score[0] == state.score[1]:
                    return GamePhase.PENALTY_SHOOTOUT
                return GamePhase.FULL_TIME
        return None

    # ── apply_* helpers ───────────────────────────────────────────────────────
    # These mutate state. They're thin so callers don't have to repeat the logic.

    def apply_goal(self, scoring_team: int, state: MatchState) -> None:
        state.score[scoring_team] += 1
        state.phase = GamePhase.GOAL_SCORED
        state.phase_ticks_remaining = state.config.goal_pause_ticks
        state.restart_position = self.pitch.center
        state.restart_team = 1 - scoring_team   # conceding team kicks off

    def apply_kickoff(self, kicking_team: int, state: MatchState) -> None:
        state.phase = GamePhase.KICKOFF
        state.restart_position = self.pitch.center
        state.restart_team = kicking_team

    def apply_half_time(self, state: MatchState) -> None:
        state.phase = GamePhase.HALF_TIME
        state.flip_attack_directions()
        state.restart_position = self.pitch.center
        state.restart_team = 1   # team that didn't kick off first half restarts

    def apply_full_time(self, state: MatchState) -> None:
        state.phase = GamePhase.FULL_TIME

    def apply_restart(self, phase: GamePhase, ball_pos: Vec2, state: MatchState) -> None:
        """Compute and store restart info, then set the phase."""
        pos, team = self.restart_info(phase, ball_pos, state)
        state.restart_position = pos
        state.restart_team = team
        state.phase = phase

    # ── Stamina drain ─────────────────────────────────────────────────────────

    def drain_stamina(self, teams: list, state: MatchState) -> None:
        """
        Stamina drain and recovery each IN_PLAY tick.

        Carriers sprint constantly → drain at 2× base rate.
        Off-ball players jog/walk → drain at base but recover based on HP.
        HP/65 recovery factor means high-HP Pokémon barely tire off-ball;
        low-HP Pokémon still accumulate fatigue throughout the match.
        """
        if state.phase != GamePhase.IN_PLAY:
            return
        base = 1.0 / state.config.total_ticks
        _HP_BASELINE = 65.0
        for team in teams:
            for player in team.players:
                if player.has_ball:
                    player.stamina = max(0.0, player.stamina - base * 2.0)
                else:
                    recovery = base * 0.5 * (player.stats.hp / _HP_BASELINE)
                    player.stamina = min(
                        player.max_stamina,
                        max(0.0, player.stamina - base + recovery),
                    )
