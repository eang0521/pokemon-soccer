from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sim.core.entities import Vec2


class GamePhase(Enum):
    PRE_KICKOFF       = "pre_kickoff"       # before match start
    KICKOFF           = "kickoff"           # waiting for kick to be taken
    IN_PLAY           = "in_play"           # normal play
    GOAL_SCORED       = "goal_scored"       # brief pause after a goal
    THROW_IN          = "throw_in"          # ball out on touchline
    GOAL_KICK         = "goal_kick"         # attacker last touched → defender restarts
    CORNER_KICK       = "corner_kick"       # defender last touched → attacker restarts
    HALF_TIME         = "half_time"         # between halves
    EXTRA_TIME_BREAK  = "extra_time_break"  # start of ET or break between ET periods
    PENALTY_SHOOTOUT  = "penalty_shootout"  # penalty kick phase
    FULL_TIME         = "full_time"         # match over


@dataclass
class MatchConfig:
    duration_seconds: float = 40 * 60     # 2400 s — two 20-minute halves
    half_duration_seconds: float = 20 * 60
    tick_rate: int = 10                    # ticks per simulated second
    goal_pause_ticks: int = 30            # pause length after a goal (3 sim-seconds)
    pitch_length: float = 45.7            # must match Pitch.length; used by attacking_goal_x
    extra_time: bool = False              # enable extra time + penalties if tied at full time
    et_period_seconds: float = 0.0        # duration of each ET period (server sets this)

    @property
    def total_ticks(self) -> int:
        return int(self.duration_seconds * self.tick_rate)

    @property
    def half_ticks(self) -> int:
        return int(self.half_duration_seconds * self.tick_rate)

    @property
    def et_period_ticks(self) -> int:
        return int(self.et_period_seconds * self.tick_rate)


@dataclass
class MatchState:
    config: MatchConfig = field(default_factory=MatchConfig)
    tick: int = 0
    score: list[int] = field(default_factory=lambda: [0, 0])
    phase: GamePhase = GamePhase.PRE_KICKOFF

    # Current ball carrier (synced from Ball.carrier each tick)
    possession_team: Optional[int] = None
    possession_player: Optional[str] = None

    # Last player to touch the ball — needed for restart decisions
    last_touch_team: Optional[int] = None
    last_touch_player: Optional[str] = None

    # Set when a restart is pending (throw-in, corner, goal kick, kickoff)
    restart_position: Optional[Vec2] = None
    restart_team: Optional[int] = None

    # Countdown used during GOAL_SCORED pause
    phase_ticks_remaining: int = 0

    # 0 = normal time, 1 = first ET period, 2 = second ET period
    et_period: int = 0

    # Which direction each team attacks this half.
    # attack_direction[i] = +1 → team i shoots at high-x goal (x = length).
    # attack_direction[i] = -1 → team i shoots at low-x goal (x = 0).
    # Flips at half-time.
    attack_direction: list[int] = field(default_factory=lambda: [1, -1])

    # ── Derived time properties ───────────────────────────────────────────────

    @property
    def simulated_seconds(self) -> float:
        return self.tick / self.config.tick_rate

    @property
    def match_minute(self) -> int:
        return int(self.simulated_seconds / 60) + 1

    @property
    def is_first_half(self) -> bool:
        return self.tick < self.config.half_ticks

    @property
    def is_over(self) -> bool:
        return self.phase == GamePhase.FULL_TIME

    @property
    def score_str(self) -> str:
        return f"{self.score[0]}–{self.score[1]}"

    # ── Attack direction helpers ───────────────────────────────────────────────

    def attacking_goal_x(self, team_id: int) -> float:
        """X-coordinate of the goal that team_id is currently shooting at."""
        return self.config.pitch_length if self.attack_direction[team_id] == 1 else 0.0

    def flip_attack_directions(self) -> None:
        self.attack_direction = [-d for d in self.attack_direction]

    def __repr__(self) -> str:
        return (
            f"MatchState(tick={self.tick}, {self.match_minute}', "
            f"score={self.score_str}, phase={self.phase.value}, "
            f"poss=team{self.possession_team})"
        )
