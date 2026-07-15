"""
Event log for the match simulator.

Events capture discrete game moments (goals, passes, tackles, etc.) and form
the basis for commentary (Step 6) and match statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sim.core.entities import Player, Vec2


class EventType(Enum):
    KICKOFF           = "kickoff"
    PASS_COMPLETE     = "pass_complete"
    PASS_INTERCEPTED  = "pass_intercepted"
    SHOT              = "shot"
    GOAL              = "goal"
    SAVE              = "save"
    SHOT_WIDE         = "shot_wide"
    TACKLE_WON        = "tackle_won"
    TACKLE_LOST       = "tackle_lost"
    THROW_IN          = "throw_in"
    GOAL_KICK         = "goal_kick"
    CORNER_KICK       = "corner_kick"
    POSSESSION_CHANGE = "possession_change"
    OFFSIDE           = "offside"
    HALF_TIME         = "half_time"
    FULL_TIME         = "full_time"
    EXTRA_TIME        = "extra_time"
    PENALTY_KICK      = "penalty_kick"


@dataclass
class Event:
    tick: int
    event_type: EventType
    position: Vec2
    player: Optional[Player] = None    # primary actor
    target: Optional[Player] = None    # secondary actor (pass target, tackle victim, etc.)
    details: dict = field(default_factory=dict)

    @property
    def match_minute(self) -> int:
        """Simulated match minute (10 ticks/s × 60 s = 600 ticks/min)."""
        return self.tick // 600 + 1

    def __repr__(self) -> str:
        actor = f" [{self.player.name}]" if self.player else ""
        return f"Event({self.match_minute}' {self.event_type.value}{actor})"


class EventLog:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def log(
        self,
        tick: int,
        event_type: EventType,
        position: Vec2,
        player: Optional[Player] = None,
        target: Optional[Player] = None,
        **details,
    ) -> Event:
        e = Event(tick=tick, event_type=event_type, position=position,
                  player=player, target=target, details=details)
        self.events.append(e)
        return e

    def get(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type == event_type]

    @property
    def goals(self) -> list[Event]:
        return self.get(EventType.GOAL)

    def summary(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(e.event_type.value for e in self.events))
