"""
Match commentary generator.

Converts Events into human-readable one-line strings.
Templates use named placeholders: {player}, {target}, {minute}, {score}.
Each EventType gets a small pool so repeated events feel varied.
"""
from __future__ import annotations

import random
from typing import Optional

from sim.engine.events import Event, EventType


# ─── Template pools ──────────────────────────────────────────────────────────
# {player}  = primary actor (event.player.name)
# {target}  = secondary actor (event.target.name)
# {minute}  = match_minute
# {score}   = "H–A" string supplied by caller

_TEMPLATES: dict[EventType, list[str]] = {
    EventType.KICKOFF: [
        "{player} gets us underway at the centre circle.",
        "The referee blows the whistle — {player} kicks off!",
        "{player} starts the match. We're off!",
    ],
    EventType.GOAL: [
        "GOAL!! {player} finds the back of the net! {score}",
        "IT'S IN!! {player} scores — {score}!",
        "{player} slots it home — what a finish! {score}",
        "GOOOAL! {player} makes no mistake! {score}",
    ],
    EventType.SHOT: [
        "{player} has a go!",
        "{player} pulls the trigger!",
        "Shot from {player}!",
        "{player} lets fly from distance!",
    ],
    EventType.SAVE: [
        "Saved by {player}!",
        "{player} gets down to keep it out!",
        "Great stop from {player}!",
        "{player} dives to deny the effort!",
    ],
    EventType.SHOT_WIDE: [
        "{player}'s effort goes wide.",
        "Off target from {player}.",
        "That's too high from {player}.",
        "{player} can't hit the target this time.",
    ],
    EventType.PASS_COMPLETE: [
        "{player} finds {target}.",
        "{player} plays it to {target}.",
        "Good ball from {player} — {target} collects.",
        "{player} releases {target} with a crisp pass.",
    ],
    EventType.PASS_INTERCEPTED: [
        "{player}'s pass to {target} is cut out!",
        "Interception — {player} gives it away.",
        "The ball never reaches {target} — well read!",
        "{player} loses possession with a loose pass.",
    ],
    EventType.TACKLE_WON: [
        "{player} wins the ball from {target}!",
        "Great tackle from {player} — {target} is dispossessed.",
        "{player} times it perfectly to take it off {target}.",
        "Crunching challenge from {player} — {target} goes down!",
    ],
    EventType.TACKLE_LOST: [
        "{target} rides the challenge from {player}.",
        "{player}'s tackle attempt is beaten.",
        "{target} dances past {player}!",
    ],
    EventType.THROW_IN: [
        "{player} restarts play with a throw-in.",
        "Throw-in — {player} keeps it simple.",
    ],
    EventType.GOAL_KICK: [
        "{player} takes the goal kick.",
        "Goal kick for {player}'s team — long clearance coming.",
    ],
    EventType.CORNER_KICK: [
        "{player} swings in the corner.",
        "Corner kick — {player} delivers into the box.",
    ],
    EventType.POSSESSION_CHANGE: [
        "{player} picks up the loose ball.",
        "{player} pounces on the loose ball.",
        "{player} claims possession.",
    ],
    EventType.HALF_TIME: [
        "HALF TIME — {score}.",
        "The referee blows for half time. {score} at the break.",
        "And that's half time. {score}.",
    ],
    EventType.FULL_TIME: [
        "FULL TIME — {score}!",
        "The final whistle — {score}.",
        "That's it! Full time, {score}.",
    ],
}


def describe(event: Event, score: Optional[list[int]] = None) -> str:
    """
    Return a one-line commentary string for *event*.

    score: live [home_goals, away_goals] — embedded into GOAL/HALF_TIME/FULL_TIME lines.
    Template selection uses event.tick as seed so the same event always picks the same line.
    """
    templates = _TEMPLATES.get(event.event_type, ["[{event_type}]"])
    rng = random.Random(event.tick ^ hash(event.event_type.value))
    template = rng.choice(templates)

    score_str = f"{score[0]}–{score[1]}" if score else "?–?"

    return template.format(
        player=event.player.name if event.player else "—",
        target=event.target.name if event.target else "—",
        minute=event.match_minute,
        score=score_str,
        event_type=event.event_type.value,
    )


# ─── Stateful commentator ────────────────────────────────────────────────────

class Commentator:
    """
    Stateful commentary stream.

    Feed events in chronological order via `next()` or pass them all to
    `stream()`.  The commentator tracks the running score so templates that
    embed {score} always show the current tally.

    Scoring is inferred from GOAL events: event.player.team_id determines
    which team's tally increments.
    """

    def __init__(self, team_names: Optional[list[str]] = None) -> None:
        self.team_names = team_names or ["Home", "Away"]
        self.score: list[int] = [0, 0]

    def next(self, event: Event) -> tuple[int, str]:
        """
        Process one event.
        Updates internal score for GOAL events, then returns (match_minute, text).
        """
        if event.event_type == EventType.GOAL and event.player is not None:
            self.score[event.player.team_id] += 1

        return event.match_minute, describe(event, score=self.score)

    def stream(self, events: list[Event]) -> list[tuple[int, str]]:
        """Process a list of events and return all (minute, text) pairs."""
        return [self.next(e) for e in events]
