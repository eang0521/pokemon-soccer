"""
ASCII terminal renderer for match state.

render_frame(state, ball, teams, pitch=None) → str
    Produces a box-drawn pitch with player positions, score, and clock.

render_event_line(minute, text) → str
    Formats a (minute, commentary) pair for the match log.

render_scoreboard(state, teams) → str
    Compact end-of-match result card.
"""
from __future__ import annotations

from typing import Optional

from sim.core.entities import Ball, Pitch, Player, Role, Team, Vec2
from sim.core.match_state import MatchState


# ─── Grid constants ───────────────────────────────────────────────────────────

_COLS = 60   # inner grid columns  (≈ 1.75 m / col)
_ROWS = 20   # inner grid rows     (≈ 3.4  m / row)

# Role symbol: home team UPPERCASE, away team lowercase
_ROLE_SYM: dict[Role, str] = {
    Role.GOALKEEPER: "G",
    Role.DEFENDER:   "D",
    Role.MIDFIELDER: "M",
    Role.FORWARD:    "F",
}


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def _to_col(x: float, pitch_length: float = 105.0) -> int:
    c = int(x / pitch_length * _COLS)
    return min(_COLS - 1, max(0, c))


def _to_row(y: float, pitch_width: float = 68.0) -> int:
    # Row 0 = top touchline (y = pitch_width), row _ROWS-1 = bottom (y = 0)
    r = int((1.0 - y / pitch_width) * _ROWS)
    return min(_ROWS - 1, max(0, r))


# ─── Main renderer ────────────────────────────────────────────────────────────

def render_frame(
    state: MatchState,
    ball: Ball,
    teams: list[Team],
    pitch: Optional[Pitch] = None,
    label: str = "",
) -> str:
    """
    Return a multi-line ASCII string showing the current match state.

    Layout (64 chars wide):
      ┌──header──┐
      │  pitch   │   (20 rows × 60 cols inner grid)
      └──────────┘
      legend line
    """
    if pitch is None:
        from sim.core.entities import Pitch as _Pitch
        pitch = _Pitch()

    # ── Build empty grid ─────────────────────────────────────────────────────
    grid: list[list[str]] = [[" "] * _COLS for _ in range(_ROWS)]

    # Centre line
    mid_col = _to_col(pitch.length / 2, pitch.length)
    for r in range(_ROWS):
        grid[r][mid_col] = "│"

    # Goal markers: bracket the goal opening on each end line
    goal_half_rows = max(1, round(pitch.goal_width / pitch.width * _ROWS / 2))
    mid_row = _ROWS // 2
    for dr in range(-goal_half_rows, goal_half_rows + 1):
        r = mid_row + dr
        if 0 <= r < _ROWS:
            if grid[r][0] == " ":
                grid[r][0] = "["
            if grid[r][_COLS - 1] == " ":
                grid[r][_COLS - 1] = "]"

    # ── Place players ─────────────────────────────────────────────────────────
    carrier = ball.carrier
    for team in teams:
        for p in team.players:
            col = _to_col(p.position.x, pitch.length)
            row = _to_row(p.position.y, pitch.width)
            base_sym = _ROLE_SYM[p.role]
            if team.team_id == 1:
                base_sym = base_sym.lower()
            if p is carrier:
                # Mark carrier with a circled symbol (ASCII: @ for home, # for away)
                sym = "@" if team.team_id == 0 else "#"
            else:
                sym = base_sym
            grid[row][col] = sym

    # ── Place loose ball ──────────────────────────────────────────────────────
    if carrier is None:
        col = _to_col(ball.position.x, pitch.length)
        row = _to_row(ball.position.y, pitch.width)
        grid[row][col] = "o"

    # ── Assemble lines ────────────────────────────────────────────────────────
    home_name  = teams[0].name if teams else "Home"
    away_name  = teams[1].name if len(teams) > 1 else "Away"
    score_str  = f"{state.score[0]}–{state.score[1]}"
    clock_str  = f"{state.match_minute}'"

    # Header fits inside _COLS+2 chars (for "│ " + content + " │")
    header_inner = f"  {home_name}  {score_str}  {away_name}"
    header_inner = f"{header_inner:<{_COLS - 6}}{clock_str:>6}"

    border_h = "─" * (_COLS + 2)
    top_border = "┌" + border_h + "┐"
    mid_border = "├" + border_h + "┤"
    bot_border = "└" + border_h + "┘"

    lines: list[str] = []
    lines.append(top_border)
    lines.append("│ " + header_inner + " │")
    lines.append(mid_border)
    for row in grid:
        lines.append("│ " + "".join(row) + " │")
    lines.append(bot_border)

    legend = (
        "  G/D/M/F = " + home_name
        + "   g/d/m/f = " + away_name
        + "   @ / # = carrier   o = loose ball"
    )
    lines.append(legend)
    if label:
        lines.append(f"  {label}")

    return "\n".join(lines)


# ─── Event log line ───────────────────────────────────────────────────────────

def render_event_line(minute: int, text: str) -> str:
    """Format a (minute, commentary) pair as a fixed-width log line."""
    return f"  {minute:3d}'  {text}"


# ─── Scoreboard ──────────────────────────────────────────────────────────────

def render_scoreboard(
    state: MatchState,
    teams: list[Team],
    event_summary: Optional[dict[str, int]] = None,
) -> str:
    """
    Produce a compact end-of-match result card.
    event_summary: optional dict from EventLog.summary().
    """
    home_name = teams[0].name if teams else "Home"
    away_name = teams[1].name if len(teams) > 1 else "Away"
    score_str = f"{state.score[0]}–{state.score[1]}"
    width = 52

    lines: list[str] = []
    lines.append("┌" + "─" * width + "┐")
    lines.append("│" + " FULL TIME ".center(width) + "│")
    lines.append("├" + ("─" * width) + "┤")

    result_line = f"  {home_name}  {score_str}  {away_name}  "
    lines.append("│" + result_line.center(width) + "│")

    if event_summary:
        lines.append("├" + ("─" * width) + "┤")
        stats = [
            ("Shots",   event_summary.get("shot", 0)),
            ("Goals",   event_summary.get("goal", 0)),
            ("Saves",   event_summary.get("save", 0)),
            ("Passes",  event_summary.get("pass_complete", 0)),
            ("Tackles", event_summary.get("tackle_won", 0)),
        ]
        for label, val in stats:
            lines.append("│" + f"  {label:<10} {val:>4}".ljust(width) + "│")

    lines.append("└" + "─" * width + "┘")
    return "\n".join(lines)
