#!/usr/bin/env python3
"""
Pokémon 6v6 Soccer Simulation — cumulative smoke tests (Steps 1–4).
"""
import json
import math
import random
import sys
from pathlib import Path

# Ensure Unicode output works on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sim.core.entities import Ball, Pitch, Player, PokemonStats, Role, Team, Vec2
from sim.core.match_state import GamePhase, MatchConfig, MatchState
from sim.core.rules import Rules
from sim.ai.considerations import geometric_mean
from sim.ai.decision import ActionCandidate, Decider

DATA_DIR = Path(__file__).parent / "sim" / "data"


def load_pokemon_db() -> dict[str, PokemonStats]:
    with open(DATA_DIR / "pokemon_stats.json") as f:
        raw = json.load(f)
    return {key: PokemonStats.from_dict(key, data) for key, data in raw.items()}


def load_teams(pokemon_db: dict[str, PokemonStats]) -> list[Team]:
    with open(DATA_DIR / "teams.json") as f:
        raw = json.load(f)
    return [
        Team.from_roster(key, data, team_id, pokemon_db)
        for team_id, (key, data) in enumerate(raw.items())
    ]


STAT_BAR_WIDTH = 20

def stat_bar(value: int, max_val: int = 200) -> str:
    filled = round(value / max_val * STAT_BAR_WIDTH)
    return "[" + "█" * filled + "░" * (STAT_BAR_WIDTH - filled) + f"] {value:3d}"


def print_team(team: Team) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {team.name}  (team_id={team.team_id})")
    print(f"{'─' * 60}")
    role_order = [Role.GOALKEEPER, Role.DEFENDER, Role.MIDFIELDER, Role.FORWARD]
    for role in role_order:
        for p in team.get_by_role(role):
            s = p.stats
            print(f"\n  [{p.role.value:3s}]  {p.name} ({'/'.join(s.types)})")
            print(f"         HP   {stat_bar(s.hp)}")
            print(f"         Atk  {stat_bar(s.attack)}")
            print(f"         Def  {stat_bar(s.defense)}")
            print(f"         SpA  {stat_bar(s.sp_attack)}")
            print(f"         SpD  {stat_bar(s.sp_defense)}")
            print(f"         Spe  {stat_bar(s.speed)}")
            print(f"         Field speed: {p.max_speed_mps:.1f} m/s  |  "
                  f"Moves: {', '.join(s.moves)}")


def print_suitability_table(pokemon_db: dict[str, PokemonStats]) -> None:
    print(f"\n{'─' * 60}")
    print("  Role suitability scores (used for position assignment)")
    print(f"{'─' * 60}")
    header = f"  {'Species':<12} {'GK':>6} {'DEF':>6} {'MID':>6} {'FWD':>6}"
    print(header)
    print(f"  {'-'*12} {'------':>6} {'------':>6} {'------':>6} {'------':>6}")
    for p in pokemon_db.values():
        print(
            f"  {p.species:<12} "
            f"{p.goalkeeper_score:>6.1f} "
            f"{p.defender_score:>6.1f} "
            f"{p.midfielder_score:>6.1f} "
            f"{p.forward_score:>6.1f}"
        )


def main() -> None:
    print("=" * 60)
    print("  Pokémon 6v6 Soccer — Step 1: Data Models")
    print("=" * 60)

    pokemon_db = load_pokemon_db()
    print(f"\nLoaded {len(pokemon_db)} Pokémon from pokemon_stats.json")

    teams = load_teams(pokemon_db)
    pitch = Pitch()
    print(f"Pitch: {pitch}")
    print(f"  Left goal (home defending):  {pitch.left_goal_center}")
    print(f"  Right goal (away defending): {pitch.right_goal_center}")

    print_suitability_table(pokemon_db)

    for team in teams:
        print_team(team)

    print(f"\n{'─' * 60}")
    print("  Sanity checks")
    print(f"{'─' * 60}")
    for team in teams:
        gk = team.goalkeeper
        assert gk is not None, f"{team.name} has no goalkeeper!"
        assert len(team.get_by_role(Role.DEFENDER)) == 2
        assert len(team.get_by_role(Role.MIDFIELDER)) == 2
        assert len(team.get_by_role(Role.FORWARD)) == 1
        print(f"  {team.name}: OK — GK={gk.name}, "
              f"DEF={[p.name for p in team.get_by_role(Role.DEFENDER)]}, "
              f"MID={[p.name for p in team.get_by_role(Role.MIDFIELDER)]}, "
              f"FWD={team.get_by_role(Role.FORWARD)[0].name}")

    test_player = teams[0].players[0]
    test_player.stamina = 0.0
    assert abs(test_player._stamina_factor() - 0.4) < 1e-6
    test_player.stamina = 0.5
    assert abs(test_player._stamina_factor() - 1.0) < 1e-6
    test_player.stamina = 1.0
    assert abs(test_player._stamina_factor() - 1.0) < 1e-6
    print("  Stamina factor: OK")

    assert pitch.is_goal(pitch.right_goal_center) == 0
    assert pitch.is_goal(pitch.left_goal_center) == 1
    assert pitch.is_goal(pitch.center) is None
    print("  Pitch.is_goal geometry: OK")

    print(f"\n{'=' * 60}")
    print("  Step 1 complete.")
    print("=" * 60)

    step2_verify(teams, pitch)


# ─── Step 2 ───────────────────────────────────────────────────────────────────

def step2_verify(teams: list[Team], pitch: Pitch) -> None:
    print(f"\n{'=' * 60}")
    print("  Step 2: Match State & Rules")
    print("=" * 60)

    rules = Rules(pitch)
    cfg = MatchConfig()

    state = MatchState(config=cfg)
    assert state.phase == GamePhase.PRE_KICKOFF
    assert state.score == [0, 0]
    assert state.tick == 0
    assert state.simulated_seconds == 0.0
    assert state.match_minute == 0
    assert state.is_first_half
    assert not state.is_over
    assert state.attacking_goal_x(0) == pitch.length
    assert state.attacking_goal_x(1) == 0.0
    print("  1. Initial MatchState:              OK")

    state.tick = cfg.tick_rate * 60
    assert state.match_minute == 1
    assert state.simulated_seconds == 60.0

    state.tick = cfg.half_ticks - 1
    assert state.is_first_half
    assert rules.check_time(state) is None

    state.tick = cfg.half_ticks
    state.phase = GamePhase.IN_PLAY
    assert rules.check_time(state) == GamePhase.HALF_TIME

    state.tick = cfg.total_ticks
    assert rules.check_time(state) == GamePhase.FULL_TIME
    state.tick = 0
    state.phase = GamePhase.IN_PLAY
    print("  2. Time tracking:                   OK")

    ball = Ball()
    state.attack_direction = [1, -1]

    ball.position = Vec2(106.0, pitch.width / 2)
    assert rules.check_goal(ball, state) == 0

    ball.position = Vec2(-1.0, pitch.width / 2)
    assert rules.check_goal(ball, state) == 1

    ball.position = pitch.center
    assert rules.check_goal(ball, state) is None

    ball.position = Vec2(106.0, 2.0)
    assert rules.check_goal(ball, state) is None

    state.flip_attack_directions()
    assert state.attacking_goal_x(0) == 0.0
    ball.position = Vec2(-1.0, pitch.width / 2)
    assert rules.check_goal(ball, state) == 0
    state.flip_attack_directions()
    print("  3. Goal detection (incl. direction flip): OK")

    state.last_touch_team = 0

    ball.position = Vec2(22.0, -1.0)
    assert rules.check_out_of_bounds(ball, state) == GamePhase.THROW_IN

    ball.position = Vec2(22.0, 34.0)   # above 32.9 m width
    assert rules.check_out_of_bounds(ball, state) == GamePhase.THROW_IN

    ball.position = Vec2(22.0, 16.0)   # centre of new pitch
    assert rules.check_out_of_bounds(ball, state) is None

    state.last_touch_team = 0
    ball.position = Vec2(47.0, 5.0)    # past right goal line (45.7 m), not in goal
    assert rules.check_out_of_bounds(ball, state) == GamePhase.GOAL_KICK

    state.last_touch_team = 1
    assert rules.check_out_of_bounds(ball, state) == GamePhase.CORNER_KICK

    state.last_touch_team = 1
    ball.position = Vec2(-1.0, 5.0)
    assert rules.check_out_of_bounds(ball, state) == GamePhase.GOAL_KICK

    state.last_touch_team = 0
    assert rules.check_out_of_bounds(ball, state) == GamePhase.CORNER_KICK
    print("  4. Out-of-bounds classification:    OK")

    state.last_touch_team = 0
    pos, team = rules.restart_info(GamePhase.THROW_IN, Vec2(22.0, -2.0), state)
    assert abs(pos.x - 22.0) < 0.1 and pos.y == 0.0 and team == 1

    state.last_touch_team = 0
    pos, team = rules.restart_info(GamePhase.GOAL_KICK, Vec2(106.0, 34.0), state)
    assert abs(pos.x - (pitch.length - 5.5)) < 0.1
    assert team == 1

    state.last_touch_team = 1
    pos, team = rules.restart_info(GamePhase.CORNER_KICK, Vec2(106.0, 50.0), state)
    assert pos.x == pitch.length and pos.y == pitch.width
    assert team == 0
    print("  5. Restart positions:               OK")

    state.score = [0, 0]
    state.phase = GamePhase.IN_PLAY
    rules.apply_goal(0, state)
    assert state.score == [1, 0]
    assert state.phase == GamePhase.GOAL_SCORED
    assert state.phase_ticks_remaining == cfg.goal_pause_ticks
    assert state.restart_team == 1

    rules.apply_goal(1, state)
    assert state.score == [1, 1]
    assert state.restart_team == 0
    print("  6. apply_goal:                      OK")

    state.phase = GamePhase.IN_PLAY
    state.tick = cfg.half_ticks
    transition = rules.check_time(state)
    assert transition == GamePhase.HALF_TIME
    rules.apply_half_time(state)
    assert state.phase == GamePhase.HALF_TIME
    assert state.attack_direction == [-1, 1]
    assert state.attacking_goal_x(0) == 0.0
    assert state.attacking_goal_x(1) == pitch.length
    print("  7. Half-time direction flip:        OK")

    state2 = MatchState(config=cfg)
    state2.phase = GamePhase.IN_PLAY
    for t in teams:
        for p in t.players:
            p.stamina = 1.0
            p.has_ball = False
    teams[0].players[0].has_ball = True

    drain_ticks = cfg.total_ticks // 3   # use 1/3 of match so stamina stays well above 0
    for _ in range(drain_ticks):
        rules.drain_stamina(teams, state2)

    carrier = teams[0].players[0]
    regular = teams[0].players[1]
    assert carrier.stamina < regular.stamina
    assert all(p.stamina > 0 for t in teams for p in t.players)
    drain_ratio = (1.0 - carrier.stamina) / (1.0 - regular.stamina)
    assert 1.4 < drain_ratio < 1.6, f"Expected ~1.5× drain ratio, got {drain_ratio:.2f}"
    print(f"  8. Stamina drain ({drain_ticks} ticks):      OK  "
          f"(carrier {carrier.stamina:.4f} vs regular {regular.stamina:.4f})")

    state3 = MatchState(config=cfg)
    state3.tick = cfg.total_ticks
    state3.phase = GamePhase.IN_PLAY
    assert rules.check_time(state3) == GamePhase.FULL_TIME
    rules.apply_full_time(state3)
    assert state3.is_over
    assert state3.phase == GamePhase.FULL_TIME
    print("  9. Full-time detection:             OK")

    print(f"\n{'─' * 60}")
    print("  MatchState repr (end of tests):")
    print(f"  {state}")
    print(f"\n{'=' * 60}")
    print("  Step 2 complete.")
    print("=" * 60)

    step3_verify(teams, pitch)
    step4_verify(teams, pitch)


# ─── Step 3 ───────────────────────────────────────────────────────────────────

def _place(player: Player, x: float, y: float) -> None:
    player.position = Vec2(x, y)


def step3_verify(teams: list[Team], pitch: Pitch) -> None:
    print(f"\n{'=' * 60}")
    print("  Step 3: Utility AI — Vertical Slice")
    print("=" * 60)

    kanto, johto = teams
    by_name = {p.name: p for team in teams for p in team.players}

    assert abs(geometric_mean([0.5, 0.5, 0.5]) - 0.5) < 1e-6
    assert geometric_mean([1.0, 1.0, 0.001]) < 0.12
    assert geometric_mean([1.0, 1.0, 1.0]) == 1.0
    assert geometric_mean([]) == 0.0
    print("  Geometric mean properties: OK")

    _place(by_name["Pikachu"],  33.0, 16.5)
    _place(by_name["Alakazam"], 26.0,  8.7)
    _place(by_name["Gengar"],   28.0, 24.0)
    _place(by_name["Rhydon"],   11.0, 16.5)
    _place(by_name["Machamp"],  12.0,  8.7)
    _place(by_name["Slowbro"],   2.5, 16.5)

    _place(by_name["Steelix"],  44.5, 16.5)
    _place(by_name["Tyranitar"], 36.0, 13.5)
    _place(by_name["Umbreon"],   37.5, 24.0)
    _place(by_name["Espeon"],    31.0, 12.5)
    _place(by_name["Ampharos"],  32.0,  6.8)
    _place(by_name["Heracross"], 24.0, 16.5)

    pikachu = by_name["Pikachu"]
    pikachu.has_ball = True
    ball = Ball(position=Vec2(pikachu.position.x, pikachu.position.y), carrier=pikachu)

    state = MatchState()
    state.phase = GamePhase.IN_PLAY

    teammates = [p for p in kanto.players if p is not pikachu]
    opponents = list(johto.players)

    decider = Decider(pitch)
    candidates = decider.all_scored(pikachu, ball, teammates, opponents, state)

    print(f"\nScenario 1 — Pikachu (FWD) at {pikachu.position}, ~12 m from goal")
    print(f"  Opponents: Steelix GK, Tyranitar+Umbreon DEF, Espeon pressuring")
    _print_candidates(candidates)

    labels = {c.label for c in candidates}
    assert "shoot" in labels
    assert "dribble" in labels
    assert any("pass" in l for l in labels)
    assert len(candidates) == 7
    assert all(0.0 <= c.score <= 1.0 for c in candidates)
    print("  Candidate structure: OK (7 candidates, all scores in [0,1])")

    shoot_c = next(c for c in candidates if c.action_type == "shoot")
    assert "shot_odds" in shoot_c.breakdown and "angle" in shoot_c.breakdown
    dribble_c = next(c for c in candidates if c.action_type == "dribble")
    assert "space_ahead" in dribble_c.breakdown and "matchup" in dribble_c.breakdown
    pass_c = next(c for c in candidates if c.action_type == "pass")
    assert "openness" in pass_c.breakdown and "lane_safety" in pass_c.breakdown
    print("  Breakdown fields: OK")

    assert shoot_c.breakdown["shot_odds"] < 0.3
    assert shoot_c.breakdown["angle"] > 0.9
    print(f"  Shot quality check: shot_odds={shoot_c.breakdown['shot_odds']:.3f} "
          f"(low vs Steelix Def 200), angle={shoot_c.breakdown['angle']:.3f} (central): OK")

    random.seed(42)
    choice_a = decider.decide(pikachu, ball, teammates, opponents, state)
    random.seed(42)
    choice_b = decider.decide(pikachu, ball, teammates, opponents, state)
    assert choice_a.label == choice_b.label
    print(f"  Determinism (same seed): OK  →  chose '{choice_a.label}' "
          f"(score={choice_a.score:.3f})")

    results = set()
    for seed in range(30):
        random.seed(seed)
        results.add(decider.decide(pikachu, ball, teammates, opponents, state).label)
    assert len(results) > 1
    print(f"  Randomness across 30 seeds: OK  ({len(results)} distinct choices seen: "
          f"{sorted(results)})")

    print(f"\nScenario 2 — Pikachu under heavy pressure (opponents within 2 m)")
    for i, opp in enumerate(opponents):
        angle = i * (2 * math.pi / len(opponents))
        opp.position = Vec2(
            pikachu.position.x + 2.0 * math.cos(angle),
            pikachu.position.y + 2.0 * math.sin(angle),
        )
    candidates_pressed = decider.all_scored(pikachu, ball, teammates, opponents, state)
    _print_candidates(candidates_pressed)

    dribble_pressed = next(c for c in candidates_pressed if c.action_type == "dribble")
    assert dribble_pressed.breakdown["space_ahead"] < 0.35
    pass_scores_pressed = [c.score for c in candidates_pressed if c.action_type == "pass"]
    assert min(pass_scores_pressed) < 0.25
    print("  Under pressure, dribble space and pass lane safety collapse: OK")

    print(f"\nScenario 3 — Pikachu in space, only GK ahead (all outfield opp at 50 m)")
    _place(by_name["Steelix"], 103.0, 34.0)
    for opp in opponents:
        if opp.name != "Steelix":
            _place(opp, 50.0 + opponents.index(opp) * 3, 10.0)
    candidates_open = decider.all_scored(pikachu, ball, teammates, opponents, state)
    _print_candidates(candidates_open)

    dribble_open = next(c for c in candidates_open if c.action_type == "dribble")
    assert dribble_open.breakdown["space_ahead"] > 0.6
    print("  In open space, dribble space score is high: OK")

    print(f"\n{'=' * 60}")
    print("  Step 3 complete.")
    print("=" * 60)


def _print_candidates(candidates: list[ActionCandidate]) -> None:
    COL = 24
    print(f"\n  {'Action':<{COL}} {'Score':>6}  Breakdown")
    print(f"  {'─' * COL} {'─' * 6}  {'─' * 50}")
    for c in candidates:
        bd = "  ".join(f"{k}={v:.3f}" for k, v in c.breakdown.items())
        star = " ◀" if c is candidates[0] else ""
        print(f"  {c.label:<{COL}} {c.score:>6.3f}  {bd}{star}")


# ─── Step 4 ───────────────────────────────────────────────────────────────────

def step4_verify(teams: list[Team], pitch: Pitch) -> None:
    from sim.ai.formations import FORMATION_1221, build_home_positions, home_distance_penalty
    from sim.ai.steering import arrive, apply_steering

    print(f"\n{'=' * 60}")
    print("  Step 4: Steering + Formation Movement")
    print("=" * 60)

    kanto, johto = teams

    # Reset every player to pitch centre, full stamina
    for team in teams:
        for p in team.players:
            p.position = Vec2(pitch.length / 2, pitch.width / 2)
            p.velocity = Vec2(0.0, 0.0)
            p.stamina = 1.0
            p.has_ball = False

    state = MatchState()
    state.phase = GamePhase.IN_PLAY

    # Ball at centre — no carrier for this test (pure off-ball movement)
    ball = Ball(position=Vec2(pitch.length / 2, pitch.width / 2))

    # ── Print home zone layout ────────────────────────────────────────────────
    print(f"\n  Formation: {FORMATION_1221.name}")
    for team in teams:
        direction = "→ right (x=105)" if state.attack_direction[team.team_id] == 1 else "← left (x=0)"
        print(f"\n  {team.name}  ({direction})")
        homes = build_home_positions(team, pitch, state, ball)
        for p in team.players:
            h = homes[p]
            print(f"    [{p.role.value:3s}]  {p.name:<12}  home: ({h.x:5.1f}, {h.y:4.1f})")

    # ── Steering simulation ───────────────────────────────────────────────────
    N_TICKS = 300
    DT = 0.1   # 10 ticks per simulated second

    for tick in range(N_TICKS):
        state.tick = tick
        for team in teams:
            homes = build_home_positions(team, pitch, state, ball)
            players = list(team.players)
            for p in players:
                desired = arrive(p.position, homes[p], p.max_speed_mps, slow_radius=5.0)
                apply_steering(p, desired, players, DT, pitch=pitch)

    final_homes = {
        **build_home_positions(kanto, pitch, state, ball),
        **build_home_positions(johto, pitch, state, ball),
    }

    # ── Convergence table ─────────────────────────────────────────────────────
    print(f"\n  After {N_TICKS} ticks ({N_TICKS * DT:.0f} sim-seconds):")
    print(f"\n  {'Player':<12}  {'Home':>16}   {'Final':>16}  {'Dist':>6}")
    print(f"  {'─' * 12}  {'─' * 16}   {'─' * 16}  {'─' * 6}")

    max_dist = 0.0
    for team in teams:
        for p in team.players:
            home = final_homes[p]
            dist = p.position.distance_to(home)
            max_dist = max(max_dist, dist)
            ok = "✓" if dist < 2.0 else "✗"
            print(f"  {p.name:<12}  ({home.x:5.1f},{home.y:4.1f})   "
                  f"({p.position.x:5.1f},{p.position.y:4.1f})  {dist:5.2f}m {ok}")

    assert max_dist < 2.0, f"Worst convergence = {max_dist:.2f}m"
    print(f"\n  Convergence OK — worst offset {max_dist:.2f} m (threshold 2.0 m)")

    # ── Separation check ──────────────────────────────────────────────────────
    min_sep = float("inf")
    for team in teams:
        for i, a in enumerate(team.players):
            for b in team.players[i + 1:]:
                d = a.position.distance_to(b.position)
                min_sep = min(min_sep, d)

    assert min_sep > 1.5, f"Players overlapping: min separation {min_sep:.2f}m"
    print(f"  Separation OK — minimum same-team gap {min_sep:.2f} m (threshold 1.5 m)")

    # ── Ball-tracking test ────────────────────────────────────────────────────
    ball_top = Ball(position=Vec2(52.5, 58.0))
    ball_mid = Ball(position=Vec2(52.5, 34.0))

    def_top = [build_home_positions(kanto, pitch, state, ball_top)[p]
               for p in kanto.get_by_role(Role.DEFENDER)]
    def_mid = [build_home_positions(kanto, pitch, state, ball_mid)[p]
               for p in kanto.get_by_role(Role.DEFENDER)]

    for top_h, mid_h, p in zip(def_top, def_mid, kanto.get_by_role(Role.DEFENDER)):
        assert top_h.y > mid_h.y, \
            f"{p.name} home y should rise when ball is at y=58"
    print("  Ball-tracking OK — DEF home zones shift with ball position")

    # ── Zone-penalty integration check ───────────────────────────────────────
    def_player = kanto.get_by_role(Role.DEFENDER)[0]
    def_home = final_homes[def_player]

    penalty_at_home = home_distance_penalty(def_player, def_home)
    assert penalty_at_home > 0.95, f"Expected ≈1.0 at home, got {penalty_at_home:.3f}"

    original_pos = def_player.position
    def_player.position = Vec2(def_home.x + 30.0, def_home.y)
    penalty_far = home_distance_penalty(def_player, def_home)
    def_player.position = original_pos
    assert penalty_far < 0.35, f"Expected <0.35 at 30m from home, got {penalty_far:.3f}"
    print(f"  Zone-penalty OK — at home: {penalty_at_home:.3f}, at 30m: {penalty_far:.3f}")

    print(f"\n{'=' * 60}")
    print("  Step 4 complete. Ready for Step 5.")
    print("=" * 60)

    step5_verify(teams, pitch)


# ─── Step 5 ───────────────────────────────────────────────────────────────────

def step5_verify(teams: list[Team], pitch: Pitch) -> None:
    from sim.engine.simulator import Simulator
    from sim.engine.events import EventType

    print(f"\n{'=' * 60}")
    print("  Step 5: Full Match Simulation")
    print("=" * 60)

    # Reset all players to full stamina before the match
    for team in teams:
        for p in team.players:
            p.stamina = 1.0
            p.has_ball = False
            p.position = Vec2(0.0, 0.0)
            p.velocity = Vec2(0.0, 0.0)

    sim = Simulator(teams=teams, pitch=pitch, seed=42)
    print("\n  Running full 40-minute match (seed=42)...")
    event_log = sim.run(verbose=True)
    state = sim.state

    # ── Basic completion checks ───────────────────────────────────────────────
    assert state.is_over, "Match never reached FULL_TIME"
    full_time_events = event_log.get(EventType.FULL_TIME)
    assert len(full_time_events) == 1, f"Expected 1 FULL_TIME event, got {len(full_time_events)}"
    half_time_events = event_log.get(EventType.HALF_TIME)
    assert len(half_time_events) == 1, f"Expected 1 HALF_TIME event, got {len(half_time_events)}"
    print("\n  Match completed, HALF_TIME and FULL_TIME events logged: OK")

    # ── Score vs goal-event consistency ──────────────────────────────────────
    goal_events = event_log.goals
    scored_total = state.score[0] + state.score[1]
    assert len(goal_events) == scored_total, (
        f"GOAL event count ({len(goal_events)}) != score sum ({scored_total})"
    )
    print(f"  Final score: {teams[0].name} {state.score[0]}–{state.score[1]} {teams[1].name}")
    print(f"  Goal events match score tally ({scored_total} goals): OK")

    # ── Match has events of expected types ────────────────────────────────────
    summary = event_log.summary()
    assert summary.get("pass_complete", 0) > 0, "No passes completed"
    assert summary.get("shot", 0) > 0, "No shots taken"
    assert summary.get("kickoff", 0) >= 1, "No kickoff logged"
    print(f"  Event types logged: {sorted(summary.keys())}")
    print(f"  Shots={summary.get('shot', 0)}, Passes={summary.get('pass_complete', 0)}, "
          f"Tackles={summary.get('tackle_won', 0)}, Saves={summary.get('save', 0)}: OK")

    # ── Stamina has drained ───────────────────────────────────────────────────
    for team in teams:
        for p in team.players:
            assert p.stamina < 0.99, f"{p.name} stamina hasn't drained at all ({p.stamina:.4f})"
    print("  All players' stamina drained during the match: OK")

    # ── Direction flip occurred ───────────────────────────────────────────────
    assert state.attack_direction == [-1, 1], (
        f"Expected directions to have flipped, got {state.attack_direction}"
    )
    print(f"  Attack direction correctly flipped for 2nd half: OK")

    # ── Per-match stats ───────────────────────────────────────────────────────
    print(f"\n  Match statistics:")
    print(f"  {'Event':<24} {'Count':>6}")
    print(f"  {'─' * 24} {'─' * 6}")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v:>6}")

    if goal_events:
        print(f"\n  Goal scorers:")
        from collections import Counter
        scorers = Counter(e.player.name for e in goal_events if e.player)
        for name, count in scorers.most_common():
            print(f"    {name}: {count}")

    # ── Stamina table ─────────────────────────────────────────────────────────
    print(f"\n  Final stamina:")
    for team in teams:
        print(f"  {team.name}:")
        for p in sorted(team.players, key=lambda x: x.stamina):
            bar = "[" + "█" * round(p.stamina * 20) + "░" * (20 - round(p.stamina * 20)) + "]"
            print(f"    {p.name:<12} {bar}  {p.stamina:.3f}")

    print(f"\n{'=' * 60}")
    print("  Step 5 complete.")
    print("=" * 60)

    step6_verify(teams, pitch)


# ─── Step 6 ───────────────────────────────────────────────────────────────────

def step6_verify(teams: list[Team], pitch: Pitch) -> None:
    from sim.engine.simulator import Simulator
    from sim.engine.events import EventType, Event
    from sim.render.commentary import describe, Commentator
    from sim.render.terminal import render_frame, render_event_line, render_scoreboard

    print(f"\n{'=' * 60}")
    print("  Step 6: Commentary & Terminal Render")
    print("=" * 60)

    kanto, johto = teams

    # ── 1. Commentary coverage — every EventType must produce a non-empty string ─
    print("\n  1. Commentary coverage")
    print(f"  {'EventType':<26} {'Sample output'}")
    print(f"  {'─' * 26} {'─' * 40}")

    dummy_player = kanto.players[0]
    dummy_target = johto.players[0]
    score = [1, 0]

    for et in EventType:
        fake_event = Event(
            tick=2700,
            event_type=et,
            position=Vec2(52.5, 34.0),
            player=dummy_player,
            target=dummy_target,
        )
        text = describe(fake_event, score=score)
        assert text, f"Empty commentary for {et.value}"
        print(f"  {et.value:<26} {text}")

    print("  All EventTypes produce commentary: OK")

    # ── 2. Pitch renderer — static formation scene ────────────────────────────
    print("\n  2. Pitch renderer — static scene at kickoff positions")

    # Place players in formation (reuse what step 4 left behind from the last
    # step5_verify run; positions may have drifted — reset cleanly).
    for team in teams:
        for p in team.players:
            p.stamina = 1.0
            p.has_ball = False
            p.velocity = Vec2(0.0, 0.0)

    from sim.ai.formations import build_home_positions
    from sim.core.match_state import GamePhase

    static_state = MatchState()
    static_state.phase = GamePhase.IN_PLAY
    static_state.score = [2, 1]
    static_state.tick = 27000  # ~45'
    ball = Ball()
    ball.position = Vec2(pitch.center.x, pitch.center.y)

    for team in teams:
        home = build_home_positions(team, pitch, static_state, ball)
        for p in team.players:
            p.position = home[p]

    # Give ball to Kanto's forward
    fwd = kanto.get_by_role(Role.FORWARD)[0]
    ball.carrier = fwd
    ball.position = Vec2(fwd.position.x, fwd.position.y)
    fwd.has_ball = True

    frame = render_frame(static_state, ball, teams, pitch)
    print()
    print(frame)

    # Verify frame contains expected player symbols
    assert "G" in frame, "Kanto GK symbol missing"
    assert "g" in frame, "Johto GK symbol missing"
    assert "@" in frame, "Ball carrier marker missing"
    assert "│" in frame, "Centre line missing"
    print("  Player symbols and centre line present: OK")

    # ── 3. Mini-match commentary replay ──────────────────────────────────────
    print("\n  3. Mini-match commentary replay (first 10 minutes, seed=42)")

    for team in teams:
        for p in team.players:
            p.stamina = 1.0
            p.has_ball = False
            p.position = Vec2(0.0, 0.0)
            p.velocity = Vec2(0.0, 0.0)

    from sim.core.match_state import MatchConfig
    short_cfg = MatchConfig(
        duration_seconds=600,       # 10 sim-minutes
        half_duration_seconds=300,  # 5-minute halves
        tick_rate=10,
        goal_pause_ticks=30,
    )
    sim = Simulator(teams=teams, pitch=pitch, config=short_cfg, seed=42)
    log = sim.run(verbose=False)
    mini_state = sim.state

    commentator = Commentator(team_names=[kanto.name, johto.name])

    # Show highlights only: goals, phase changes, first kickoff
    HIGHLIGHTS = {EventType.GOAL, EventType.HALF_TIME, EventType.FULL_TIME, EventType.KICKOFF}
    highlight_events = [e for e in log.events if e.event_type in HIGHLIGHTS]
    # Cap at first kickoff + all goals + phase changes to keep output concise
    shown: list = []
    kickoff_done = False
    for e in highlight_events:
        if e.event_type == EventType.KICKOFF:
            if not kickoff_done:
                shown.append(e)
                kickoff_done = True
        else:
            shown.append(e)

    print()
    for minute, text in commentator.stream(shown):
        print(render_event_line(minute, text))

    # ── 4. Final scoreboard ───────────────────────────────────────────────────
    print()
    print(render_scoreboard(mini_state, teams, event_summary=log.summary()))

    # ── 5. Score/commentary consistency check ─────────────────────────────────
    goals_in_log = log.goals
    assert len(goals_in_log) == mini_state.score[0] + mini_state.score[1]
    print("\n  Goal count vs score tally: OK")
    assert mini_state.is_over
    print("  Match reached FULL_TIME: OK")

    # Commentator's tracked score should match simulation score after GOAL events
    # (commentator only sees NOTABLE events, not all — so just verify no crash)
    print("  Commentary stream produced no errors: OK")

    print(f"\n{'=' * 60}")
    print("  Step 6 complete.")
    print("=" * 60)

    step7_verify(teams, pitch)


# ─── Step 7 ───────────────────────────────────────────────────────────────────

def step7_verify(teams: list[Team], pitch: Pitch) -> None:
    from sim.engine import physics
    from sim.engine.simulator import Simulator
    from sim.engine.events import EventType

    print(f"\n{'=' * 60}")
    print("  Step 7: Ball Physics")
    print("=" * 60)

    kanto, johto = teams
    fwd = kanto.get_by_role(Role.FORWARD)[0]

    # ── 1. Physics unit tests ─────────────────────────────────────────────────
    print("\n  1. Physics unit tests")

    ps = physics.pass_speed(fwd)
    ss = physics.shot_speed(fwd)
    assert 10.0 <= ps <= 20.0, f"pass_speed out of range: {ps}"
    assert 15.0 <= ss <= 30.0, f"shot_speed out of range: {ss}"
    print(f"  pass_speed({fwd.name}) = {ps:.1f} m/s, shot_speed = {ss:.1f} m/s: OK")

    # kick sets ball velocity in the right direction
    ball_a = Ball()
    ball_a.position = Vec2(20.0, 34.0)
    physics.kick(ball_a, Vec2(80.0, 34.0), 15.0)
    assert abs(ball_a.velocity.x - 15.0) < 0.01
    assert abs(ball_a.velocity.y) < 0.01
    print(f"  kick (horizontal): vx={ball_a.velocity.x:.2f}, vy={ball_a.velocity.y:.3f}: OK")

    # step advances and decelerates
    ball_b = Ball()
    ball_b.position = Vec2(0.0, 0.0)
    ball_b.velocity = Vec2(10.0, 0.0)
    physics.step(ball_b, 0.1)
    assert 0.9 < ball_b.position.x < 1.1
    assert ball_b.velocity.x < 10.0
    print(f"  step(dt=0.1): x={ball_b.position.x:.3f}, vx={ball_b.velocity.x:.3f}: OK")

    # step: ball eventually stops
    ball_c = Ball()
    ball_c.position = Vec2(0.0, 0.0)
    ball_c.velocity = Vec2(5.0, 0.0)
    for _ in range(200):
        physics.step(ball_c, 0.1)
    assert ball_c.velocity.x < physics.MIN_SPEED + 0.01
    print(f"  step to rest: final vx={ball_c.velocity.x:.4f}: OK")

    # goal_crossing: horizontal shot from x=38, vx=20 → travels up to 80m → reaches 45.7
    ball_d = Ball()
    ball_d.position = Vec2(38.0, 16.0)
    ball_d.velocity = Vec2(20.0, 0.0)
    cross = physics.goal_crossing(ball_d, pitch.length)
    assert cross is not None, f"Ball at x=38 vx=20 should reach x={pitch.length} (7.7m < 80m max)"
    y_cross, t_cross = cross
    assert abs(y_cross - 16.0) < 0.5
    assert t_cross > 0.0
    print(f"  goal_crossing: y={y_cross:.2f}, t={t_cross:.2f}s: OK")

    # goal_crossing: ball moving away — None
    ball_e = Ball()
    ball_e.position = Vec2(38.0, 16.0)
    ball_e.velocity = Vec2(-20.0, 0.0)
    assert physics.goal_crossing(ball_e, pitch.length) is None
    print(f"  goal_crossing (moving away): None: OK")

    # goal_crossing: too slow to reach — None (vx=1, v²/(2a)=0.2m, 7.7m to goal)
    ball_f = Ball()
    ball_f.position = Vec2(38.0, 16.0)
    ball_f.velocity = Vec2(1.0, 0.0)
    assert physics.goal_crossing(ball_f, pitch.length) is None
    print(f"  goal_crossing (too slow): None: OK")

    # can_reach
    assert physics.can_reach(Vec2(0.0, 0.0), 8.0, Vec2(3.0, 0.0), 1.0, reach=1.5)
    assert not physics.can_reach(Vec2(0.0, 0.0), 3.0, Vec2(30.0, 0.0), 1.0, reach=1.5)
    print(f"  can_reach: OK")

    # ── 2. 10-minute match with physics ──────────────────────────────────────
    print("\n  2. 10-minute match with ball physics (seed=42)")

    for team in teams:
        for p in team.players:
            p.stamina = 1.0
            p.has_ball = False
            p.position = Vec2(0.0, 0.0)
            p.velocity = Vec2(0.0, 0.0)

    from sim.core.match_state import MatchConfig
    short_cfg = MatchConfig(
        duration_seconds=600,
        half_duration_seconds=300,
        tick_rate=10,
        goal_pause_ticks=30,
    )
    sim = Simulator(teams=teams, pitch=pitch, config=short_cfg, seed=42)
    log = sim.run(verbose=False)
    state = sim.state
    summary = log.summary()

    assert state.is_over, "Match never reached FULL_TIME"
    goal_events = log.goals
    scored_total = state.score[0] + state.score[1]
    assert len(goal_events) == scored_total, (
        f"GOAL events ({len(goal_events)}) ≠ score sum ({scored_total})"
    )
    assert summary.get("shot", 0) > 0, "No shots taken"
    assert summary.get("pass_complete", 0) > 0, "No passes completed"
    assert summary.get("half_time", 0) == 1
    assert summary.get("full_time", 0) == 1

    print(f"\n  Final score: {kanto.name} {state.score[0]}–{state.score[1]} {johto.name}")
    print(f"\n  Event breakdown:")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"    {k:<24} {v:>5}")

    # ── 3. Event quality ──────────────────────────────────────────────────────
    print("\n  3. Event quality checks")

    pass_complete = log.get(EventType.PASS_COMPLETE)
    if pass_complete:
        pc = pass_complete[0]
        assert pc.player is not None and pc.target is not None
        print(f"  PASS_COMPLETE: {pc.player.name} → {pc.target.name} @ {pc.match_minute}': OK")

    saves = log.get(EventType.SAVE)
    if saves:
        sv = saves[0]
        assert sv.player is not None
        assert sv.player.role == Role.GOALKEEPER, (
            f"SAVE logged for non-GK: {sv.player.role}"
        )
        print(f"  SAVE by GK: {sv.player.name} @ {sv.match_minute}': OK")
    else:
        print(f"  No SAVE events in this match (GK may not have been tested): noted")

    print(f"\n  Shots={summary.get('shot',0)}, "
          f"Pass completions={summary.get('pass_complete',0)}, "
          f"Interceptions={summary.get('pass_intercepted',0)}, "
          f"Saves={summary.get('save',0)}")

    print(f"\n{'=' * 60}")
    print("  Step 7 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
