from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── 2D Vector ───────────────────────────────────────────────────────────────

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def magnitude_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> Vec2:
        mag = self.magnitude
        if mag < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def distance_to(self, other: Vec2) -> float:
        return (self - other).magnitude

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def lerp(self, other: Vec2, t: float) -> Vec2:
        t = max(0.0, min(1.0, t))
        return Vec2(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)


# ─── Pokémon base stats ───────────────────────────────────────────────────────

@dataclass
class PokemonStats:
    species: str
    pokedex_id: int
    types: list[str]
    hp: int
    attack: int
    defense: int
    sp_attack: int
    sp_defense: int
    speed: int
    moves: list[str]
    sprite_url: str = ""

    @classmethod
    def from_dict(cls, _key: str, data: dict) -> PokemonStats:
        s = data["base_stats"]
        return cls(
            species=data["species"],
            pokedex_id=data["pokedex_id"],
            types=data["types"],
            hp=s["hp"],
            attack=s["attack"],
            defense=s["defense"],
            sp_attack=s["sp_attack"],
            sp_defense=s["sp_defense"],
            speed=s["speed"],
            moves=data.get("moves", []),
            sprite_url=data.get("sprite_url", ""),
        )

    # Role suitability scores — tied directly to the in-game mechanics that
    # benefit each position.  HP is excluded here; it governs stamina capacity.
    @property
    def forward_score(self) -> float:
        # Attack → shot speed; Speed → escaping defenders / reaching position
        return (self.attack + self.speed) / 2.0

    @property
    def defender_score(self) -> float:
        # Defense → tackle success; Speed → tracking runners
        return (self.defense + self.speed) / 2.0

    @property
    def midfielder_score(self) -> float:
        # SpA → pass speed/range; SpD → reading lanes / interception
        return (self.sp_attack + self.sp_defense) / 2.0

    @property
    def goalkeeper_score(self) -> float:
        # Defense → blocking shots; SpD → positioning / angle reading
        return (self.defense + self.sp_defense) / 2.0

    @property
    def stamina_capacity(self) -> float:
        # HP → stamina pool size; baseline HP ~65 → ~0.83.
        # Higher HP = larger pool = lasts longer before fatigue kicks in.
        return 0.5 + (self.hp / 200.0)

    @property
    def ovr(self) -> float:
        # Weights reflect each stat's impact on in-game mechanics.
        # Calibrated so Arceus (120 in all stats) = exactly 100 OVR.
        raw = (
            0.12 * self.hp
            + 0.17 * self.attack
            + 0.19 * self.defense
            + 0.20 * self.sp_attack
            + 0.10 * self.sp_defense
            + 0.22 * self.speed
        )
        return raw / 1.2


# ─── Role ────────────────────────────────────────────────────────────────────

class Role(Enum):
    GOALKEEPER = "gk"
    DEFENDER = "def"
    MIDFIELDER = "mid"
    FORWARD = "fwd"


# ─── Player ──────────────────────────────────────────────────────────────────

@dataclass(eq=False)
class Player:
    """
    eq=False: use identity-based equality and hash (object.__eq__ / object.__hash__).
    Two Player instances are never equal just because their stats match.
    """
    name: str            # display name, e.g. "Pikachu"
    team_id: int         # 0 = home, 1 = away
    stats: PokemonStats
    role: Role
    position: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    stamina: float = 1.0    # current stamina; initialised to max_stamina in __post_init__
    has_ball: bool = False

    # Per-move state; populated lazily when moves are used
    move_cooldowns: dict[str, int] = field(default_factory=dict)  # ticks remaining
    move_pp: dict[str, int] = field(default_factory=dict)          # PP remaining
    active_effects: dict[str, int] = field(default_factory=dict)   # effect → ticks left

    def __post_init__(self) -> None:
        self.stamina = self.stats.stamina_capacity

    @property
    def max_stamina(self) -> float:
        return self.stats.stamina_capacity

    # ── Stamina factor ────────────────────────────────────────────────────────
    # Above 50% of max: full effectiveness.  Below 50%: linear drop to 0.4.
    def _stamina_factor(self) -> float:
        relative = self.stamina / self.max_stamina if self.max_stamina > 0 else 0.0
        if relative >= 0.5:
            return 1.0
        return 0.4 + 0.6 * (relative / 0.5)

    # ── Effective stats ───────────────────────────────────────────────────────
    @property
    def effective_speed(self) -> float:
        boost = self.active_effects.get("speed_boost_multiplier", 1.0)
        return self.stats.speed * self._stamina_factor() * boost

    @property
    def effective_attack(self) -> float:
        return self.stats.attack * self._stamina_factor()

    @property
    def effective_defense(self) -> float:
        return self.stats.defense * self._stamina_factor()

    @property
    def effective_sp_attack(self) -> float:
        return self.stats.sp_attack * self._stamina_factor()

    @property
    def effective_sp_defense(self) -> float:
        return self.stats.sp_defense * self._stamina_factor()

    @property
    def max_speed_mps(self) -> float:
        # Maps Pokémon Speed (30–150 range) onto field speed 4.0–10.0 m/s
        return 4.0 + (self.effective_speed / 150.0) * 6.0

    def __repr__(self) -> str:
        return f"Player({self.name}, {self.role.value}, team={self.team_id}, pos={self.position})"


# ─── Ball ─────────────────────────────────────────────────────────────────────

@dataclass
class Ball:
    position: Vec2 = field(default_factory=Vec2)
    velocity: Vec2 = field(default_factory=Vec2)
    carrier: Optional[Player] = None

    @property
    def is_loose(self) -> bool:
        return self.carrier is None

    def __repr__(self) -> str:
        if self.carrier:
            return f"Ball(pos={self.position}, carrier={self.carrier.name})"
        return f"Ball(pos={self.position}, vel={self.velocity}, loose)"


# ─── Team ─────────────────────────────────────────────────────────────────────

@dataclass
class Team:
    name: str
    team_id: int
    players: list[Player] = field(default_factory=list)

    def get_by_role(self, role: Role) -> list[Player]:
        return [p for p in self.players if p.role == role]

    @property
    def goalkeeper(self) -> Optional[Player]:
        gks = self.get_by_role(Role.GOALKEEPER)
        return gks[0] if gks else None

    @classmethod
    def from_roster(
        cls,
        _team_key: str,
        team_data: dict,
        team_id: int,
        pokemon_db: dict[str, PokemonStats],
    ) -> Team:
        pokemon_list = [pokemon_db[k] for k in team_data["roster"]]
        players = _assign_roles(pokemon_list, team_id)
        return cls(name=team_data["name"], team_id=team_id, players=players)

    @classmethod
    def from_custom_roster(
        cls,
        name: str,
        team_id: int,
        roster: list[str],
        pokemon_db: dict[str, PokemonStats],
        manual_roles: Optional[dict[str, str]] = None,
    ) -> Team:
        """Build a team from a picker selection, with auto or manual role assignment."""
        if manual_roles:
            role_map = {
                "gk":  Role.GOALKEEPER,
                "def": Role.DEFENDER,
                "mid": Role.MIDFIELDER,
                "fwd": Role.FORWARD,
            }
            players = [
                Player(
                    name=pokemon_db[k].species,
                    team_id=team_id,
                    stats=pokemon_db[k],
                    role=role_map[manual_roles[k].lower()],
                )
                for k in roster
            ]
        else:
            players = _assign_roles([pokemon_db[k] for k in roster], team_id)
        return cls(name=name, team_id=team_id, players=players)

    @property
    def team_ovr(self) -> float:
        if not self.players:
            return 0.0
        return sum(p.stats.ovr for p in self.players) / len(self.players)

    def __repr__(self) -> str:
        lines = [f"Team: {self.name}"]
        for p in self.players:
            lines.append(f"  [{p.role.value:3s}]  {p.name}")
        return "\n".join(lines)


_OUTFIELD_SLOTS = [
    Role.DEFENDER, Role.DEFENDER,
    Role.MIDFIELDER, Role.MIDFIELDER,
    Role.FORWARD,
]

_ROLE_SCORE: dict[Role, str] = {
    Role.GOALKEEPER: "goalkeeper_score",
    Role.DEFENDER:   "defender_score",
    Role.MIDFIELDER: "midfielder_score",
    Role.FORWARD:    "forward_score",
}


def _assign_roles(pokemon_list: list[PokemonStats], team_id: int) -> list[Player]:
    """
    Optimal role assignment: GK is always the best GK candidate (uniquely high-stakes),
    then the remaining five are globally optimised over DEF×2 / MID×2 / FWD via all
    5! = 120 permutations. This beats purely greedy without producing absurd lineups.
    """
    pool = list(pokemon_list)

    gk_stats = max(pool, key=lambda p: p.goalkeeper_score)
    pool.remove(gk_stats)

    best_total = -1.0
    best_outfield: list[tuple[PokemonStats, Role]] = []

    for perm in itertools.permutations(pool):
        total = sum(
            getattr(poke, _ROLE_SCORE[role])
            for poke, role in zip(perm, _OUTFIELD_SLOTS)
        )
        if total > best_total:
            best_total = total
            best_outfield = list(zip(perm, _OUTFIELD_SLOTS))

    assignments = [(gk_stats, Role.GOALKEEPER)] + best_outfield
    return [
        Player(name=stats.species, team_id=team_id, stats=stats, role=role)
        for stats, role in assignments
    ]


# ─── Pitch ────────────────────────────────────────────────────────────────────

@dataclass
class Pitch:
    # 6v6 pitch dimensions (meters).  ≈50 yd × 36 yd; goal 18.5 ft wide.
    length: float = 45.7    # x-axis: 0 (left/home goal) → 45.7 (right/away goal)  ≈ 50 yd
    width: float = 32.9     # y-axis: 0 (bottom) → 32.9 (top)                     ≈ 36 yd
    goal_width: float = 5.64  # 18.5 ft
    goal_height: float = 1.98  # 6.5 ft

    @property
    def center(self) -> Vec2:
        return Vec2(self.length / 2, self.width / 2)

    @property
    def left_goal_center(self) -> Vec2:
        return Vec2(0.0, self.width / 2)

    @property
    def right_goal_center(self) -> Vec2:
        return Vec2(self.length, self.width / 2)

    def attacking_goal(self, team_id: int) -> Vec2:
        """Goal the team is shooting at."""
        return self.right_goal_center if team_id == 0 else self.left_goal_center

    def defending_goal(self, team_id: int) -> Vec2:
        """Goal the team is protecting."""
        return self.left_goal_center if team_id == 0 else self.right_goal_center

    def is_in_bounds(self, pos: Vec2) -> bool:
        return 0.0 <= pos.x <= self.length and 0.0 <= pos.y <= self.width

    def is_goal(self, pos: Vec2) -> Optional[int]:
        """
        Returns the team_id that just scored, or None if no goal.
        A goal is scored when the ball crosses the goal line within the goal width.
        """
        half_gw = self.goal_width / 2
        mid_y = self.width / 2
        if pos.x >= self.length and abs(pos.y - mid_y) <= half_gw:
            return 0   # team 0 scored (attacked right)
        if pos.x <= 0.0 and abs(pos.y - mid_y) <= half_gw:
            return 1   # team 1 scored (attacked left)
        return None

    def clamp(self, pos: Vec2) -> Vec2:
        return Vec2(
            max(0.0, min(self.length, pos.x)),
            max(0.0, min(self.width, pos.y)),
        )

    def third_boundaries(self, team_id: int) -> dict[str, tuple[float, float]]:
        """X-ranges for defensive/middle/attacking thirds, from team_id's perspective."""
        L = self.length
        if team_id == 0:
            return {"defensive": (0.0, L / 3), "middle": (L / 3, 2 * L / 3), "attacking": (2 * L / 3, L)}
        else:
            return {"defensive": (2 * L / 3, L), "middle": (L / 3, 2 * L / 3), "attacking": (0.0, L / 3)}

    def __repr__(self) -> str:
        return f"Pitch({self.length}m × {self.width}m)"
