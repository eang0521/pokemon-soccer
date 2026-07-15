"""
FastAPI web server for the Pokemon Soccer viewer.

Run:
    pip install fastapi uvicorn
    cd C:/root/poke-soccer
    py -m uvicorn web.server:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sim.core.entities import Pitch, PokemonStats, Team
from sim.core.match_state import MatchConfig
from sim.engine.simulator import Simulator

DATA_DIR   = ROOT / "sim" / "data"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Pokemon Soccer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SPRITE_BASE = "https://img.pokemondb.net/sprites/home/normal"


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _load_raw() -> dict:
    with open(DATA_DIR / "pokemon_stats.json", encoding="utf-8") as f:
        return json.load(f)

def _load_pokemon_db() -> dict[str, PokemonStats]:
    return {k: PokemonStats.from_dict(k, v) for k, v in _load_raw().items()}


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/pokemon")
def get_pokemon():
    """All 1025 base-form Pokemon sorted by Pokedex ID, with sprite URLs and role scores."""
    raw = _load_raw()
    db  = _load_pokemon_db()

    all_pokemon = {}
    for key, data in raw.items():
        stats = db[key]
        all_pokemon[key] = {
            "key":        key,
            "name":       data["species"],
            "pokedex_id": data["pokedex_id"],
            "types":      data["types"],
            "sprite_url": data.get("sprite_url", "%s/%s.png" % (SPRITE_BASE, key)),
            "stats":      data["base_stats"],
            "ovr": round(stats.ovr, 1),
            "scores": {
                "gk":  round(stats.goalkeeper_score, 1),
                "def": round(stats.defender_score,   1),
                "mid": round(stats.midfielder_score, 1),
                "fwd": round(stats.forward_score,    1),
            },
        }
    return JSONResponse(all_pokemon)


class SimRequest(BaseModel):
    home_roster: list[str]
    away_roster: list[str]
    home_name:   str = "Team 1"
    away_name:   str = "Team 2"
    home_roles:  Optional[dict[str, str]] = None  # {pokemon_key: "gk"|"def"|"mid"|"fwd"}
    away_roles:  Optional[dict[str, str]] = None
    duration_minutes: int = 40
    seed:         int = 42
    sample_every: int = 3
    extra_time:   bool = False


@app.post("/api/simulate")
def simulate(req: SimRequest):
    """Run a full simulation and return frame data for the viewer."""
    if len(req.home_roster) != 6:
        raise HTTPException(400, "home_roster must have exactly 6 Pokemon")
    if len(req.away_roster) != 6:
        raise HTTPException(400, "away_roster must have exactly 6 Pokemon")

    pokemon_db = _load_pokemon_db()

    for key in req.home_roster + req.away_roster:
        if key not in pokemon_db:
            raise HTTPException(400, "Unknown Pokemon: %s" % key)

    home_team = Team.from_custom_roster(
        req.home_name, 0, req.home_roster, pokemon_db, req.home_roles
    )
    away_team = Team.from_custom_roster(
        req.away_name, 1, req.away_roster, pokemon_db, req.away_roles
    )

    pitch      = Pitch()
    duration_s = req.duration_minutes * 60
    et_period_s = max(1, round(req.duration_minutes / 6)) * 60 if req.extra_time else 0.0
    cfg = MatchConfig(
        duration_seconds=duration_s,
        half_duration_seconds=duration_s // 2,
        extra_time=req.extra_time,
        et_period_seconds=et_period_s,
    )

    sim = Simulator(teams=[home_team, away_team], pitch=pitch, config=cfg, seed=req.seed)
    _event_log, frames = sim.run_capturing(sample_every=req.sample_every)

    def player_meta(p):
        return {
            "name":       p.name,
            "role":       p.role.value,
            "types":      p.stats.types,
            "sprite_url": p.stats.sprite_url if hasattr(p.stats, "sprite_url") else "",
        }

    return JSONResponse({
        "home": {
            "name":    home_team.name,
            "ovr":     round(home_team.team_ovr, 1),
            "players": [player_meta(p) for p in home_team.players],
        },
        "away": {
            "name":    away_team.name,
            "ovr":     round(away_team.team_ovr, 1),
            "players": [player_meta(p) for p in away_team.players],
        },
        "pitch": {
            "length":     pitch.length,
            "width":      pitch.width,
            "goal_width": pitch.goal_width,
        },
        "duration_minutes": req.duration_minutes,
        "sample_every":     req.sample_every,
        "tick_rate":        cfg.tick_rate,
        "total_ticks":      sim.state.tick,
        "frames":           frames,
        "goals":            sim._goals,
        "went_to_et":       sim.state.et_period > 0,
        "penalty_kicks":    sim._penalty_kicks,
        "penalty_winner":   sim._penalty_winner,
    })
