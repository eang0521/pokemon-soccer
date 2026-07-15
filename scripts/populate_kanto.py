#!/usr/bin/env python3
"""
Fetch all 151 Kanto Pokemon from PokeAPI (SSL-verify disabled for local dev)
and write sim/data/pokemon_stats.json.

Run once from the project root:
    py scripts/populate_kanto.py
"""
import json
import ssl
import time
import urllib.request
from pathlib import Path

POKE_API    = "https://pokeapi.co/api/v2/pokemon"
SPRITE_BASE = "https://img.pokemondb.net/sprites/home/normal"
OUT_PATH    = Path(__file__).parent.parent / "sim" / "data" / "pokemon_stats.json"

DISPLAY_NAMES = {
    "nidoran-f": "Nidoran F",
    "nidoran-m": "Nidoran M",
    "mr-mime":   "Mr. Mime",
    "farfetchd": "Farfetch'd",
}

# Unverified SSL context for environments with corporate MITM proxies
SSL_CTX = ssl._create_unverified_context()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15, context=SSL_CTX) as r:
        return json.loads(r.read())


def get_moves(pokemon_data: dict) -> list:
    moves = []
    for entry in pokemon_data.get("moves", []):
        for vgd in entry.get("version_group_details", []):
            if vgd.get("move_learn_method", {}).get("name") == "level-up":
                moves.append(entry["move"]["name"].replace("-", "_"))
                break
        if len(moves) >= 4:
            break
    return moves if moves else ["tackle", "growl"]


def main():
    result = {}

    for dex_id in range(1, 152):
        print("  #%03d ..." % dex_id, end=" ")
        try:
            data = fetch_json("%s/%d" % (POKE_API, dex_id))
        except Exception as e:
            print("ERROR: %s" % e)
            continue

        key     = data["name"]
        species = DISPLAY_NAMES.get(key, key.replace("-", " ").title())
        types   = [t["type"]["name"].title() for t in data["types"]]
        raw_st  = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

        result[key] = {
            "species":    species,
            "pokedex_id": dex_id,
            "types":      types,
            "base_stats": {
                "hp":         raw_st["hp"],
                "attack":     raw_st["attack"],
                "defense":    raw_st["defense"],
                "sp_attack":  raw_st["special-attack"],
                "sp_defense": raw_st["special-defense"],
                "speed":      raw_st["speed"],
            },
            "moves":      get_moves(data),
            "sprite_url": "%s/%s.png" % (SPRITE_BASE, key),
        }
        print("%s (%s)" % (species, "/".join(types)))
        time.sleep(0.07)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nSaved %d Pokemon to %s" % (len(result), OUT_PATH))


if __name__ == "__main__":
    main()
