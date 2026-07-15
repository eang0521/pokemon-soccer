#!/usr/bin/env python3
"""
Fetch all 1025 national-dex base-form Pokemon from PokeAPI and write
sim/data/pokemon_stats.json.

Only IDs 1-1025 are fetched; PokeAPI assigns those IDs exclusively to
base forms (alternate/regional forms get IDs above 10000).

Run once from the project root:
    py scripts/populate_all.py
"""
import json
import ssl
import time
import urllib.request
from pathlib import Path

# PokeAPI is served behind Cloudflare which blocks headless requests.
# The api-data repo on GitHub hosts the same static JSON and is freely accessible.
POKE_DATA_BASE = "https://raw.githubusercontent.com/PokeAPI/api-data/master/data/api/v2/pokemon"
SPRITE_BASE    = "https://img.pokemondb.net/sprites/home/normal"
OUT_PATH       = Path(__file__).parent.parent / "sim" / "data" / "pokemon_stats.json"

# Names that require special handling (special chars, punctuation, hyphen intent).
DISPLAY_NAMES: dict[str, str] = {
    # Gen 1
    "nidoran-f":  "Nidoran♀",   # ♀
    "nidoran-m":  "Nidoran♂",   # ♂
    "farfetchd":  "Farfetch’d", # '
    "mr-mime":    "Mr. Mime",
    # Gen 2
    "ho-oh":      "Ho-Oh",
    # Gen 4
    "mime-jr":    "Mime Jr.",
    "porygon-z":  "Porygon-Z",
    # Gen 6
    "flabebe":    "Flabébé",  # Flabébé  (e + combining acute)
    # Gen 7
    "type-null":  "Type: Null",
    "jangmo-o":   "Jangmo-o",
    "hakamo-o":   "Hakamo-o",
    "kommo-o":    "Kommo-o",
    "tapu-koko":  "Tapu Koko",
    "tapu-lele":  "Tapu Lele",
    "tapu-bulu":  "Tapu Bulu",
    "tapu-fini":  "Tapu Fini",
    # Gen 8
    "sirfetchd":  "Sirfetch’d",
    "mr-rime":    "Mr. Rime",
    # Gen 9 – Ruinous quartet (hyphenated in official names)
    "wo-chien":   "Wo-Chien",
    "chien-pao":  "Chien-Pao",
    "ting-lu":    "Ting-Lu",
    "chi-yu":     "Chi-Yu",
}

SSL_CTX = ssl._create_unverified_context()


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20, context=SSL_CTX) as r:
        return json.loads(r.read())


def get_moves(pokemon_data: dict) -> list[str]:
    moves: list[str] = []
    for entry in pokemon_data.get("moves", []):
        for vgd in entry.get("version_group_details", []):
            if vgd.get("move_learn_method", {}).get("name") == "level-up":
                moves.append(entry["move"]["name"].replace("-", "_"))
                break
        if len(moves) >= 4:
            break
    return moves if moves else ["tackle", "growl"]


def species_name(api_name: str) -> str:
    if api_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[api_name]
    return api_name.replace("-", " ").title()


def main() -> None:
    # Windows terminals default to cp1252; reconfigure stdout so ♀/♂ print cleanly.
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    result: dict = {}
    failed: list[int] = []

    for dex_id in range(1, 1026):
        print(f"  #{dex_id:04d} ...", end=" ", flush=True)
        try:
            data = fetch_json(f"{POKE_DATA_BASE}/{dex_id}/index.json")
        except Exception as exc:
            print(f"ERROR: {exc}")
            failed.append(dex_id)
            time.sleep(1.0)   # back off on error
            continue

        key    = data["name"]
        types  = [t["type"]["name"].title() for t in data["types"]]
        raw_st = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

        result[key] = {
            "species":    species_name(key),
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
            "sprite_url": f"{SPRITE_BASE}/{key}.png",
        }
        print(f"{result[key]['species']} ({'/'.join(types)})")
        time.sleep(0.02)

    if not result:
        print("\nNo data fetched — file not written (existing data preserved).")
        return

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(result)} Pokemon to {OUT_PATH}")
    if failed:
        print(f"Failed IDs (re-run to retry): {failed}")


if __name__ == "__main__":
    main()
