"""Her savaşı (bir satır) sabit uzunlukta bir sayı vektörüne çeviren fonksiyonlar.

Kullanım:
    from src.features import build_features
    X = build_features(train_df)   # battle_id indeksli DataFrame
"""

import pandas as pd

STATUSES = ["slp", "par", "frz", "tox", "psn", "brn"]
BOOSTS = ["atk", "def", "spa", "spd", "spe"]


def last_known_state(timeline, side):
    """Her Pokémon'un timeline'da SON görüldüğü andaki durumu: {isim: state}."""
    last = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state
    return last


def side_features(timeline, side):
    """Bir oyuncunun (p1 veya p2) 30 turdaki gidişatını özetler."""
    f = {}
    last = last_known_state(timeline, side)

    # --- 30. turdaki takım durumu ---
    f["seen"] = len(last)                                   # sahaya çıkan farklı Pokémon
    f["fainted"] = sum(s["status"] == "fnt" for s in last.values())
    hp_seen = sum(s["hp_pct"] for s in last.values())
    f["hp_total"] = hp_seen + (6 - len(last))               # görülmeyenler tam can varsayılır
    for st in STATUSES:
        f[f"n_{st}"] = sum(s["status"] == st for s in last.values())

    # --- son turda sahadaki Pokémon ---
    active = timeline[-1][f"{side}_pokemon_state"]
    f["active_hp"] = active["hp_pct"]
    f["active_boost_sum"] = sum(active["boosts"][b] for b in BOOSTS)
    for b in BOOSTS:
        f[f"active_boost_{b}"] = active["boosts"][b]

    # --- 30 tur boyunca davranış ---
    names = [t[f"{side}_pokemon_state"]["name"] for t in timeline]
    f["switches"] = sum(a != b for a, b in zip(names, names[1:]))
    f["no_move_turns"] = sum(t[f"{side}_move_details"] is None for t in timeline)
    moves = [t[f"{side}_move_details"] for t in timeline if t[f"{side}_move_details"]]
    f["status_moves"] = sum(m["category"] == "status" for m in moves)
    f["avg_base_power"] = sum(m["base_power"] for m in moves) / len(moves) if moves else 0.0

    # --- bu oyuncunun RAKİBE verdiği hasar ---
    # Rakip aynı Pokémon iken HP düşüşlerini topluyoruz (switch anlarını atlıyoruz)
    opp = "p2" if side == "p1" else "p1"
    dmg = 0.0
    for prev, cur in zip(timeline, timeline[1:]):
        a, b = prev[f"{opp}_pokemon_state"], cur[f"{opp}_pokemon_state"]
        if a["name"] == b["name"]:
            dmg += max(a["hp_pct"] - b["hp_pct"], 0.0)
    f["damage_dealt"] = dmg
    return f


def battle_features(battle):
    """Tek bir savaş (JSON satırı / dict) için özellik sözlüğü."""
    tl = battle["battle_timeline"]
    f = {"battle_id": battle["battle_id"]}
    p1 = side_features(tl, "p1")
    p2 = side_features(tl, "p2")
    f.update({f"p1_{k}": v for k, v in p1.items()})
    f.update({f"p2_{k}": v for k, v in p2.items()})

    # Farklar: pozitifse P1 lehine
    for k in ["hp_total", "active_hp", "active_boost_sum", "damage_dealt", "seen"]:
        f[f"diff_{k}"] = p1[k] - p2[k]
    f["diff_fainted"] = p2["fainted"] - p1["fainted"]   # rakibin bayılanı fazlaysa P1 lehine
    f["diff_disabled"] = (p2["n_slp"] + p2["n_frz"]) - (p1["n_slp"] + p1["n_frz"])
    return f


def build_features(df):
    """DataFrame (train veya test) → özellik tablosu (battle_id indeksli)."""
    rows = [battle_features(b) for b in df.to_dict("records")]
    return pd.DataFrame(rows).set_index("battle_id")
