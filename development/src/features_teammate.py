"""Takım arkadaşımızın özellikleri (v5'te bizim özelliklerle birleştirildi).

Kaynak: takım arkadaşının `FDS Challenge.ipynb` notebook'u (`extract_features`, 73 sütun).
Mantık ve sütun adları (`f_...`) korundu. Yarışma kurallarına uyum için üç değişiklik yapıldı:

1. Pokedex sadece train'den oluşturulur (orijinalde train + test).
2. Elle yazılmış hamle grupları veriden çıkarılan listelerle değiştirildi:
   - recovery_moves → `load_healing_moves` (src/features.py)
   - setup_moves    → `load_setup_moves` (kullanıldıktan sonra kullananın boost'u artan hamleler)
3. Elle gruplanmış "phaze" (roar, whirlwind) ve "cleric" (healbell, aromatherapy) özellikleri,
   hamle başına ayrı sütunlara bölündü: bunlar artık bir tablo değil, görülen hamle adının kendisi.

Kullanım:
    pokedex = build_pokedex(train_df)
    setup = load_setup_moves([train_df])
    f = teammate_features(battle, pokedex, chart, healing_moves, setup)
"""

import numpy as np

SINGLE_MOVES = ["roar", "whirlwind", "healbell", "aromatherapy", "rapidspin"]


def build_pokedex(df):
    """Tür adı → Pokémon detayı (stat, tip). Sadece verilen veriden (train)."""
    pokedex = {}
    for battle in df.to_dict("records"):
        for poke in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            pokedex.setdefault(poke["name"], poke)
    return pokedex


def load_setup_moves(dfs, min_uses=30, min_rate=0.5):
    """Kullanıldığı turların en az %50'sinde kullananın boost toplamı artan hamleler.

    Elle liste yerine veriden: curse, swordsdance, bellydrum, agility, growth, amnesia, ...
    """
    uses, ups = {}, {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for side in ("p1", "p2"):
                prev = None
                for turn in battle["battle_timeline"]:
                    s = turn[f"{side}_pokemon_state"]
                    move = turn[f"{side}_move_details"]
                    boost = sum(s["boosts"].values())
                    if move and prev and prev[0] == s["name"]:
                        uses[move["name"]] = uses.get(move["name"], 0) + 1
                        if boost > prev[1]:
                            ups[move["name"]] = ups.get(move["name"], 0) + 1
                    prev = (s["name"], boost)
    return {m for m, n in uses.items() if n >= min_uses and ups.get(m, 0) / n >= min_rate}


def calculate_max_multiplier(atk_types, def_types, type_chart):
    max_mult = 0.0
    for at in atk_types:
        if at == "notype":
            continue
        current = 1.0
        for dt in def_types:
            if dt == "notype":
                continue
            current *= type_chart.get((at, dt), 1.0)
        max_mult = max(max_mult, current)
    return max_mult


def check_immunity(atk_types, def_types, type_chart):
    return any(type_chart.get((at, dt), 1.0) == 0.0
               for at in atk_types if at != "notype"
               for dt in def_types if dt != "notype")


def teammate_features(battle, pokedex, type_chart, healing_moves, setup_moves):
    f = {}
    timeline = battle["battle_timeline"]
    p1_team = battle["p1_team_details"]
    p1_names = [p["name"] for p in p1_team]

    # P2'nin görülen Pokémon'ları
    # sorted: set sırası her Python açılışında değişir; sabit sıra = her çalıştırmada aynı toplamlar
    p2_names = sorted({battle["p2_lead_details"]["name"]} | {t["p2_pokemon_state"]["name"] for t in timeline})
    p2_stats = [pokedex[n] for n in p2_names if n in pokedex]

    last_turn = timeline[-1]["turn"]
    turn_25 = max(1, last_turn - 5)

    f["f_p2_revealed_count"] = len(p2_names)
    f["f_p2_hidden_count"] = 6 - len(p2_names)
    f["f_p1_has_snorlax"] = int("snorlax" in p1_names)
    f["f_p2_has_snorlax"] = int("snorlax" in p2_names)

    p1_hp = {p: 1.0 for p in p1_names}
    p2_hp = {p: 1.0 for p in p2_names}
    p1_status = {p: "nostatus" for p in p1_names}
    p2_status = {p: "nostatus" for p in p2_names}
    p1_moves = {p: set() for p in p1_names}
    p2_moves = {p: set() for p in p2_names}
    p1_hist, p2_hist = {}, {}

    p1_sw = p2_sw = p1_sw5 = p2_sw5 = 0
    p1_setup = p2_setup = p1_rec = p2_rec = p1_failed = p2_failed = 0
    p1_spikes = p2_spikes = 0          # tarafın kendi sahasında diken var mı
    for side in ("p1", "p2"):
        for mv in SINGLE_MOVES:
            f[f"f_{side}_used_{mv}"] = 0

    prev1 = timeline[0]["p1_pokemon_state"]["name"]
    prev2 = timeline[0]["p2_pokemon_state"]["name"]
    for turn in timeline:
        t = turn["turn"]
        s1, s2 = turn["p1_pokemon_state"], turn["p2_pokemon_state"]
        if s1["name"] != prev1:
            p1_sw += 1
            p1_sw5 += t >= turn_25
        if s2["name"] != prev2:
            p2_sw += 1
            p2_sw5 += t >= turn_25
        prev1, prev2 = s1["name"], s2["name"]

        p1_hp[s1["name"]] = s1["hp_pct"]
        p2_hp[s2["name"]] = s2["hp_pct"]
        p1_status[s1["name"]] = s1["status"]
        p2_status[s2["name"]] = s2["status"]
        p1_hist[t] = sum(p1_hp.values()) / 6.0
        p2_hist[t] = (sum(p2_hp.values()) + f["f_p2_hidden_count"]) / 6.0

        for side, move, user, target_state, moves in (
                ("p1", turn["p1_move_details"], s1["name"], s2, p1_moves),
                ("p2", turn["p2_move_details"], s2["name"], s1, p2_moves)):
            if not move:
                continue
            name = move["name"]
            moves.setdefault(user, set()).add(name)
            if name in healing_moves:
                if side == "p1": p1_rec += 1
                else: p2_rec += 1
            if name in setup_moves:
                if side == "p1": p1_setup += 1
                else: p2_setup += 1
            if name in SINGLE_MOVES:
                f[f"f_{side}_used_{name}"] = 1
            if name == "spikes":                       # rakibin sahasına diken
                if side == "p1": p2_spikes = 1
                else: p1_spikes = 1
            if name == "rapidspin":                    # kendi sahasını temizler
                if side == "p1": p1_spikes = 0
                else: p2_spikes = 0
            if move["category"] != "status" and move["type"] != "threequestionmarks" \
                    and "protect" in target_state["effects"]:
                if side == "p1": p1_failed += 1
                else: p2_failed += 1

    f["f_p1_net_hp_pct"] = p1_hist[last_turn]
    f["f_hp_differential"] = p1_hist[last_turn] - p2_hist[last_turn]
    diff_25 = p1_hist.get(turn_25, p1_hist[1]) - p2_hist.get(turn_25, p2_hist[1])
    f["f_momentum_delta_hp_last_5"] = f["f_hp_differential"] - diff_25
    f["f_p1_switches_last_5"] = p1_sw5
    f["f_p2_switches_last_5"] = p2_sw5

    for st in ["slp", "tox", "par", "brn", "frz", "fnt"]:
        f[f"f_p1_status_{st}"] = sum(s == st for s in p1_status.values())
        f[f"f_p2_status_{st}"] = sum(s == st for s in p2_status.values())
    f["f_alive_diff"] = (6 - f["f_p1_status_fnt"]) - (6 - f["f_p2_status_fnt"])

    # Uyuyor ama Rest + Sleep Talk ile hâlâ saldırabiliyor
    f["f_p1_resttalk_slp"] = sum(s == "slp" and {"rest", "sleeptalk"} <= p1_moves.get(p, set())
                                 for p, s in p1_status.items())
    f["f_p2_resttalk_slp"] = sum(s == "slp" and {"rest", "sleeptalk"} <= p2_moves.get(p, set())
                                 for p, s in p2_status.items())

    last1, last2 = timeline[-1]["p1_pokemon_state"], timeline[-1]["p2_pokemon_state"]
    for stat in ["atk", "def", "spa", "spd", "spe"]:
        f[f"f_p1_active_boost_{stat}"] = last1["boosts"][stat]
        f[f"f_p2_active_boost_{stat}"] = last2["boosts"][stat]

    items = [p.get("item", "") for p in p1_team]
    f["f_p1_leftovers_count"] = items.count("leftovers")
    for it in ["thickclub", "lightball", "miracleberry"]:
        f[f"f_p1_has_{it}"] = int(it in items)

    f["f_p2_revealed_move_capacity_ratio"] = sum(len(m) for m in p2_moves.values()) / (len(p2_names) * 4)
    f["f_switch_diff"] = p1_sw - p2_sw
    f["f_p1_setup_count"], f["f_p2_setup_count"] = p1_setup, p2_setup
    f["f_p1_recovery_count"], f["f_p2_recovery_count"] = p1_rec, p2_rec
    f["f_p1_failed_moves"], f["f_p2_failed_moves"] = p1_failed, p2_failed
    f["f_p1_field_spikes"], f["f_p2_field_spikes"] = p1_spikes, p2_spikes

    # Takımlar arası stat ve tip eşleşmesi
    for k, key in [("speed", "base_spe"), ("def", "base_def"), ("spd", "base_spd")]:
        f[f"f_p1_{k}_std"] = np.std([p[key] for p in p1_team])
    if p2_stats:
        for k, key in [("speed", "base_spe"), ("def", "base_def"), ("spd", "base_spd")]:
            f[f"f_p2_{k}_std"] = np.std([p[key] for p in p2_stats])
        m1 = {k: np.mean([p[k] for p in p1_team]) for k in ["base_spe", "base_atk", "base_spa", "base_def", "base_spd"]}
        m2 = {k: np.mean([p[k] for p in p2_stats]) for k in ["base_spe", "base_atk", "base_spa", "base_def", "base_spd"]}
        f["f_speed_advantage"] = m1["base_spe"] - m2["base_spe"]
        f["f_p1_physical_pressure"] = m1["base_atk"] / (m2["base_def"] + 1)
        f["f_p1_special_pressure"] = m1["base_spa"] / (m2["base_spd"] + 1)
        f["f_p2_physical_pressure"] = m2["base_atk"] / (m1["base_def"] + 1)
        f["f_p2_special_pressure"] = m2["base_spa"] / (m1["base_spd"] + 1)
        mult1 = [calculate_max_multiplier(a["types"], b["types"], type_chart) for a in p1_team for b in p2_stats]
        mult2 = [calculate_max_multiplier(b["types"], a["types"], type_chart) for a in p1_team for b in p2_stats]
        f["f_p1_max_offensive_multiplier"] = np.mean(mult1)
        f["f_p2_max_offensive_multiplier"] = np.mean(mult2)
        f["f_p1_immunity_count"] = sum(check_immunity(a["types"], b["types"], type_chart) for a in p1_team for b in p2_stats)
        f["f_p2_immunity_count"] = sum(check_immunity(b["types"], a["types"], type_chart) for a in p1_team for b in p2_stats)
    else:
        for k in ["f_p2_speed_std", "f_p2_def_std", "f_p2_spd_std", "f_speed_advantage",
                  "f_p1_immunity_count", "f_p2_immunity_count"]:
            f[k] = 0.0
        for k in ["f_p1_physical_pressure", "f_p1_special_pressure", "f_p2_physical_pressure",
                  "f_p2_special_pressure", "f_p1_max_offensive_multiplier", "f_p2_max_offensive_multiplier"]:
            f[k] = 1.0

    t1 = next((p["types"] for p in p1_team if p["name"] == last1["name"]), ["notype"])
    t2 = next((p["types"] for p in p2_stats if p["name"] == last2["name"]), ["notype"])
    f["f_active_matchup_p1_to_p2"] = calculate_max_multiplier(t1, t2, type_chart)
    f["f_active_matchup_p2_to_p1"] = calculate_max_multiplier(t2, t1, type_chart)
    return f
