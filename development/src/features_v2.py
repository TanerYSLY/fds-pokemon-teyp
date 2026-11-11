"""Her savaşı (bir satır) sabit uzunlukta bir sayı vektörüne çeviren fonksiyonlar — v2 (DONDURULDU).

v1'den farklar için UPDATES.md'ye bakın.

Kullanım:
    from src.features import load_species, load_type_chart, build_features, select_features
    species = load_species([train_df, test_df])
    chart = load_type_chart("data/type_chart_gen2.csv")
    X = build_features(train_df, species, chart)       # battle_id indeksli DataFrame
    X_lin = select_features(X, "linear")                # Logistic Regression için
"""

import pandas as pd

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ["atk", "def", "spa", "spe"]          # spd çok seyrek (%0.7) → atıldı
DISABLED = ["slp", "frz"]                      # hamle yapamayan Pokémon
POISONED = ["tox", "psn"]


# ---------------------------------------------------------------------------
# Yardımcı tablolar (sadece competition verisinden)
# ---------------------------------------------------------------------------

def load_species(dfs):
    """Tür → {types, stat'lar, total} sözlüğü.

    P2'nin görülen Pokémon'larının stat/tiplerini bilmek için P1 takımlarından ve
    P2 lead'lerinden derlenir. Etiket kullanılmaz → leakage yok.
    """
    species = {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
                if p["name"] not in species:
                    info = {s: p[s] for s in STATS}
                    info["types"] = [t for t in p["types"] if t != "notype"]
                    info["total"] = sum(p[s] for s in STATS)
                    species[p["name"]] = info
    return species


def load_type_chart(path):
    """(saldıran_tip, savunan_tip) → çarpan. Tabloda olmayan çiftler 1.0."""
    tc = pd.read_csv(path)
    return {(a, d): m for a, d, m in zip(tc.attacking_type, tc.defending_type, tc.multiplier)}


def effectiveness(chart, atk_type, def_types):
    m = 1.0
    for d in def_types:
        m *= chart.get((atk_type, d), 1.0)
    return m


def type_advantage(chart, atk_types, def_types):
    """Saldıranın kendi tiplerinden (STAB vekili) savunana en iyi çarpan."""
    if not atk_types or not def_types:
        return 1.0
    return max(effectiveness(chart, t, def_types) for t in atk_types)


# ---------------------------------------------------------------------------
# Savaş içi özellikler
# ---------------------------------------------------------------------------

def last_known_state(timeline, side):
    """Her Pokémon'un timeline'da SON görüldüğü andaki durumu: {isim: state}."""
    last = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state
    return last


def hp_flow(timeline, side):
    """Bir oyuncunun Pokémon'larının aldığı toplam hasar ve iyileşme.

    Her Pokémon'un son bilinen canını takip eder (ilk görüldüğünde 1.0 varsayılır).
    v1'deki hata düzeltildi: switch ile gelen Pokémon'un aynı turda aldığı hasar
    ve 1. turdaki hasar artık sayılıyor; iyileşme (Rest, Leftovers) ayrı tutuluyor.
    """
    hp = {}
    taken = healed = 0.0
    for turn in timeline:
        s = turn[f"{side}_pokemon_state"]
        prev = hp.get(s["name"], 1.0)
        taken += max(prev - s["hp_pct"], 0.0)
        healed += max(s["hp_pct"] - prev, 0.0)
        hp[s["name"]] = s["hp_pct"]
    return taken, healed


def hit_quality(timeline, side, species, chart):
    """Hasar veren hamlelerin rakibin (o turdaki) tiplerine karşı etkinliği."""
    opp = "p2" if side == "p1" else "p1"
    super_eff = resisted = 0
    for turn in timeline:
        move = turn[f"{side}_move_details"]
        if not move or move["base_power"] == 0:
            continue
        target = species.get(turn[f"{opp}_pokemon_state"]["name"])
        if target is None:
            continue
        m = effectiveness(chart, move["type"], target["types"])
        super_eff += m >= 2
        resisted += m < 1
    return super_eff, resisted


def alive_strength(last, roster, species):
    """Ayakta kalan Pokémon'ların gücü: Σ (base stat toplamı / 600) × kalan can.

    roster: bilinen Pokémon isimleri (P1 için 6'sı, P2 için görülenler).
    Hiç sahaya çıkmayanlar tam can sayılır.
    """
    total = 0.0
    for name in roster:
        state = last.get(name)
        if state is not None and state["status"] == "fnt":
            continue
        hp = state["hp_pct"] if state is not None else 1.0
        info = species.get(name)
        if info is not None:
            total += info["total"] / 600 * hp
    return total


def side_features(battle, side, species, chart):
    """Bir oyuncunun (p1 veya p2) 30 turdaki gidişatını özetler."""
    tl = battle["battle_timeline"]
    f = {}
    last = last_known_state(tl, side)

    # --- 30. turdaki takım durumu ---
    f["seen"] = len(last)
    f["fainted"] = sum(s["status"] == "fnt" for s in last.values())
    f["hp_total"] = sum(s["hp_pct"] for s in last.values()) + (6 - len(last))
    f["hp_seen_mean"] = sum(s["hp_pct"] for s in last.values()) / len(last)
    f["n_slp"] = sum(s["status"] == "slp" for s in last.values())
    f["n_par"] = sum(s["status"] == "par" for s in last.values())
    f["n_poisoned"] = sum(s["status"] in POISONED for s in last.values())
    f["n_disabled"] = sum(s["status"] in DISABLED for s in last.values())
    f["n_brn"] = sum(s["status"] == "brn" for s in last.values())

    # --- ayakta kalan gücü (YENİ) ---
    if side == "p1":
        roster = [p["name"] for p in battle["p1_team_details"]]
    else:
        roster = list(last)
    f["alive_strength"] = alive_strength(last, roster, species)

    # --- son turda sahadaki Pokémon ---
    active = tl[-1][f"{side}_pokemon_state"]
    f["active_hp"] = active["hp_pct"]
    f["active_boost_sum"] = sum(active["boosts"][b] for b in BOOSTS)
    for b in BOOSTS:
        f[f"active_boost_{b}"] = active["boosts"][b]
    f["active_threat"] = max(f["active_boost_sum"], 0) * active["hp_pct"]   # güçlenmiş VE canlı
    info = species.get(active["name"])
    f["active_spe"] = info["base_spe"] if info else 0
    f["active_total"] = info["total"] if info else 0

    # --- 30 tur boyunca davranış ---
    names = [t[f"{side}_pokemon_state"]["name"] for t in tl]
    f["switches"] = sum(a != b for a, b in zip(names, names[1:]))
    f["no_move_turns"] = sum(t[f"{side}_move_details"] is None for t in tl)
    moves = [t[f"{side}_move_details"] for t in tl if t[f"{side}_move_details"]]
    f["status_moves"] = sum(m["category"] == "status" for m in moves)
    f["avg_base_power"] = sum(m["base_power"] for m in moves) / len(moves) if moves else 0.0
    f["max_boost_sum"] = max(sum(t[f"{side}_pokemon_state"]["boosts"][b] for b in BOOSTS) for t in tl)

    # --- hasar / iyileşme (DÜZELTİLDİ) ---
    f["damage_taken"], f["healed"] = hp_flow(tl, side)

    # --- tip kalitesi (YENİ) ---
    f["super_eff_hits"], f["resisted_hits"] = hit_quality(tl, side, species, chart)
    return f


def battle_features(battle, species, chart):
    """Tek bir savaş (dict) için özellik sözlüğü."""
    f = {"battle_id": battle["battle_id"]}
    p1 = side_features(battle, "p1", species, chart)
    p2 = side_features(battle, "p2", species, chart)
    f.update({f"p1_{k}": v for k, v in p1.items()})
    f.update({f"p2_{k}": v for k, v in p2.items()})

    # Farklar: pozitifse P1 lehine
    for k in ["hp_total", "alive_strength", "active_hp", "active_boost_sum", "active_threat",
              "active_spe", "super_eff_hits", "seen", "max_boost_sum", "healed"]:
        f[f"diff_{k}"] = p1[k] - p2[k]
    f["diff_fainted"] = p2["fainted"] - p1["fainted"]
    f["diff_damage"] = p2["damage_taken"] - p1["damage_taken"]       # P1'in verdiği − aldığı
    f["diff_disabled"] = p2["n_disabled"] - p1["n_disabled"]

    # Son turdaki sahadaki eşleşme (YENİ): tip avantajı farkı
    a1 = species.get(battle["battle_timeline"][-1]["p1_pokemon_state"]["name"])
    a2 = species.get(battle["battle_timeline"][-1]["p2_pokemon_state"]["name"])
    t1 = a1["types"] if a1 else []
    t2 = a2["types"] if a2 else []
    f["active_type_adv"] = type_advantage(chart, t1, t2) - type_advantage(chart, t2, t1)
    return f


def build_features(df, species, chart):
    """DataFrame (train veya test) → özellik tablosu (battle_id indeksli)."""
    rows = [battle_features(b, species, chart) for b in df.to_dict("records")]
    return pd.DataFrame(rows).set_index("battle_id")


def select_features(X, kind):
    """Model ailesine göre sütun seti.

    linear: diff_* sütunları p1_* − p2_* ile birebir aynı bilgi → atılır
            (aksi halde katsayılar yorumlanamaz). active_boost_sum da alt boost'ların toplamı.
    tree:   hepsi tutulur — ağaçlar iki sütunu kendi başına çıkaramaz, farklar işe yarar.
    """
    if kind == "tree":
        return X
    drop = [c for c in X.columns if c.startswith("diff_")]
    drop += ["p1_active_boost_sum", "p2_active_boost_sum"]
    drop += ["p1_healed", "p2_healed"]   # hp_total + damage_taken − healed = 6 (birebir bağımlı)
    return X.drop(columns=[c for c in drop if c in X.columns])   # ablation'da bazıları zaten yok
