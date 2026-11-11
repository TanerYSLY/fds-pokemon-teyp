"""Her savaşı (bir satır) sabit uzunlukta bir sayı vektörüne çeviren fonksiyonlar — v3.

Önceki sürümlerden farklar için UPDATES.md'ye bakın (v2: src/features_v2.py, v1: src/features_v1.py).

Kullanım:
    from src.features import (load_species, load_type_chart, load_healing_moves, load_identity_vocab,
                              build_features, select_features)
    species = load_species([train_df])          # yardımcı tablolar sadece train'den
    chart = load_type_chart("data/type_chart_gen2.csv")
    healing = load_healing_moves([train_df])
    vocab = load_identity_vocab([train_df])
    X = build_features(train_df, species, chart, healing, vocab)   # battle_id indeksli DataFrame
    X_lin = select_features(X, "linear")                     # Logistic Regression için
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


def load_healing_moves(dfs, min_uses=30, min_heal_rate=0.25, big_gain=0.2):
    """Kullananın canını büyük ölçüde artıran hamleler (fikir #13).

    Elle liste yazmak yerine veriden çıkarılır: bir hamle kullanıldığı turların
    en az %25'inde kullananın canı ≥0.2 artıyorsa "iyileşme hamlesi" sayılır.
    Etiket kullanılmaz. Sonuç: recover, softboiled, milkdrink, rest, moonlight, ...
    """
    uses, heals = {}, {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for side in ("p1", "p2"):
                hp = {}
                for turn in battle["battle_timeline"]:
                    s = turn[f"{side}_pokemon_state"]
                    move = turn[f"{side}_move_details"]
                    prev = hp.get(s["name"], 1.0)
                    if move:
                        uses[move["name"]] = uses.get(move["name"], 0) + 1
                        if s["hp_pct"] - prev >= big_gain:
                            heals[move["name"]] = heals.get(move["name"], 0) + 1
                    hp[s["name"]] = s["hp_pct"]
    return {m for m, n in uses.items() if n >= min_uses and heals.get(m, 0) / n >= min_heal_rate}


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


def alive_offense(last, roster, species):
    """Ayakta kalanların saldırı gücü: Σ max(atk, spa) / 100 × kalan can (fikir #15).

    alive_strength savunmayı da topladığı için "sünger" Pokémon'ları (Blissey, Shuckle)
    güçlü gösteriyordu; burada sadece hasar verme kapasitesi sayılır.
    """
    total = 0.0
    for name in roster:
        state = last.get(name)
        if state is not None and state["status"] == "fnt":
            continue
        hp = state["hp_pct"] if state is not None else 1.0
        info = species.get(name)
        if info is not None:
            total += max(info["base_atk"], info["base_spa"]) / 100 * hp
    return total


def recent_damage(timeline, side, last_n=10):
    """Bu oyuncunun son `last_n` turda rakibe verdiği hasar (fikir #16)."""
    opp = "p2" if side == "p1" else "p1"
    hp = {}
    dmg = 0.0
    for turn in timeline:
        s = turn[f"{opp}_pokemon_state"]
        prev = hp.get(s["name"], 1.0)
        if turn["turn"] > len(timeline) - last_n:
            dmg += max(prev - s["hp_pct"], 0.0)
        hp[s["name"]] = s["hp_pct"]
    return dmg


def alive_healers(timeline, side, last, healing_moves):
    """Ayakta olan ve 30 turda en az bir kez iyileşme hamlesi kullanmış Pokémon sayısı (fikir #13)."""
    healers = {t[f"{side}_pokemon_state"]["name"] for t in timeline
               if t[f"{side}_move_details"] and t[f"{side}_move_details"]["name"] in healing_moves}
    return sum(last[n]["status"] != "fnt" for n in healers if n in last)


def side_features(battle, side, species, chart, healing_moves):
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

    # --- v3: hata analizinden gelen fikirler ---
    f["alive_healers"] = alive_healers(tl, side, last, healing_moves)                     # #13
    f["n_low_hp"] = sum(s["status"] != "fnt" and s["hp_pct"] < 0.3 for s in last.values())  # #14
    f["alive_offense"] = alive_offense(last, roster, species)                             # #15
    f["recent_damage"] = recent_damage(tl, side)                                          # #16

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


def battle_features(battle, species, chart, healing_moves):
    """Tek bir savaş (dict) için özellik sözlüğü."""
    f = {"battle_id": battle["battle_id"]}
    p1 = side_features(battle, "p1", species, chart, healing_moves)
    p2 = side_features(battle, "p2", species, chart, healing_moves)
    f.update({f"p1_{k}": v for k, v in p1.items()})
    f.update({f"p2_{k}": v for k, v in p2.items()})

    # Farklar: pozitifse P1 lehine
    for k in ["hp_total", "alive_strength", "active_hp", "active_boost_sum", "active_threat",
              "active_spe", "super_eff_hits", "seen", "max_boost_sum", "healed",
              "alive_healers", "alive_offense", "recent_damage"]:
        f[f"diff_{k}"] = p1[k] - p2[k]
    f["diff_fainted"] = p2["fainted"] - p1["fainted"]
    f["diff_damage"] = p2["damage_taken"] - p1["damage_taken"]       # P1'in verdiği − aldığı
    f["diff_disabled"] = p2["n_disabled"] - p1["n_disabled"]
    f["diff_low_hp"] = p2["n_low_hp"] - p1["n_low_hp"]               # rakibin neredeyse ölüsü fazlaysa P1 lehine

    # Son turdaki sahadaki eşleşme (YENİ): tip avantajı farkı
    a1 = species.get(battle["battle_timeline"][-1]["p1_pokemon_state"]["name"])
    a2 = species.get(battle["battle_timeline"][-1]["p2_pokemon_state"]["name"])
    t1 = a1["types"] if a1 else []
    t2 = a2["types"] if a2 else []
    f["active_type_adv"] = type_advantage(chart, t1, t2) - type_advantage(chart, t2, t1)
    return f


def alive_names(battle):
    """30. turda ayakta olan Pokémon isimleri: (P1, P2).

    P1: takımdaki 6'dan bayılmayanlar (hiç sahaya çıkmayanlar dahil).
    P2: sadece GÖRÜLEN ve bayılmayanlar (görülmeyenleri bilmiyoruz).
    """
    tl = battle["battle_timeline"]
    last1, last2 = last_known_state(tl, "p1"), last_known_state(tl, "p2")
    p1 = {p["name"] for p in battle["p1_team_details"]
          if not (p["name"] in last1 and last1[p["name"]]["status"] == "fnt")}
    p2 = {n for n, s in last2.items() if s["status"] != "fnt"}
    return p1, p2


def load_identity_vocab(dfs, min_count=300):
    """Kimlik sütunu açılacak Pokémon'lar (fikir #17): P1 takımlarında en az `min_count` kez görülenler.

    Sadece sıklığa bakılır, etiket kullanılmaz → leakage yok.
    """
    counts = {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for p in battle["p1_team_details"]:
                counts[p["name"]] = counts.get(p["name"], 0) + 1
    return sorted(n for n, c in counts.items() if c >= min_count)


def identity_features(battle, vocab):
    """Multi-hot: her sık Pokémon için 'P1'de ayakta mı?', 'P2'de görüldü ve ayakta mı?' (1/0)."""
    p1, p2 = alive_names(battle)
    f = {}
    for name in vocab:
        f[f"p1_alive_{name}"] = int(name in p1)
        f[f"p2_alive_{name}"] = int(name in p2)
    return f


def build_features(df, species, chart, healing_moves, vocab=None, teammate=None):
    """DataFrame (train veya test) → özellik tablosu (battle_id indeksli).

    vocab verilirse Pokémon kimliği sütunları eklenir (v4).
    teammate = {"pokedex": ..., "setup_moves": ...} verilirse takım arkadaşının
    özellikleri (`f_...`, src/features_teammate.py) eklenir (v5).
    """
    if teammate is not None:
        from src.features_teammate import teammate_features
    rows = []
    for b in df.to_dict("records"):
        f = battle_features(b, species, chart, healing_moves)
        if vocab is not None:
            f.update(identity_features(b, vocab))
        if teammate is not None:
            f.update(teammate_features(b, teammate["pokedex"], chart, healing_moves, teammate["setup_moves"]))
        rows.append(f)
    return pd.DataFrame(rows).set_index("battle_id")


# v5 (bizim + takım arkadaşının özellikleri) LogReg setinden atılan gruplar.
# Birleşik sette grup grup çıkarma (ablation) ile bulundu, bkz. UPDATES.md fikir #23.
V5_LINEAR_DROP = {
    # bizim "davranış" grubu: birleşik sette gürültü (çıkarınca 15 parçanın 14'ünde skor arttı)
    "ours_behavior": ["switches", "no_move_turns", "status_moves", "avg_base_power", "max_boost_sum"],
    # bizim hasar: aynı bilgi arkadaşın can geçmişinde var
    "ours_damage": ["damage_taken", "recent_damage"],
}
V5_TEAMMATE_LINEAR_DROP = [
    # bizimkiyle çakışan durum bilgisi, momentum ve eşyalar: çıkarınca skor değişmedi
    "f_momentum", "switches_last_5", "f_switch_diff", "leftovers", "thickclub", "lightball", "miracleberry",
    "f_hp_differential", "f_p1_net_hp", "f_p1_status_", "f_p2_status_", "f_alive_diff", "_active_boost_",
    "f_p2_revealed_count", "f_p2_hidden_count", "has_snorlax", "f_active_matchup",
]


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
    if any(c.startswith("f_") for c in X.columns):                # v5: gereksiz grupları at
        drop += [f"{side}_{k}" for side in ("p1", "p2") for ks in V5_LINEAR_DROP.values() for k in ks]
        drop += [c for c in X.columns if c.startswith("f_") and any(p in c for p in V5_TEAMMATE_LINEAR_DROP)]
    return X.drop(columns=[c for c in drop if c in X.columns])   # ablation'da bazıları zaten yok
