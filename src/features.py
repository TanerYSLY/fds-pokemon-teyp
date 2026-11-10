"""Turn each battle into one row of numbers.

we compute a fixed list of numbers (features) from the first 30 turns
Every battle gets the same columns so the result is a normal table one row per battle.

Lookup tables (species stats, healing moves, frequent Pokemon) are built from
the training data only, so nothing from the test set leaks into them.
"""

import pandas as pd

from src.features_teammate import teammate_features

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ["atk", "def", "spa", "spe"]  # we left out spd boosts, they show up in less than 1% of turns
DISABLED = ["slp", "frz"]              # asleep or frozen Pokemon can't move
POISONED = ["tox", "psn"]              # bad poison and normal poison, we count them together


# ---------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------

def load_species(dfs):
    """name -> types and base stats.

    For P2 we only know the lead's stats, but the same Pokemon shows up in
    other battles' P1 teams. So we collect every Pokemon we see once and can
    then look up the stats of any P2 Pokemon that comes out later.
    """
    species = {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
                if p["name"] not in species:  # stats are the same every time, so the first one is enough
                    info = {s: p[s] for s in STATS}
                    info["types"] = [t for t in p["types"] if t != "notype"]  # "notype" = no second type
                    info["total"] = sum(p[s] for s in STATS)  # base stat total, a rough "how strong" number
                    species[p["name"]] = info
    return species


def load_type_chart(path):
    """(attacking type, defending type) -> multiplier, e.g. ("fire", "grass") -> 2.0.
    Pairs that are not in the file are normal damage (1.0)."""
    tc = pd.read_csv(path)
    return {(a, d): m for a, d, m in zip(tc.attacking_type, tc.defending_type, tc.multiplier)}


def load_healing_moves(dfs, min_uses=30, min_heal_rate=0.25, big_gain=0.2):
    """Find the moves that heal the user.

    We didn't want to type a list of healing moves by hand (the rules say only
    competition data), so we let the data tell us: if the user's HP goes up by
    0.2 or more after a move in at least 25% of its uses, we call it a healing move.
    min_uses=30 skips rare moves where a few lucky turns could fool us.
    This gives recover, softboiled, milkdrink, rest, moonlight, ...
    """
    uses, heals = {}, {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for side in ("p1", "p2"):
                hp = {}  # last HP we saw for each Pokemon of this side
                for turn in battle["battle_timeline"]:
                    s = turn[f"{side}_pokemon_state"]
                    move = turn[f"{side}_move_details"]
                    prev = hp.get(s["name"], 1.0)  # first time we see it -> assume full HP
                    if move:
                        uses[move["name"]] = uses.get(move["name"], 0) + 1
                        if s["hp_pct"] - prev >= big_gain:
                            heals[move["name"]] = heals.get(move["name"], 0) + 1
                    hp[s["name"]] = s["hp_pct"]
    return {m for m, n in uses.items() if n >= min_uses and heals.get(m, 0) / n >= min_heal_rate}


def load_identity_vocab(dfs, min_count=300):
    """Pokemon that appear in at least 300 P1 teams (30 of them).

    Some Pokemon matter more than their stats say (Snorlax, Zapdos, ...), so
    these get their own 1/0 columns. Rare ones would only add noise.
    We only count how often they appear, we never look at who won.
    """
    counts = {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for p in battle["p1_team_details"]:
                counts[p["name"]] = counts.get(p["name"], 0) + 1
    return sorted(n for n, c in counts.items() if c >= min_count)


def effectiveness(chart, atk_type, def_types):
    """Multiplier of one attack type against a Pokemon with one or two types.
    Two types multiply: ice vs dragon/flying = 2 * 2 = 4."""
    m = 1.0
    for d in def_types:
        m *= chart.get((atk_type, d), 1.0)
    return m


def type_advantage(chart, atk_types, def_types):
    """Best multiplier the attacker gets using its own types.

    We don't know the attacker's moves, but Pokemon usually carry moves of
    their own type, so we use the types as a guess.
    """
    if not atk_types or not def_types:  # unknown Pokemon -> call it neutral
        return 1.0
    return max(effectiveness(chart, t, def_types) for t in atk_types)


# ---------------------------------------------------------------
# Helpers for one side of a battle
# ---------------------------------------------------------------

def last_known_state(timeline, side):
    """name -> state of each Pokemon the last time it was on the field.

    The timeline only shows the active Pokemon. A Pokemon that switched out on
    turn 10 still has the HP it had on turn 10, so we keep the last state we saw.
    """
    last = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state  # later turns overwrite earlier ones
    return last


def hp_flow(timeline, side):
    """Total damage taken and total HP healed over the 30 turns.

    We compare each Pokemon's HP with the last time we saw it. Going down is
    damage, going up is healing (rest, leftovers, ...).
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
    """How many of this side's attacks were super effective and how many were resisted.
    A player who keeps picking good attacks is probably playing better."""
    opp = "p2" if side == "p1" else "p1"
    super_eff = resisted = 0
    for turn in timeline:
        move = turn[f"{side}_move_details"]
        if not move or move["base_power"] == 0:  # no move, or a status move that does no damage
            continue
        target = species.get(turn[f"{opp}_pokemon_state"]["name"])
        if target is None:
            continue
        m = effectiveness(chart, move["type"], target["types"])
        super_eff += m >= 2
        resisted += m < 1
    return super_eff, resisted


def alive_strength(last, roster, species):
    """How much strength this side has left: sum of (base stat total / 600) * hp
    over the Pokemon that haven't fainted. 600 is about a strong Pokemon's total,
    so a strong Pokemon at full HP counts as about 1."""
    total = 0.0
    for name in roster:
        state = last.get(name)
        if state is not None and state["status"] == "fnt":  # fainted, adds nothing
            continue
        hp = state["hp_pct"] if state is not None else 1.0  # never came out -> still full HP
        info = species.get(name)
        if info is not None:
            total += info["total"] / 600 * hp
    return total


def alive_offense(last, roster, species):
    """Like alive_strength but only counts attacking power: max(atk, spa) / 100 * hp.

    When we looked at the battles our model got wrong, walls like Blissey and
    Shuckle made a team look strong in alive_strength even though they can't
    deal much damage. This one only counts how hard the team can hit.
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
    """Damage this side dealt to the opponent in the last 10 turns.
    Shows who has the momentum at the moment the battle is cut off."""
    opp = "p2" if side == "p1" else "p1"
    hp = {}
    dmg = 0.0
    for turn in timeline:
        s = turn[f"{opp}_pokemon_state"]
        prev = hp.get(s["name"], 1.0)
        if turn["turn"] > len(timeline) - last_n:  # only count the last 10 turns
            dmg += max(prev - s["hp_pct"], 0.0)
        hp[s["name"]] = s["hp_pct"]
    return dmg


def alive_healers(timeline, side, last, healing_moves):
    """How many Pokemon that are still alive used a healing move at least once.
    A Pokemon that can heal itself will probably last longer after turn 30."""
    healers = {t[f"{side}_pokemon_state"]["name"] for t in timeline
               if t[f"{side}_move_details"] and t[f"{side}_move_details"]["name"] in healing_moves}
    return sum(last[n]["status"] != "fnt" for n in healers if n in last)


def side_features(battle, side, species, chart, healing_moves):
    """All features for one player (p1 or p2). The names get a p1_/p2_ prefix later."""
    tl = battle["battle_timeline"]
    f = {}
    last = last_known_state(tl, side)

    # --- team state at turn 30 ---
    f["seen"] = len(last)  # how many different Pokemon came out
    f["fainted"] = sum(s["status"] == "fnt" for s in last.values())
    f["hp_total"] = sum(s["hp_pct"] for s in last.values()) + (6 - len(last))  # unseen ones count as full HP
    f["hp_seen_mean"] = sum(s["hp_pct"] for s in last.values()) / len(last)
    f["n_slp"] = sum(s["status"] == "slp" for s in last.values())
    f["n_par"] = sum(s["status"] == "par" for s in last.values())
    f["n_poisoned"] = sum(s["status"] in POISONED for s in last.values())
    f["n_disabled"] = sum(s["status"] in DISABLED for s in last.values())
    f["n_brn"] = sum(s["status"] == "brn" for s in last.values())

    # --- strength left ---
    # for P1 we know all 6 team members, for P2 only the ones that came out
    if side == "p1":
        roster = [p["name"] for p in battle["p1_team_details"]]
    else:
        roster = list(last)
    f["alive_strength"] = alive_strength(last, roster, species)
    f["alive_healers"] = alive_healers(tl, side, last, healing_moves)
    f["n_low_hp"] = sum(s["status"] != "fnt" and s["hp_pct"] < 0.3 for s in last.values())  # almost fainted
    f["alive_offense"] = alive_offense(last, roster, species)
    f["recent_damage"] = recent_damage(tl, side)

    # --- the Pokemon on the field at the last turn ---
    active = tl[-1][f"{side}_pokemon_state"]
    f["active_hp"] = active["hp_pct"]
    f["active_boost_sum"] = sum(active["boosts"][b] for b in BOOSTS)
    for b in BOOSTS:
        f[f"active_boost_{b}"] = active["boosts"][b]
    # a boosted Pokemon is only scary if it also has HP left, so we multiply the two
    f["active_threat"] = max(f["active_boost_sum"], 0) * active["hp_pct"]
    info = species.get(active["name"])
    f["active_spe"] = info["base_spe"] if info else 0
    f["active_total"] = info["total"] if info else 0

    # --- what the player did during the 30 turns ---
    names = [t[f"{side}_pokemon_state"]["name"] for t in tl]
    f["switches"] = sum(a != b for a, b in zip(names, names[1:]))  # active Pokemon changed from one turn to the next
    f["no_move_turns"] = sum(t[f"{side}_move_details"] is None for t in tl)  # switched, asleep, paralyzed, ...
    moves = [t[f"{side}_move_details"] for t in tl if t[f"{side}_move_details"]]
    f["status_moves"] = sum(m["category"] == "status" for m in moves)
    f["avg_base_power"] = sum(m["base_power"] for m in moves) / len(moves) if moves else 0.0
    f["max_boost_sum"] = max(sum(t[f"{side}_pokemon_state"]["boosts"][b] for b in BOOSTS) for t in tl)

    f["damage_taken"], f["healed"] = hp_flow(tl, side)
    f["super_eff_hits"], f["resisted_hits"] = hit_quality(tl, side, species, chart)
    return f


def battle_features(battle, species, chart, healing_moves):
    """Features of both players plus some P1 - P2 differences."""
    f = {"battle_id": battle["battle_id"]}
    p1 = side_features(battle, "p1", species, chart, healing_moves)
    p2 = side_features(battle, "p2", species, chart, healing_moves)
    f.update({f"p1_{k}": v for k, v in p1.items()})
    f.update({f"p2_{k}": v for k, v in p2.items()})

    # Differences, positive = good for P1. Trees can't easily subtract two
    # columns themselves, so we give them the difference directly.
    for k in ["hp_total", "alive_strength", "active_hp", "active_boost_sum", "active_threat",
              "active_spe", "super_eff_hits", "seen", "max_boost_sum", "healed",
              "alive_healers", "alive_offense", "recent_damage"]:
        f[f"diff_{k}"] = p1[k] - p2[k]
    # for these "bad" things we flip it (p2 - p1) so positive still means good for P1
    f["diff_fainted"] = p2["fainted"] - p1["fainted"]
    f["diff_damage"] = p2["damage_taken"] - p1["damage_taken"]
    f["diff_disabled"] = p2["n_disabled"] - p1["n_disabled"]
    f["diff_low_hp"] = p2["n_low_hp"] - p1["n_low_hp"]

    # type matchup of the two Pokemon on the field at the last turn
    a1 = species.get(battle["battle_timeline"][-1]["p1_pokemon_state"]["name"])
    a2 = species.get(battle["battle_timeline"][-1]["p2_pokemon_state"]["name"])
    t1 = a1["types"] if a1 else []
    t2 = a2["types"] if a2 else []
    f["active_type_adv"] = type_advantage(chart, t1, t2) - type_advantage(chart, t2, t1)
    return f


def identity_features(battle, vocab):
    """1/0 columns: is this Pokemon still alive on P1's side / on P2's side?

    For P1 we know the whole team, so a Pokemon that never came out counts as alive.
    For P2 we only know the ones that came out, so unseen ones are 0.
    """
    tl = battle["battle_timeline"]
    last1, last2 = last_known_state(tl, "p1"), last_known_state(tl, "p2")
    p1 = {p["name"] for p in battle["p1_team_details"]
          if not (p["name"] in last1 and last1[p["name"]]["status"] == "fnt")}
    p2 = {n for n, s in last2.items() if s["status"] != "fnt"}
    f = {}
    for name in vocab:
        f[f"p1_alive_{name}"] = int(name in p1)
        f[f"p2_alive_{name}"] = int(name in p2)
    return f


def build_features(df, species, chart, healing_moves, vocab, pokedex, setup_moves):
    """Whole DataFrame (train or test) -> feature table with battle_id as index."""
    rows = []
    for b in df.to_dict("records"):
        f = battle_features(b, species, chart, healing_moves)  # our features
        f.update(identity_features(b, vocab))                  # the 1/0 Pokemon columns
        f.update(teammate_features(b, pokedex, chart, healing_moves, setup_moves))  # Enes's f_ features
        rows.append(f)
    return pd.DataFrame(rows).set_index("battle_id")


# ---------------------------------------------------------------
# Which columns each model gets
# ---------------------------------------------------------------

# Columns we remove for logistic regression. We removed feature groups one at
# a time and checked CV: without these the score went up or stayed the same.
LINEAR_DROP = ["switches", "no_move_turns", "status_moves", "avg_base_power", "max_boost_sum",
               "damage_taken", "recent_damage",  # Enes's HP history already has this
               "active_boost_sum",  # just the sum of the single boosts
               "healed"]            # hp_total + damage_taken - healed = 6, so one of them is redundant
# Enes's columns that overlap with ours (any column whose name contains one of these)
TEAMMATE_LINEAR_DROP = [
    "f_momentum", "switches_last_5", "f_switch_diff", "leftovers", "thickclub", "lightball", "miracleberry",
    "f_hp_differential", "f_p1_net_hp", "f_p1_status_", "f_p2_status_", "f_alive_diff", "_active_boost_",
    "f_p2_revealed_count", "f_p2_hidden_count", "has_snorlax", "f_active_matchup",
]


def select_features(X, kind):
    """Trees get all 217 columns. Logistic regression gets 143.

    For logistic regression, columns that are copies or sums of other columns
    make the weights unstable and hard to read. diff_ columns are just p1 - p2,
    so the model can already learn them from the p1_ and p2_ columns.
    """
    if kind == "tree":
        return X
    drop = [c for c in X.columns if c.startswith("diff_")]
    drop += [f"{side}_{k}" for side in ("p1", "p2") for k in LINEAR_DROP]
    drop += [c for c in X.columns if c.startswith("f_") and any(p in c for p in TEAMMATE_LINEAR_DROP)]
    return X.drop(columns=drop)
