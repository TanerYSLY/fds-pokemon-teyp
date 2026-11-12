"""
Feature engineering for Submission 1 (FDS Pokémon Battle Prediction, Gen 2 OU).

Turns each battle (the first 30 turns) into one row of numbers. The table has
217 columns in two groups:
- the main features (`p1_*`, `p2_*`, `diff_*` and the `p1_alive_*` / `p2_alive_*` columns),
- the `f_*` features (`f_features`), which are the same kind of features as in
  Submission 2.

All lookup tables (species stats, healing moves, setup moves, frequent
Pokémon) are built from the training data only. `select_features` picks the
columns used by each model.
"""

import numpy as np
import pandas as pd

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BOOSTS = ["atk", "def", "spa", "spe"]
DISABLED = ["slp", "frz"]          # asleep or frozen Pokémon can't move
POISONED = ["tox", "psn"]          # bad poison and normal poison are counted together
SINGLE_MOVES = ["roar", "whirlwind", "healbell", "aromatherapy", "rapidspin"]  # one "was it used?" column each


# Lookup tables

def load_species(dfs):
    """Build {name: types and base stats} from the P1 teams and P2 leads."""
    species = {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
                if p["name"] not in species:
                    info = {s: p[s] for s in STATS}
                    info["types"] = [t for t in p["types"] if t != "notype"]
                    info["total"] = sum(p[s] for s in STATS)  # base stat total
                    species[p["name"]] = info
    return species


def load_type_chart(path):
    """Load the type chart as {(attacking type, defending type): multiplier}.
    Pairs that are not in the file are normal damage (1.0)."""
    tc = pd.read_csv(path)
    return {(a, d): m for a, d, m in zip(tc.attacking_type, tc.defending_type, tc.multiplier)}


def build_pokedex(df):
    """Build {name: full Pokémon info (stats, types)}; the first occurrence of each name is kept."""
    pokedex = {}
    for battle in df.to_dict("records"):
        for poke in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            pokedex.setdefault(poke["name"], poke)
    return pokedex


def load_healing_moves(dfs, min_uses=30, min_heal_rate=0.25, big_gain=0.2):
    """Find the moves that heal the user: moves after which the user's HP goes
    up by at least `big_gain` in at least `min_heal_rate` of their uses.
    Moves used fewer than `min_uses` times are ignored."""
    uses, heals = {}, {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for side in ("p1", "p2"):
                hp = {}  # last HP seen for each Pokémon of this side
                for turn in battle["battle_timeline"]:
                    s = turn[f"{side}_pokemon_state"]
                    move = turn[f"{side}_move_details"]
                    prev = hp.get(s["name"], 1.0)  # first time seen -> full HP
                    if move:
                        uses[move["name"]] = uses.get(move["name"], 0) + 1
                        if s["hp_pct"] - prev >= big_gain:
                            heals[move["name"]] = heals.get(move["name"], 0) + 1
                    hp[s["name"]] = s["hp_pct"]
    return {m for m, n in uses.items() if n >= min_uses and heals.get(m, 0) / n >= min_heal_rate}


def load_setup_moves(dfs, min_uses=30, min_rate=0.5):
    """Find the setup moves: moves after which the user's boosts go up in at
    least `min_rate` of their uses. Moves used fewer than `min_uses` times are ignored."""
    uses, ups = {}, {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for side in ("p1", "p2"):
                prev = None  # (name, boost sum) of the previous turn
                for turn in battle["battle_timeline"]:
                    s = turn[f"{side}_pokemon_state"]
                    move = turn[f"{side}_move_details"]
                    boost = sum(s["boosts"].values())
                    # compare only if the same Pokémon was active on the previous turn
                    if move and prev and prev[0] == s["name"]:
                        uses[move["name"]] = uses.get(move["name"], 0) + 1
                        if boost > prev[1]:
                            ups[move["name"]] = ups.get(move["name"], 0) + 1
                    prev = (s["name"], boost)
    return {m for m, n in uses.items() if n >= min_uses and ups.get(m, 0) / n >= min_rate}


def load_identity_vocab(dfs, min_count=300):
    """List the Pokémon that appear in at least `min_count` P1 teams (30 of them).
    Each one gets its own alive / fainted column."""
    counts = {}
    for df in dfs:
        for battle in df.to_dict("records"):
            for p in battle["p1_team_details"]:
                counts[p["name"]] = counts.get(p["name"], 0) + 1
    return sorted(n for n, c in counts.items() if c >= min_count)


# Type-effectiveness helpers

def effectiveness(chart, atk_type, def_types):
    """Multiplier of one attack type against a Pokémon with one or two types
    (the multipliers of the two defending types are multiplied)."""
    m = 1.0
    for d in def_types:
        m *= chart.get((atk_type, d), 1.0)
    return m


def type_advantage(chart, atk_types, def_types):
    """Best multiplier the attacker gets using its own types as attack types.
    Returns 1.0 if either side has no known types."""
    if not atk_types or not def_types:
        return 1.0
    return max(effectiveness(chart, t, def_types) for t in atk_types)


def calculate_max_multiplier(atk_types, def_types, type_chart):
    """Best type multiplier the attacker's types get against the defender
    (0.0 if there is no attacking type). The placeholder type "notype" is ignored."""
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
    """True if any of the attacker's types does 0 damage to any of the defender's types."""
    return any(type_chart.get((at, dt), 1.0) == 0.0
               for at in atk_types if at != "notype"
               for dt in def_types if dt != "notype")


# Helpers for one side of a battle

def last_known_state(timeline, side):
    """Return {name: state} with the last state seen for each Pokémon of this side."""
    last = {}
    for turn in timeline:
        state = turn[f"{side}_pokemon_state"]
        last[state["name"]] = state
    return last


def hp_flow(timeline, side):
    """Total HP lost and total HP healed over the timeline, comparing each
    Pokémon's HP with the last time it was seen."""
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
    """Count the side's damaging attacks that were super effective and that were resisted."""
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
    """Sum of (base stat total / 600) * HP over the Pokémon that have not fainted.
    Pokémon that never came out count as full HP."""
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
    """Sum of max(base attack, base special attack) / 100 * HP over the Pokémon
    that have not fainted. Same as alive_strength but counts only attacking power."""
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
    """Damage this side dealt to the opponent in the last `last_n` turns."""
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
    """Number of Pokémon still alive that used a healing move at least once."""
    healers = {t[f"{side}_pokemon_state"]["name"] for t in timeline
               if t[f"{side}_move_details"] and t[f"{side}_move_details"]["name"] in healing_moves}
    return sum(last[n]["status"] != "fnt" for n in healers if n in last)


def side_features(battle, side, species, chart, healing_moves):
    """All main features for one player (p1 or p2); the p1_ / p2_ prefix is added later."""
    tl = battle["battle_timeline"]
    f = {}
    last = last_known_state(tl, side)

    # team state at the last turn
    f["seen"] = len(last)
    f["fainted"] = sum(s["status"] == "fnt" for s in last.values())
    f["hp_total"] = sum(s["hp_pct"] for s in last.values()) + (6 - len(last))  # unseen Pokémon count as full HP
    f["hp_seen_mean"] = sum(s["hp_pct"] for s in last.values()) / len(last)
    f["n_slp"] = sum(s["status"] == "slp" for s in last.values())
    f["n_par"] = sum(s["status"] == "par" for s in last.values())
    f["n_poisoned"] = sum(s["status"] in POISONED for s in last.values())
    f["n_disabled"] = sum(s["status"] in DISABLED for s in last.values())
    f["n_brn"] = sum(s["status"] == "brn" for s in last.values())

    # strength left (P1: the whole team is known; P2: only the Pokémon that came out)
    if side == "p1":
        roster = [p["name"] for p in battle["p1_team_details"]]
    else:
        roster = list(last)
    f["alive_strength"] = alive_strength(last, roster, species)
    f["alive_healers"] = alive_healers(tl, side, last, healing_moves)
    f["n_low_hp"] = sum(s["status"] != "fnt" and s["hp_pct"] < 0.3 for s in last.values())
    f["alive_offense"] = alive_offense(last, roster, species)
    f["recent_damage"] = recent_damage(tl, side)

    # the Pokémon on the field at the last turn
    active = tl[-1][f"{side}_pokemon_state"]
    f["active_hp"] = active["hp_pct"]
    f["active_boost_sum"] = sum(active["boosts"][b] for b in BOOSTS)
    for b in BOOSTS:
        f[f"active_boost_{b}"] = active["boosts"][b]
    f["active_threat"] = max(f["active_boost_sum"], 0) * active["hp_pct"]  # positive boosts times HP left
    info = species.get(active["name"])
    f["active_spe"] = info["base_spe"] if info else 0
    f["active_total"] = info["total"] if info else 0

    # what the player did during the timeline
    names = [t[f"{side}_pokemon_state"]["name"] for t in tl]
    f["switches"] = sum(a != b for a, b in zip(names, names[1:]))
    f["no_move_turns"] = sum(t[f"{side}_move_details"] is None for t in tl)
    moves = [t[f"{side}_move_details"] for t in tl if t[f"{side}_move_details"]]
    f["status_moves"] = sum(m["category"] == "status" for m in moves)
    f["avg_base_power"] = sum(m["base_power"] for m in moves) / len(moves) if moves else 0.0
    f["max_boost_sum"] = max(sum(t[f"{side}_pokemon_state"]["boosts"][b] for b in BOOSTS) for t in tl)

    f["damage_taken"], f["healed"] = hp_flow(tl, side)
    f["super_eff_hits"], f["resisted_hits"] = hit_quality(tl, side, species, chart)
    return f


def battle_features(battle, species, chart, healing_moves):
    """Main features of both players plus the P1 - P2 differences (positive = good for P1)."""
    f = {"battle_id": battle["battle_id"]}
    p1 = side_features(battle, "p1", species, chart, healing_moves)
    p2 = side_features(battle, "p2", species, chart, healing_moves)
    f.update({f"p1_{k}": v for k, v in p1.items()})
    f.update({f"p2_{k}": v for k, v in p2.items()})

    for k in ["hp_total", "alive_strength", "active_hp", "active_boost_sum", "active_threat",
              "active_spe", "super_eff_hits", "seen", "max_boost_sum", "healed",
              "alive_healers", "alive_offense", "recent_damage"]:
        f[f"diff_{k}"] = p1[k] - p2[k]
    # for these the difference is flipped (P2 - P1) so positive is still good for P1
    f["diff_fainted"] = p2["fainted"] - p1["fainted"]
    f["diff_damage"] = p2["damage_taken"] - p1["damage_taken"]
    f["diff_disabled"] = p2["n_disabled"] - p1["n_disabled"]
    f["diff_low_hp"] = p2["n_low_hp"] - p1["n_low_hp"]

    # type matchup of the two active Pokémon at the last turn
    a1 = species.get(battle["battle_timeline"][-1]["p1_pokemon_state"]["name"])
    a2 = species.get(battle["battle_timeline"][-1]["p2_pokemon_state"]["name"])
    t1 = a1["types"] if a1 else []
    t2 = a2["types"] if a2 else []
    f["active_type_adv"] = type_advantage(chart, t1, t2) - type_advantage(chart, t2, t1)
    return f


def identity_features(battle, vocab):
    """1/0 columns telling whether each frequent Pokémon is still alive on P1's side
    and on P2's side. For P1 a Pokémon that never came out counts as alive; for P2
    only the Pokémon that came out are known, so unseen ones are 0."""
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


# f_ features

def f_features(battle, pokedex, type_chart, healing_moves, setup_moves):
    """The f_ features of one battle: team composition, HP and momentum, statuses,
    active boosts, P1 items, move usage, stat spreads and type matchups.
    The Pokémon timeline is replayed turn by turn to track HP, status and moves."""
    f = {}
    timeline = battle["battle_timeline"]
    p1_team = battle["p1_team_details"]
    p1_names = [p["name"] for p in p1_team]

    # P2 Pokémon seen so far (sorted so the float sums below do not depend on set order)
    p2_names = sorted({battle["p2_lead_details"]["name"]} | {t["p2_pokemon_state"]["name"] for t in timeline})
    p2_stats = [pokedex[n] for n in p2_names if n in pokedex]

    last_turn = timeline[-1]["turn"]
    turn_25 = max(1, last_turn - 5)  # start of the last-5-turns window

    f["f_p2_revealed_count"] = len(p2_names)
    f["f_p2_hidden_count"] = 6 - len(p2_names)
    f["f_p1_has_snorlax"] = int("snorlax" in p1_names)
    f["f_p2_has_snorlax"] = int("snorlax" in p2_names)

    # per-Pokémon state, updated turn by turn
    p1_hp = {p: 1.0 for p in p1_names}
    p2_hp = {p: 1.0 for p in p2_names}
    p1_status = {p: "nostatus" for p in p1_names}
    p2_status = {p: "nostatus" for p in p2_names}
    p1_moves = {p: set() for p in p1_names}
    p2_moves = {p: set() for p in p2_names}
    p1_hist, p2_hist = {}, {}  # turn -> average team HP

    # counters
    p1_sw = p2_sw = p1_sw5 = p2_sw5 = 0
    p1_setup = p2_setup = p1_rec = p2_rec = p1_failed = p2_failed = 0
    p1_spikes = p2_spikes = 0  # 1 if there are spikes on this side's field
    for side in ("p1", "p2"):
        for mv in SINGLE_MOVES:
            f[f"f_{side}_used_{mv}"] = 0

    prev1 = timeline[0]["p1_pokemon_state"]["name"]
    prev2 = timeline[0]["p2_pokemon_state"]["name"]
    for turn in timeline:
        t = turn["turn"]
        s1, s2 = turn["p1_pokemon_state"], turn["p2_pokemon_state"]

        # switches (total and in the last 5 turns)
        if s1["name"] != prev1:
            p1_sw += 1
            p1_sw5 += t >= turn_25
        if s2["name"] != prev2:
            p2_sw += 1
            p2_sw5 += t >= turn_25
        prev1, prev2 = s1["name"], s2["name"]

        # HP and status, and the team HP of this turn (unseen P2 Pokémon count as full HP)
        p1_hp[s1["name"]] = s1["hp_pct"]
        p2_hp[s2["name"]] = s2["hp_pct"]
        p1_status[s1["name"]] = s1["status"]
        p2_status[s2["name"]] = s2["status"]
        p1_hist[t] = sum(p1_hp.values()) / 6.0
        p2_hist[t] = (sum(p2_hp.values()) + f["f_p2_hidden_count"]) / 6.0

        # moves of both players
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
            if name == "spikes":  # spikes go on the opponent's side
                if side == "p1": p2_spikes = 1
                else: p1_spikes = 1
            if name == "rapidspin":  # rapid spin clears the user's own side
                if side == "p1": p1_spikes = 0
                else: p2_spikes = 0
            # an attack into protect is a failed move
            if move["category"] != "status" and move["type"] != "threequestionmarks" \
                    and "protect" in target_state["effects"]:
                if side == "p1": p1_failed += 1
                else: p2_failed += 1

    # final HP and its change over the last 5 turns
    f["f_p1_net_hp_pct"] = p1_hist[last_turn]
    f["f_hp_differential"] = p1_hist[last_turn] - p2_hist[last_turn]
    diff_25 = p1_hist.get(turn_25, p1_hist[1]) - p2_hist.get(turn_25, p2_hist[1])
    f["f_momentum_delta_hp_last_5"] = f["f_hp_differential"] - diff_25
    f["f_p1_switches_last_5"] = p1_sw5
    f["f_p2_switches_last_5"] = p2_sw5

    # number of Pokémon with each status
    for st in ["slp", "tox", "par", "brn", "frz", "fnt"]:
        f[f"f_p1_status_{st}"] = sum(s == st for s in p1_status.values())
        f[f"f_p2_status_{st}"] = sum(s == st for s in p2_status.values())
    f["f_alive_diff"] = (6 - f["f_p1_status_fnt"]) - (6 - f["f_p2_status_fnt"])

    # asleep Pokémon that know both rest and sleeptalk
    f["f_p1_resttalk_slp"] = sum(s == "slp" and {"rest", "sleeptalk"} <= p1_moves.get(p, set())
                                 for p, s in p1_status.items())
    f["f_p2_resttalk_slp"] = sum(s == "slp" and {"rest", "sleeptalk"} <= p2_moves.get(p, set())
                                 for p, s in p2_status.items())

    # boosts of the two active Pokémon at the last turn
    last1, last2 = timeline[-1]["p1_pokemon_state"], timeline[-1]["p2_pokemon_state"]
    for stat in ["atk", "def", "spa", "spd", "spe"]:
        f[f"f_p1_active_boost_{stat}"] = last1["boosts"][stat]
        f[f"f_p2_active_boost_{stat}"] = last2["boosts"][stat]

    # P1 items (P2's items are not known)
    items = [p.get("item", "") for p in p1_team]
    f["f_p1_leftovers_count"] = items.count("leftovers")
    for it in ["thickclub", "lightball", "miracleberry"]:
        f[f"f_p1_has_{it}"] = int(it in items)

    # revealed share of P2's movesets (4 moves per Pokémon), switches and move counters
    f["f_p2_revealed_move_capacity_ratio"] = sum(len(m) for m in p2_moves.values()) / (len(p2_names) * 4)
    f["f_switch_diff"] = p1_sw - p2_sw
    f["f_p1_setup_count"], f["f_p2_setup_count"] = p1_setup, p2_setup
    f["f_p1_recovery_count"], f["f_p2_recovery_count"] = p1_rec, p2_rec
    f["f_p1_failed_moves"], f["f_p2_failed_moves"] = p1_failed, p2_failed
    f["f_p1_field_spikes"], f["f_p2_field_spikes"] = p1_spikes, p2_spikes

    # stat spread, average stats, pressure and type matchups between the two teams
    for k, key in [("speed", "base_spe"), ("def", "base_def"), ("spd", "base_spd")]:
        f[f"f_p1_{k}_std"] = np.std([p[key] for p in p1_team])
    if p2_stats:
        for k, key in [("speed", "base_spe"), ("def", "base_def"), ("spd", "base_spd")]:
            f[f"f_p2_{k}_std"] = np.std([p[key] for p in p2_stats])
        m1 = {k: np.mean([p[k] for p in p1_team]) for k in ["base_spe", "base_atk", "base_spa", "base_def", "base_spd"]}
        m2 = {k: np.mean([p[k] for p in p2_stats]) for k in ["base_spe", "base_atk", "base_spa", "base_def", "base_spd"]}
        f["f_speed_advantage"] = m1["base_spe"] - m2["base_spe"]
        # average attack of one team divided by the matching average defense of the other
        f["f_p1_physical_pressure"] = m1["base_atk"] / (m2["base_def"] + 1)
        f["f_p1_special_pressure"] = m1["base_spa"] / (m2["base_spd"] + 1)
        f["f_p2_physical_pressure"] = m2["base_atk"] / (m1["base_def"] + 1)
        f["f_p2_special_pressure"] = m2["base_spa"] / (m1["base_spd"] + 1)
        # average best multiplier and number of immunities over every P1 vs P2 pair
        mult1 = [calculate_max_multiplier(a["types"], b["types"], type_chart) for a in p1_team for b in p2_stats]
        mult2 = [calculate_max_multiplier(b["types"], a["types"], type_chart) for a in p1_team for b in p2_stats]
        f["f_p1_max_offensive_multiplier"] = np.mean(mult1)
        f["f_p2_max_offensive_multiplier"] = np.mean(mult2)
        f["f_p1_immunity_count"] = sum(check_immunity(a["types"], b["types"], type_chart) for a in p1_team for b in p2_stats)
        f["f_p2_immunity_count"] = sum(check_immunity(b["types"], a["types"], type_chart) for a in p1_team for b in p2_stats)
    else:
        # no P2 Pokémon in the pokedex: neutral values
        for k in ["f_p2_speed_std", "f_p2_def_std", "f_p2_spd_std", "f_speed_advantage",
                  "f_p1_immunity_count", "f_p2_immunity_count"]:
            f[k] = 0.0
        for k in ["f_p1_physical_pressure", "f_p1_special_pressure", "f_p2_physical_pressure",
                  "f_p2_special_pressure", "f_p1_max_offensive_multiplier", "f_p2_max_offensive_multiplier"]:
            f[k] = 1.0

    # type matchup of the two active Pokémon
    t1 = next((p["types"] for p in p1_team if p["name"] == last1["name"]), ["notype"])
    t2 = next((p["types"] for p in p2_stats if p["name"] == last2["name"]), ["notype"])
    f["f_active_matchup_p1_to_p2"] = calculate_max_multiplier(t1, t2, type_chart)
    f["f_active_matchup_p2_to_p1"] = calculate_max_multiplier(t2, t1, type_chart)
    return f


# Feature table and column selection

def build_features(df, species, chart, healing_moves, vocab, pokedex, setup_moves):
    """Turn a whole DataFrame (train or test) into a feature table indexed by battle_id."""
    rows = []
    for b in df.to_dict("records"):
        f = battle_features(b, species, chart, healing_moves)
        f.update(identity_features(b, vocab))
        f.update(f_features(b, pokedex, chart, healing_moves, setup_moves))
        rows.append(f)
    return pd.DataFrame(rows).set_index("battle_id")


# Main columns removed for logistic regression (each one is a copy, a sum or a
# difference of other columns, or did not help in CV).
LINEAR_DROP = ["switches", "no_move_turns", "status_moves", "avg_base_power", "max_boost_sum",
               "damage_taken", "recent_damage",
               "active_boost_sum",
               "healed"]
# f_ columns removed for logistic regression (any column whose name contains one of these)
F_LINEAR_DROP = [
    "f_momentum", "switches_last_5", "f_switch_diff", "leftovers", "thickclub", "lightball", "miracleberry",
    "f_hp_differential", "f_p1_net_hp", "f_p1_status_", "f_p2_status_", "f_alive_diff", "_active_boost_",
    "f_p2_revealed_count", "f_p2_hidden_count", "has_snorlax", "f_active_matchup",
]


def select_features(X, kind):
    """Return the columns for a model: "tree" gets all 217 columns, "linear"
    (logistic regression) gets 143 without the diff_ columns and the columns in
    LINEAR_DROP and F_LINEAR_DROP."""
    if kind == "tree":
        return X
    drop = [c for c in X.columns if c.startswith("diff_")]
    drop += [f"{side}_{k}" for side in ("p1", "p2") for k in LINEAR_DROP]
    drop += [c for c in X.columns if c.startswith("f_") and any(p in c for p in F_LINEAR_DROP)]
    return X.drop(columns=drop)
