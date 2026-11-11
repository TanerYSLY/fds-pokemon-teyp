"""
Feature engineering for Submission 2 (FDS Pokémon Battle Prediction, Gen 2 OU).

Turns the raw competition data into a flat, tabular feature matrix
(73 features). This module also holds the shared building blocks:
Submission 3 (`src/submission_3_features.py`) imports it and only adds one
more feature group (Snorlax) on top.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm


# Constants
BOOST_STATS = ["atk", "def", "spa", "spd", "spe"]
STATUS_TYPES = ["slp", "tox", "par", "brn", "frz", "fnt"]
RECENT_WINDOW = 5  # number of final turns used for the "recent" / momentum features

# Strategic Gen 2 OU move groups
RECOVERY_MOVES = {"rest", "recover", "softboiled", "milkdrink", "morningsun", "synthesis", "moonlight"}
SETUP_MOVES = {"curse", "swordsdance", "amnesia", "agility", "bellydrum", "growth"}
PHAZE_MOVES = {"roar", "whirlwind"}            # force the opponent to switch
CLERIC_MOVES = {"healbell", "aromatherapy"}    # cure the team's status conditions
HAZARD_CONTROL_MOVES = {"rapidspin"}           # remove entry hazards


# 1. Data loading
def load_jsonl(file_path):
    """Read a .jsonl file (one JSON object per line) into a list of dicts."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_type_chart(type_path):
    """
    Load the supplied Gen 2 type chart CSV into a lookup dictionary.

    Returns
    -------
    dict[(attacking_type, defending_type)] -> damage multiplier (0, 0.5, 1, 2)
    """
    type_df = pd.read_csv(type_path)
    type_chart = {}
    for _, row in type_df.iterrows():
        type_chart[(row["attacking_type"], row["defending_type"])] = row["multiplier"]
    return type_chart


# 2. Pokédex built from the competition data
def build_pokedex(dataset):
    """
    Build a {pokemon_name: details} lookup (base stats, types, ...) using
    the Pokémon that appear in the provided battles.

    Player 2 only reveals its Pokémon progressively, so we need this lookup to
    find the base stats / types of every P2 Pokémon seen during the timeline
    (the same species may appear in P1's team details in some other battle).
    The first occurrence of each name is kept.
    """
    pokedex = {}
    for battle in dataset:
        for poke in battle.get("p1_team_details", []):
            if poke["name"] not in pokedex:
                pokedex[poke["name"]] = poke
        if "p2_lead_details" in battle:
            lead = battle["p2_lead_details"]
            if lead["name"] not in pokedex:
                pokedex[lead["name"]] = lead
    return pokedex


# 3. Type-effectiveness helpers
def calculate_max_multiplier(atk_types, def_types, type_chart):
    """
    Best damage multiplier an attacker can get against a defender, using the
    attacker's STAB types as the attacking types.

    For every attacking type we multiply the multipliers against each of the
    defender's types (dual-type defenders), then keep the maximum.
    The placeholder type "notype" is ignored.
    """
    max_mult = 0.0
    for at in atk_types:
        if at == "notype":
            continue
        current_mult = 1.0
        for dt in def_types:
            if dt == "notype":
                continue
            current_mult *= type_chart.get((at, dt), 1.0)
        if current_mult > max_mult:
            max_mult = current_mult
    return max_mult


def check_immunity(atk_types, def_types, type_chart):
    """Return True if any attacking type has a 0x multiplier on any defending type."""
    for at in atk_types:
        if at == "notype":
            continue
        for dt in def_types:
            if dt == "notype":
                continue
            if type_chart.get((at, dt), 1.0) == 0.0:
                return True
    return False


# 4. Timeline tracking
@dataclass
class SideState:
    """
    Running state of ONE player, updated turn by turn while replaying the
    observed part of the battle. Everything is tracked per Pokémon name.
    """
    names: object            # P1: list of the 6 team names; P2: set of revealed names
    hidden_count: int        # P2 Pokémon not revealed yet (assumed at full HP); 0 for P1
    prev_active: str         # name of the Pokémon that was active on the previous turn
    hps: dict                # last known HP fraction (1.0 = full)
    status: dict             # last known status (nostatus, slp, tox, par, brn, frz, fnt)
    revealed_moves: dict     # set of moves each Pokémon has used so far
    boosts: dict             # last known stat boosts
    hp_history: dict = field(default_factory=dict)  # turn -> team-wide HP fraction
    switches: int = 0
    switches_last_5: int = 0
    setup: int = 0           # setup moves used (Curse, Swords Dance, ...)
    recovery: int = 0        # recovery moves used (Rest, Recover, ...)
    failed: int = 0          # damaging moves used into a Protect
    spikes_on_field: int = 0  # 1 if spikes are on THIS player's side of the field
    used_phaze: int = 0
    used_cleric: int = 0
    used_rapidspin: int = 0


def new_side_state(names, hidden_count, first_active):
    """Create a fresh SideState: everyone at full HP, no status, no moves, no boosts."""
    return SideState(
        names=names,
        hidden_count=hidden_count,
        prev_active=first_active,
        hps={p: 1.0 for p in names},
        status={p: "nostatus" for p in names},
        revealed_moves={p: set() for p in names},
        boosts={p: {stat: 0 for stat in BOOST_STATS} for p in names},
    )


def record_switch(side, active_name, t_num, recent_start_turn):
    """Count a switch if the active Pokémon changed since the previous turn."""
    if active_name != side.prev_active:
        side.switches += 1
        if t_num >= recent_start_turn:
            side.switches_last_5 += 1
    side.prev_active = active_name


def record_active_state(side, pokemon_state):
    """Store the HP, status and boosts of the Pokémon that is active this turn."""
    name = pokemon_state["name"]
    side.hps[name] = pokemon_state["hp_pct"]
    side.status[name] = pokemon_state["status"]
    side.boosts[name] = pokemon_state["boosts"]


def record_team_hp(side, t_num):
    """
    Save the team-wide HP fraction (average over 6 slots) for this turn.
    Hidden P2 Pokémon are assumed to be at full HP.
    """
    side.hp_history[t_num] = (sum(side.hps.values()) + side.hidden_count) / 6.0


def record_move(side, opponent, move, active_name, opponent_state):
    """
    Update the counters / flags of `side` after it used `move` this turn.

    `opponent_state` is the opponent's active Pokémon state this turn, used to
    detect damaging moves wasted into a Protect.
    """
    move_name = move["name"]
    side.revealed_moves[active_name].add(move_name)

    # Strategic move categories
    if move_name in RECOVERY_MOVES:
        side.recovery += 1
    if move_name in SETUP_MOVES:
        side.setup += 1
    if move_name in PHAZE_MOVES:
        side.used_phaze = 1
    if move_name in CLERIC_MOVES:
        side.used_cleric = 1
    if move_name in HAZARD_CONTROL_MOVES:
        side.used_rapidspin = 1

    # Entry hazards: Spikes land on the opponent's side, Rapid Spin clears our own
    if move_name == "spikes":
        opponent.spikes_on_field = 1
    if move_name == "rapidspin":
        side.spikes_on_field = 0

    # Damaging move into a Protect = wasted turn
    if move["category"] != "status" and move["type"] != "threequestionmarks":
        if "protect" in opponent_state["effects"]:
            side.failed += 1


def process_timeline(timeline, p1, p2, recent_start_turn):
    """Replay the observed turns, updating both players' SideState in place."""
    for turn in timeline:
        t_num = turn["turn"]
        p1_state = turn["p1_pokemon_state"]
        p2_state = turn["p2_pokemon_state"]

        record_switch(p1, p1_state["name"], t_num, recent_start_turn)
        record_switch(p2, p2_state["name"], t_num, recent_start_turn)

        record_active_state(p1, p1_state)
        record_active_state(p2, p2_state)

        record_team_hp(p1, t_num)
        record_team_hp(p2, t_num)

        p1_move = turn.get("p1_move_details")
        p2_move = turn.get("p2_move_details")
        if p1_move:
            record_move(p1, p2, p1_move, p1_state["name"], p2_state)
        if p2_move:
            record_move(p2, p1, p2_move, p2_state["name"], p1_state)


# 5. Feature groups
def composition_features(p1_team_names, p2_revealed_names):
    """How much of P2's team is known, and whether either side has Snorlax."""
    return {
        "f_p2_revealed_count": len(p2_revealed_names),
        "f_p2_hidden_count": 6 - len(p2_revealed_names),
        "f_p1_has_snorlax": 1 if "snorlax" in p1_team_names else 0,
        "f_p2_has_snorlax": 1 if "snorlax" in p2_revealed_names else 0,
    }


def move_flag_features(p1, p2):
    """Binary flags: has each player used a phazing / cleric / Rapid Spin move?"""
    return {
        "f_p1_used_phaze": p1.used_phaze,
        "f_p2_used_phaze": p2.used_phaze,
        "f_p1_used_cleric": p1.used_cleric,
        "f_p2_used_cleric": p2.used_cleric,
        "f_p1_used_rapidspin": p1.used_rapidspin,
        "f_p2_used_rapidspin": p2.used_rapidspin,
    }


def hp_features(p1, p2, last_turn_num):
    """Team HP of P1 and the HP differential at the last observed turn (positive = P1 ahead)."""
    p1_hp = p1.hp_history[last_turn_num]
    p2_hp = p2.hp_history[last_turn_num]
    return {
        "f_p1_net_hp_pct": p1_hp,
        "f_hp_differential": p1_hp - p2_hp,
    }


def momentum_features(p1, p2, last_turn_num, recent_start_turn):
    """
    Momentum: how much did the HP differential change over the last ~5 turns?
    Also the number of switches each player made in that window.
    """
    hp_diff_now = p1.hp_history[last_turn_num] - p2.hp_history[last_turn_num]
    hp_diff_before = (
        p1.hp_history.get(recent_start_turn, p1.hp_history[1])
        - p2.hp_history.get(recent_start_turn, p2.hp_history[1])
    )
    return {
        "f_momentum_delta_hp_last_5": hp_diff_now - hp_diff_before,
        "f_p1_switches_last_5": p1.switches_last_5,
        "f_p2_switches_last_5": p2.switches_last_5,
    }


def status_features(p1, p2):
    """
    Number of Pokémon of each team in each status condition (incl. fainted),
    plus the difference in Pokémon still alive (positive = P1 ahead).
    """
    features = {}
    for st in STATUS_TYPES:
        features[f"f_p1_status_{st}"] = sum(1 for s in p1.status.values() if s == st)
        features[f"f_p2_status_{st}"] = sum(1 for s in p2.status.values() if s == st)

    p1_alive = 6 - features["f_p1_status_fnt"]
    p2_alive = 6 - features["f_p2_status_fnt"]
    features["f_alive_diff"] = p1_alive - p2_alive
    return features


def count_rest_talk_sleepers(side):
    """Asleep Pokémon known to have both Rest and Sleep Talk (sleep is less of a liability)."""
    return sum(
        1 for p, s in side.status.items()
        if s == "slp" and "rest" in side.revealed_moves[p] and "sleeptalk" in side.revealed_moves[p]
    )


def rest_talk_features(p1, p2):
    """Rest + Sleep Talk sleepers for both players."""
    return {
        "f_p1_resttalk_slp": count_rest_talk_sleepers(p1),
        "f_p2_resttalk_slp": count_rest_talk_sleepers(p2),
    }


def active_boost_features(last_p1_state, last_p2_state):
    """Stat boosts of the two active Pokémon at the last observed turn."""
    features = {}
    for stat in BOOST_STATS:
        features[f"f_p1_active_boost_{stat}"] = last_p1_state["boosts"][stat]
        features[f"f_p2_active_boost_{stat}"] = last_p2_state["boosts"][stat]
    return features


def item_features(p1_team_details):
    """Held items of P1's team (only P1's items are fully known)."""
    p1_items = [p.get("item", "") for p in p1_team_details]
    return {
        "f_p1_leftovers_count": p1_items.count("leftovers"),
        "f_p1_has_thickclub": 1 if "thickclub" in p1_items else 0,
        "f_p1_has_lightball": 1 if "lightball" in p1_items else 0,
        "f_p1_has_miracleberry": 1 if "miracleberry" in p1_items else 0,
    }


def move_coverage_features(p2):
    """How much of P2's movesets we have already seen (4 move slots per Pokémon)."""
    total_unique_moves = sum(len(moves) for moves in p2.revealed_moves.values())
    return {
        "f_p2_revealed_move_capacity_ratio": total_unique_moves / (len(p2.names) * 4),
    }


def strategy_features(p1, p2):
    """Play-style counters accumulated over the timeline (switching, setup, recovery, hazards)."""
    return {
        "f_switch_diff": p1.switches - p2.switches,
        "f_p1_setup_count": p1.setup,
        "f_p2_setup_count": p2.setup,
        "f_p1_recovery_count": p1.recovery,
        "f_p2_recovery_count": p2.recovery,
        "f_p1_failed_moves": p1.failed,
        "f_p2_failed_moves": p2.failed,
        "f_p1_field_spikes": p1.spikes_on_field,
        "f_p2_field_spikes": p2.spikes_on_field,
    }


def stat_values(pokemon, key):
    """List of one base stat (e.g. 'base_spe') over a group of Pokémon."""
    return [p[key] for p in pokemon]


def avg_stat(pokemon, key):
    """Average of one base stat over a group of Pokémon."""
    return np.mean(stat_values(pokemon, key))


def p1_stat_spread_features(p1_team_details):
    """Spread (std) of P1's base Speed / Defense / Special Defense: how uneven the team is."""
    return {
        "f_p1_speed_std": np.std(stat_values(p1_team_details, "base_spe")),
        "f_p1_def_std": np.std(stat_values(p1_team_details, "base_def")),
        "f_p1_spd_std": np.std(stat_values(p1_team_details, "base_spd")),
    }


def base_stat_matchup_features(p1_team_details, p2_stats):
    """
    P2 stat spread, the average Speed gap, and the offence-vs-defence
    "pressure" of each team on the other (the +1 avoids division by zero).

    Neutral values are returned when no P2 Pokémon was found in the pokedex.
    """
    if not p2_stats:
        return {
            "f_p2_speed_std": 0.0,
            "f_p2_def_std": 0.0,
            "f_p2_spd_std": 0.0,
            "f_speed_advantage": 0.0,
            "f_p1_physical_pressure": 1.0,
            "f_p1_special_pressure": 1.0,
            "f_p2_physical_pressure": 1.0,
            "f_p2_special_pressure": 1.0,
        }

    features = {
        "f_p2_speed_std": np.std(stat_values(p2_stats, "base_spe")),
        "f_p2_def_std": np.std(stat_values(p2_stats, "base_def")),
        "f_p2_spd_std": np.std(stat_values(p2_stats, "base_spd")),
        # Positive = P1 is faster on average
        "f_speed_advantage": avg_stat(p1_team_details, "base_spe") - avg_stat(p2_stats, "base_spe"),
    }

    p1_avg_atk = avg_stat(p1_team_details, "base_atk")
    p1_avg_spa = avg_stat(p1_team_details, "base_spa")
    p1_avg_def = avg_stat(p1_team_details, "base_def")
    p1_avg_spd = avg_stat(p1_team_details, "base_spd")

    p2_avg_atk = avg_stat(p2_stats, "base_atk")
    p2_avg_spa = avg_stat(p2_stats, "base_spa")
    p2_avg_def = avg_stat(p2_stats, "base_def")
    p2_avg_spd = avg_stat(p2_stats, "base_spd")

    features["f_p1_physical_pressure"] = p1_avg_atk / (p2_avg_def + 1)
    features["f_p1_special_pressure"] = p1_avg_spa / (p2_avg_spd + 1)
    features["f_p2_physical_pressure"] = p2_avg_atk / (p1_avg_def + 1)
    features["f_p2_special_pressure"] = p2_avg_spa / (p1_avg_spd + 1)
    return features


def type_matchup_features(p1_team_details, p2_stats, type_chart):
    """
    Type matchups over every (P1 Pokémon, revealed P2 Pokémon) pair: the mean
    best offensive multiplier of each side and the number of immunities.

    Neutral values are returned when no P2 Pokémon was found in the pokedex.
    """
    if not p2_stats:
        return {
            "f_p1_max_offensive_multiplier": 1.0,
            "f_p2_max_offensive_multiplier": 1.0,
            "f_p1_immunity_count": 0,
            "f_p2_immunity_count": 0,
        }

    p1_mults, p2_mults = [], []
    p1_immunities = p2_immunities = 0

    for p1_poke in p1_team_details:
        for p2_poke in p2_stats:
            p1_mults.append(calculate_max_multiplier(p1_poke["types"], p2_poke["types"], type_chart))
            p2_mults.append(calculate_max_multiplier(p2_poke["types"], p1_poke["types"], type_chart))

            if check_immunity(p1_poke["types"], p2_poke["types"], type_chart):
                p1_immunities += 1
            if check_immunity(p2_poke["types"], p1_poke["types"], type_chart):
                p2_immunities += 1

    return {
        "f_p1_max_offensive_multiplier": np.mean(p1_mults),
        "f_p2_max_offensive_multiplier": np.mean(p2_mults),
        "f_p1_immunity_count": p1_immunities,
        "f_p2_immunity_count": p2_immunities,
    }


def active_matchup_features(last_p1_state, last_p2_state, p1_team_details, p2_stats, type_chart):
    """Type matchup between the two Pokémon active at the last observed turn."""
    last_p1_types = next(
        (p["types"] for p in p1_team_details if p["name"] == last_p1_state["name"]), ["notype"]
    )
    last_p2_types = next(
        (p["types"] for p in p2_stats if p["name"] == last_p2_state["name"]), ["notype"]
    )
    return {
        "f_active_matchup_p1_to_p2": calculate_max_multiplier(last_p1_types, last_p2_types, type_chart),
        "f_active_matchup_p2_to_p1": calculate_max_multiplier(last_p2_types, last_p1_types, type_chart),
    }


# 6. Per-battle feature extraction
@dataclass
class BattleContext:
    """
    Everything the feature groups need to know about ONE battle: the static
    team information plus both players' state after replaying the timeline.
    """
    p1_team_details: list    # P1's six Pokémon (base stats, types, items)
    p1_team_names: list
    p2_revealed_names: set   # P2 Pokémon revealed so far
    p2_stats: list           # pokedex entries of the revealed P2 Pokémon
    last_turn_num: int
    recent_start_turn: int   # first turn of the "recent" window
    last_p1_state: dict      # P1's active Pokémon at the last observed turn
    last_p2_state: dict
    p1: SideState
    p2: SideState


def get_p2_revealed_names(battle, timeline):
    """Names of all P2 Pokémon revealed so far (the lead plus everything seen on the timeline)."""
    revealed = set([battle["p2_lead_details"]["name"]])
    for turn in timeline:
        revealed.add(turn["p2_pokemon_state"]["name"])
    return revealed


def lookup_pokedex_entries(names, pokedex):
    """Base stats / types of the given Pokémon, skipping names missing from the pokedex."""
    return [pokedex[name] for name in names if name in pokedex]


def build_battle_context(battle, pokedex):
    """
    Collect the static information of a battle and replay its observed
    timeline. Returns None if the battle has no timeline.
    """
    timeline = battle.get("battle_timeline", [])
    if not timeline:
        return None

    p1_team_details = battle["p1_team_details"]
    p1_team_names = [p["name"] for p in p1_team_details]
    p2_revealed_names = get_p2_revealed_names(battle, timeline)
    p2_stats = lookup_pokedex_entries(p2_revealed_names, pokedex)

    # "Recent window": the last ~5 turns of the observed timeline
    last_turn_num = timeline[-1]["turn"]
    recent_start_turn = max(1, last_turn_num - RECENT_WINDOW)

    # Replay the timeline to get each player's running state
    p1 = new_side_state(p1_team_names, 0, timeline[0]["p1_pokemon_state"]["name"])
    p2 = new_side_state(p2_revealed_names, 6 - len(p2_revealed_names),
                        timeline[0]["p2_pokemon_state"]["name"])
    process_timeline(timeline, p1, p2, recent_start_turn)

    return BattleContext(
        p1_team_details=p1_team_details,
        p1_team_names=p1_team_names,
        p2_revealed_names=p2_revealed_names,
        p2_stats=p2_stats,
        last_turn_num=last_turn_num,
        recent_start_turn=recent_start_turn,
        last_p1_state=timeline[-1]["p1_pokemon_state"],
        last_p2_state=timeline[-1]["p2_pokemon_state"],
        p1=p1,
        p2=p2,
    )


def build_feature_groups(ctx, type_chart):
    """
    Compute all feature groups of Submission 2 for one battle.
    The order of the groups below defines the column order of the matrix.
    """
    p1, p2 = ctx.p1, ctx.p2
    features = {}
    features.update(composition_features(ctx.p1_team_names, ctx.p2_revealed_names))
    features.update(move_flag_features(p1, p2))
    features.update(hp_features(p1, p2, ctx.last_turn_num))
    features.update(momentum_features(p1, p2, ctx.last_turn_num, ctx.recent_start_turn))
    features.update(status_features(p1, p2))
    features.update(rest_talk_features(p1, p2))
    features.update(active_boost_features(ctx.last_p1_state, ctx.last_p2_state))
    features.update(item_features(ctx.p1_team_details))
    features.update(move_coverage_features(p2))
    features.update(strategy_features(p1, p2))
    features.update(p1_stat_spread_features(ctx.p1_team_details))
    features.update(base_stat_matchup_features(ctx.p1_team_details, ctx.p2_stats))
    features.update(type_matchup_features(ctx.p1_team_details, ctx.p2_stats, type_chart))
    features.update(active_matchup_features(
        ctx.last_p1_state, ctx.last_p2_state, ctx.p1_team_details, ctx.p2_stats, type_chart))
    return features


def insert_extra_groups(features, extra_groups, ctx):
    """
    Insert additional feature groups into `features`, each right after an
    existing feature.

    extra_groups : dict {name of an existing feature: function(ctx) -> dict}
    """
    missing = [name for name in extra_groups if name not in features]
    if missing:
        raise KeyError(f"Cannot insert extra feature groups after unknown features: {missing}")

    result = {}
    for name, value in features.items():
        result[name] = value
        if name in extra_groups:
            result.update(extra_groups[name](ctx))
    return result


def extract_features(battle, pokedex, type_chart, extra_groups=None):
    """
    Turn one battle (dict) into a flat dict of numeric features.

    Steps: (1) replay the battle (`build_battle_context`), (2) compute the
    feature groups (`build_feature_groups`), (3) optionally insert extra
    groups (see `insert_extra_groups`).
    """
    features = {"battle_id": battle.get("battle_id")}

    ctx = build_battle_context(battle, pokedex)
    if ctx is None:
        return features

    groups = build_feature_groups(ctx, type_chart)
    if extra_groups:
        groups = insert_extra_groups(groups, extra_groups, ctx)
    features.update(groups)

    # Target (None for the test set, where `player_won` is absent)
    features["label_player_won"] = battle.get("player_won")

    return features


# 7. Wrapper used by the make_submission scripts
def build_feature_frames(train_data, test_data, type_chart, extra_groups=None):
    """
    Build the train and test feature DataFrames.

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Columns: `battle_id`, all `f_*` features, and `label_player_won`
        (NaN/None in the test frame).
    """
    pokedex = build_pokedex(train_data + test_data)

    train_df = pd.DataFrame([extract_features(b, pokedex, type_chart, extra_groups)
                             for b in tqdm(train_data, desc="train features")])
    test_df = pd.DataFrame([extract_features(b, pokedex, type_chart, extra_groups)
                            for b in tqdm(test_data, desc="test features")])
    return train_df, test_df