"""
Feature engineering for Submission 3 (FDS Pokémon Battle Prediction, Gen 2 OU).

Submission 3 uses every feature of Submission 2 (`src/submission_2_features.py`)
plus one additional group: the Snorlax features (Snorlax HP, Attack boost and
whether Curse was revealed, for both players). That is 79 features
instead of 73.

Everything else (data loading, pokédex, timeline tracking, all the other
feature groups) is imported from `submission_2_features.py` unchanged. The
Snorlax group is inserted right after the HP features (after
`f_hp_differential`), which keeps the column order of the submitted model
exactly as it was.
"""

from src.submission_2_features import build_feature_frames as build_base_feature_frames
from src.submission_2_features import extract_features as extract_base_features
from src.submission_2_features import load_jsonl, load_type_chart  # re-exported for make_submission_3.py

__all__ = ["load_jsonl", "load_type_chart", "snorlax_features", "extract_features", "build_feature_frames"]

# Name of the Submission 2 feature after which the Snorlax group is inserted
SNORLAX_INSERT_AFTER = "f_hp_differential"


def snorlax_features(ctx):
    """
    Snorlax is a key win condition in Gen 2 OU (Curse sets, huge bulk): its HP,
    Attack boost and whether Curse was revealed, for both players.

    `ctx` is the `BattleContext` of the battle (see submission_2_features.py).
    """
    p1, p2 = ctx.p1, ctx.p2
    p1_has = "snorlax" in p1.names
    p2_has = "snorlax" in p2.names
    return {
        "f_p1_snorlax_hp": p1.hps.get("snorlax", 0.0) if p1_has else 0.0,
        "f_p2_snorlax_hp": p2.hps.get("snorlax", 0.0) if p2_has else 0.0,
        "f_p1_snorlax_atk_boost": p1.boosts.get("snorlax", {}).get("atk", 0) if p1_has else 0,
        "f_p2_snorlax_atk_boost": p2.boosts.get("snorlax", {}).get("atk", 0) if p2_has else 0,
        "f_p1_snorlax_curse_revealed": 1 if p1_has and "curse" in p1.revealed_moves.get("snorlax", []) else 0,
        "f_p2_snorlax_curse_revealed": 1 if p2_has and "curse" in p2.revealed_moves.get("snorlax", []) else 0,
    }


# The additional feature groups of Submission 3: {feature to insert after: group function}
EXTRA_GROUPS = {SNORLAX_INSERT_AFTER: snorlax_features}


def extract_features(battle, pokedex, type_chart):
    """Submission 2 features for one battle, plus the Snorlax group."""
    return extract_base_features(battle, pokedex, type_chart, extra_groups=EXTRA_GROUPS)


def build_feature_frames(train_data, test_data, type_chart):
    """Train and test feature DataFrames: Submission 2 features plus the Snorlax group."""
    return build_base_feature_frames(train_data, test_data, type_chart, extra_groups=EXTRA_GROUPS)
