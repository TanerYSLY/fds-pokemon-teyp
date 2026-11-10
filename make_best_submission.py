"""Builds our best submission: logistic regression + HistGradientBoosting, weights 2:1.

"""

import os
import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so "src" can be imported from anywhere
from src.submission_best_features import (build_features, load_healing_moves, load_identity_vocab, load_species,
                          load_type_chart, select_features)
from src.submission_best_features_teammate import build_pokedex, load_setup_moves

data = sys.argv[1] if len(sys.argv) > 1 else "data"
train = pd.read_json(os.path.join(data, "train.jsonl"), lines=True)
test = pd.read_json(os.path.join(data, "test.jsonl"), lines=True)

# 1. Lookup tables. Only from train, so the test set doesn't influence anything.
species = load_species([train])
chart = load_type_chart(os.path.join(data, "type_chart_gen2.csv"))
healing = load_healing_moves([train])
vocab = load_identity_vocab([train])
pokedex = build_pokedex(train)
setup = load_setup_moves([train])

# 2. Features: one row per battle
print("building features...")
X = build_features(train, species, chart, healing, vocab, pokedex, setup)
X_test = build_features(test, species, chart, healing, vocab, pokedex, setup)
y = train.set_index("battle_id").loc[X.index, "player_won"]  # labels in the same order as X

# 3. Train both models on the whole training set
print("training...")
# Logistic regression needs scaled features. Small C = strong regularization,
# which worked best because many of our features are correlated.
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=5000))
# HistGradientBoosting params come from RandomizedSearchCV. min_samples_leaf is
# large on purpose: with small leaves the trees memorized the training set.
hgb = HistGradientBoostingClassifier(learning_rate=0.03313073862781126, max_iter=575, max_leaf_nodes=19,
                                     min_samples_leaf=290, l2_regularization=0.12563152773938666,
                                     random_state=42)
lr.fit(select_features(X, "linear"), y)
hgb.fit(select_features(X, "tree"), y)

# 4. Average the win probabilities. Logistic regression was the better model in CV,
# so it gets weight 2. Probability above 0.5 -> we predict P1 wins.
p = (2 * lr.predict_proba(select_features(X_test, "linear"))[:, 1]
     + hgb.predict_proba(select_features(X_test, "tree"))[:, 1]) / 3
submission = pd.DataFrame({"battle_id": X_test.index, "player_won": (p > 0.5).astype(int)})

out = "best_submission.csv"
submission.to_csv(out, index=False)
print(f"saved {out} ({len(submission)} rows)")
