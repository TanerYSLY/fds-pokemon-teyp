"""
Submission 3: Logistic Regression + CatBoost soft-voting ensemble.

Pipeline
--------
1. Load the competition data and build features (src/submission_3_features.py).
2. Build the soft-voting ensemble with the hyperparameters and ensemble
   weights listed in BEST_PARAMS (see the note there on how they were found).
3. Report the 5-fold stratified CV accuracy on train.jsonl.
4. Fit the ensemble on the full training set and predict the test set.
5. Write the predictions in the sample_submission.csv format.

Usage
-----
    python make_submission_3.py                       # auto-detects Kaggle vs local paths
    python make_submission_3.py --data-dir <path> --output-dir <path>
"""

import argparse
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.submission_3_features import build_feature_frames, load_jsonl, load_type_chart

# Configuration
COMPETITION_DIR = "fds-pokemon-battles-prediction-2026"
KAGGLE_DATA_DIR = f"/kaggle/input/{COMPETITION_DIR}"

SEED = 42          # used for the CV splits and both models
N_FOLDS = 5
SUBMISSION_FILENAME = "lr_cat_79_features_v2.csv"

# Hyperparameters and soft-voting weights of the final model.
# These were found with Optuna (TPE sampler, seed 42, 100 trials), jointly
# optimising both models' hyperparameters and the ensemble weights by
# maximising the mean accuracy of a 5-fold stratified CV on train.jsonl
# (best trial: 0.6891 mean CV accuracy). The search is not re-run here so the
# final submission is reproduced exactly and quickly.
BEST_PARAMS = {
    # Logistic Regression
    "lr_c": 0.07830591333580003,
    # CatBoost
    "cb_iterations": 1970,
    "cb_lr": 0.01987056118032681,
    "cb_depth": 9,
    "cb_l2": 7.249997638789686,
    "cb_random_strength": 0.44843199034866,
    "cb_border_count": 240,
    # Soft-voting weights
    "w_lr": 3.912985323019429,
    "w_cb": 2.726474219695041,
}


# Model construction
def build_ensemble(params):
    """
    Build the soft-voting ensemble from a dict of hyperparameters.

    - Logistic Regression (with feature standardisation), default L2 penalty
    - CatBoost gradient-boosted trees
    - Soft voting: the models' predicted probabilities are averaged using the
      weights `w_lr` and `w_cb`.
    """
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=params["lr_c"],
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        )),
    ])

    cb_model = CatBoostClassifier(
        iterations=params["cb_iterations"],
        learning_rate=params["cb_lr"],
        depth=params["cb_depth"],
        l2_leaf_reg=params["cb_l2"],
        random_strength=params["cb_random_strength"],
        border_count=params["cb_border_count"],
        eval_metric="Logloss",
        verbose=0,
        random_state=SEED,
    )

    return VotingClassifier(
        estimators=[("lr", lr_pipeline), ("catboost", cb_model)],
        voting="soft",
        weights=[params["w_lr"], params["w_cb"]],
        n_jobs=-1,
    )


# Main
parser = argparse.ArgumentParser(description="Create Submission 3.")
parser.add_argument("--data-dir", default=None,
                    help="Folder with train.jsonl, test.jsonl, sample_submission.csv, type_chart_gen2.csv")
parser.add_argument("--output-dir", default=None,
                    help="Where to write the submission CSV (default: /kaggle/working on Kaggle, else .)")
args = parser.parse_args()

# Paths: auto-detect the Kaggle environment unless given explicitly
on_kaggle = os.path.exists(KAGGLE_DATA_DIR)
data_dir = args.data_dir or (KAGGLE_DATA_DIR if on_kaggle else COMPETITION_DIR)
output_dir = args.output_dir or ("/kaggle/working" if on_kaggle else ".")
os.makedirs(output_dir, exist_ok=True)

# 1. Data + features
train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
test_data = load_jsonl(os.path.join(data_dir, "test.jsonl"))
type_chart = load_type_chart(os.path.join(data_dir, "type_chart_gen2.csv"))
print(f"Train battles: {len(train_data)} | Test battles: {len(test_data)}")

train_df, test_df = build_feature_frames(train_data, test_data, type_chart)

feature_cols = [c for c in train_df.columns if c not in ("battle_id", "label_player_won")]
X_train = train_df[feature_cols]
y_train = train_df["label_player_won"]
X_test = test_df[feature_cols]

# 2. Ensemble with the tuned hyperparameters
ensemble = build_ensemble(BEST_PARAMS)

# 3. Cross-validated accuracy on the training set
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"Evaluating the ensemble with {N_FOLDS}-fold CV...")
scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Fold accuracies: {np.round(scores, 4)}")
print(f"Mean accuracy:   {scores.mean():.4f} ± {scores.std():.4f}")
print("-" * 50)

# 4. Fit on the full training set and predict the test set
print("Fitting the final ensemble on the full training set...")
ensemble.fit(X_train, y_train)
print("Predicting the test set...")
test_predictions = ensemble.predict(X_test)

# 5. Write the submission
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
predictions_by_id = pd.Series(test_predictions, index=test_df["battle_id"])
submission_df = sample_submission.copy()
submission_df["player_won"] = submission_df["battle_id"].map(predictions_by_id).astype(int)

out_path = os.path.join(output_dir, SUBMISSION_FILENAME)
submission_df.to_csv(out_path, index=False)
print(f"Submission saved to '{out_path}'")
