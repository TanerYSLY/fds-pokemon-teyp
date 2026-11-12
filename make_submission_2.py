"""
Submission 2: Logistic Regression + CatBoost soft-voting ensemble (equal weights).

Usage
-----
    python make_submission_2.py                       # auto-detects Kaggle vs local paths
    python make_submission_2.py --data-dir <path> --output-dir <path>
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

from src.submission_2_features import build_feature_frames, load_jsonl, load_type_chart

# Configuration
COMPETITION_DIR = "fds-pokemon-battles-prediction-2026"
KAGGLE_DATA_DIR = f"/kaggle/input/{COMPETITION_DIR}"

SEED = 42          # used for the CV splits and both models
N_FOLDS = 5
SUBMISSION_FILENAME = "lr_cat_73_features.csv"

# Hyperparameters of the two models. Each model was tuned SEPARATELY with
# Optuna (the Logistic Regression and CatBoost searches were independent), and
# the tuned models were then combined with equal weights. The search is not
# re-run here so the final submission is reproduced exactly and quickly.
LR_PARAMS = {
    "C": 0.019618501281982222,
}

CATBOOST_PARAMS = {
    "iterations": 1601,
    "learning_rate": 0.007622507163328224,
    "depth": 8,
    "l2_leaf_reg": 8.784785985319969,
    "random_strength": 0.7090587412583095,
    "border_count": 53,
}

ENSEMBLE_WEIGHTS = [1, 1]   # [logistic regression, catboost]: plain probability average


# Model construction
def build_ensemble():
    """
    Build the soft-voting ensemble.

    - Logistic Regression (with feature standardisation), default L2 penalty
    - CatBoost gradient-boosted trees
    - Soft voting: the models' predicted probabilities are averaged with
      equal weights.
    """
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=LR_PARAMS["C"],
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        )),
    ])

    cb_model = CatBoostClassifier(
        **CATBOOST_PARAMS,
        eval_metric="Logloss",
        verbose=0,
        random_state=SEED,
    )

    return VotingClassifier(
        estimators=[("lr", lr_pipeline), ("catboost", cb_model)],
        voting="soft",
        weights=ENSEMBLE_WEIGHTS,
    )


# Main
parser = argparse.ArgumentParser(description="Create Submission 2.")
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

# Data + features
train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
test_data = load_jsonl(os.path.join(data_dir, "test.jsonl"))
type_chart = load_type_chart(os.path.join(data_dir, "type_chart_gen2.csv"))
print(f"Train battles: {len(train_data)} | Test battles: {len(test_data)}")

train_df, test_df = build_feature_frames(train_data, test_data, type_chart)

feature_cols = [c for c in train_df.columns if c not in ("battle_id", "label_player_won")]
X_train = train_df[feature_cols]
y_train = train_df["label_player_won"]
X_test = test_df[feature_cols]

# Ensemble with the tuned hyperparameters
ensemble = build_ensemble()

# Cross-validated accuracy on the training set
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
print(f"Evaluating the ensemble with {N_FOLDS}-fold CV...")
scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Fold accuracies: {np.round(scores, 4)}")
print(f"Mean accuracy:   {scores.mean():.4f} ± {scores.std():.4f}")
print("-" * 50)

# Fit on the full training set and predict the test set
print("Fitting the final ensemble on the full training set...")
ensemble.fit(X_train, y_train)
print("Predicting the test set...")
test_predictions = ensemble.predict(X_test)

# Write the submission
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
predictions_by_id = pd.Series(test_predictions, index=test_df["battle_id"])
submission_df = sample_submission.copy()
submission_df["player_won"] = submission_df["battle_id"].map(predictions_by_id).astype(int)

out_path = os.path.join(output_dir, SUBMISSION_FILENAME)
submission_df.to_csv(out_path, index=False)
print(f"Submission saved to '{out_path}'")
