# Pokemon Battle Prediction (FDS 2026)

Our code for the Sapienza Fundamentals of Data Science Kaggle challenge. Given the first 30 turns of a Gen 2 OU battle, predict whether player 1 wins.

Final result: 0.686 on the private leaderboard (0.696 public), 13th out of 47 teams.

## Our three submissions

| | Model | Features | CV | Public | Private | Script |
|---|---|---|---|---|---|---|
| 1 (best) | Logistic regression + HistGradientBoosting, weights 2:1 | 217 (both feature sets) | 0.688 | 0.696 | 0.686 | `make_submission_1.py` |
| 2 | Logistic regression + CatBoost, weights 1:1 | 73 (feature set of `submission_2_features.py`) | 0.684 | 0.692 | 0.680 | `make_submission_2.py` |
| 3 | Logistic regression + CatBoost, tuned together with Optuna | 79 (the same 73 + 6 Snorlax features) | 0.689 | 0.689 | 0.679 | `make_submission_3.py` |

CV for submissions is 5-fold repeated 3 times.

## What we did (submission 1)

Each battle becomes one row of features. Most of them describe the state at turn 30: HP left, how many Pokemon fainted, status conditions, boosts of the active Pokemon, type matchups, and how strong the remaining team is. We also added 1/0 columns for the 30 most common Pokemon (is it still alive?) and the `f_` columns, the same kind of features as in submissions 2 and 3.

Lookup tables like species stats or which moves heal are built from the training data only. The only extra file is the type chart that came with the competition.

The final model averages two models: logistic regression (weight 2) and HistGradientBoosting (weight 1). Logistic regression gets a smaller feature set, because many columns are just copies of others (for example `diff_hp_total` is `p1_hp_total - p2_hp_total`).

We compared everything with 5-fold CV repeated 3 times. CV accuracy:

| Model | CV |
|---|---|
| Logistic regression | 0.684 |
| HistGradientBoosting | 0.677 |
| Ensemble 2:1 | 0.688 |

## Submissions 2 and 3

These use the `f_` feature set alone, with logistic regression and CatBoost. In submission 3 we added six Snorlax features and tuned the hyperparameters and the voting weights together with Optuna.

They use the first version of that feature code. There, five small move groups (recovery, setup, phazing, cleric, rapid spin) are written by hand and the species table also reads the test teams (no labels). In submission 1 we replaced both: the move groups are found from the training data and the species table uses the training set only.

## Files

- `src/submission_1_features.py` - features of submission 1: our main features plus the `f_` features, all built from the training data only
- `src/submission_2_features.py`, `src/submission_3_features.py` - features of submissions 2 and 3 (submission 3 adds the Snorlax features on top of submission 2)
- `src/validation.py` - the CV setup of submissions
- `make_submission_1.py`, `make_submission_2.py`, `make_submission_3.py` - each one trains a model and writes one submission file
- `experiments_submission_1.ipynb` - the experiments behind submission 1
- `experiments_submission_2_and_3.ipynb` - the experiments behind submissions 2 and 3

## Running it

Put `train.jsonl`, `test.jsonl`, `sample_submission.csv` and `type_chart_gen2.csv` in a `data/` folder, then:

```
pip install -r requirements.txt
python make_submission_1.py data
python make_submission_2.py --data-dir data
python make_submission_3.py --data-dir data
```

Each script takes a few minutes. `make_submission_1.py` takes the data folder as its first argument (for example `/kaggle/input/<competition>`), the other two take `--data-dir` and find the Kaggle folder by themselves. The submission files are written to the current folder (`/kaggle/working` on Kaggle for submissions 2 and 3): `submission_1.csv`, `lr_cat_73_features.csv` and `lr_cat_79_features_v2.csv`.