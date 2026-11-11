# Pokemon Battle Prediction (FDS 2026)

Our code for the Sapienza Fundamentals of Data Science Kaggle challenge. Given the first 30 turns of a Gen 2 OU battle, predict whether player 1 wins.

Final result: 0.686 on the private leaderboard (0.696 public), 13th out of 47 teams.

## Our three submissions

| | Model | Features | CV | Public | Private | Script |
|---|---|---|---|---|---|---|
| 1 (best) | Logistic regression + HistGradientBoosting, weights 2:1 | 217 (both feature sets) | 0.688 | 0.696 | 0.686 | `make_best_submission.py` |
| 2 | Logistic regression + CatBoost, weights 1:1 | 71 (Enes's feature set) | 0.684 | 0.692 | 0.680 | `make_submission_2.py` |
| 3 | Logistic regression + CatBoost, tuned together with Optuna | 77 (the same 71 + 6 Snorlax features) | 0.689 | 0.689 | 0.679 | `make_submission_3.py` |

CV for submission 1 is 5-fold repeated 3 times. For submissions 2 and 3 it is one 5-fold run.

## What we did (submission 1)

Each battle becomes one row of features. Most of them describe the state at turn 30: HP left, how many Pokemon fainted, status conditions, boosts of the active Pokemon, type matchups, and how strong the remaining team is. We also added 1/0 columns for the 30 most common Pokemon (is it still alive?) and the features our teammate Enes wrote (the `f_` columns).

Lookup tables like species stats or which moves heal are built from the training data only. The only extra file is the type chart that came with the competition.

The final model averages two models: logistic regression (weight 2) and HistGradientBoosting (weight 1). Logistic regression gets a smaller feature set, because many columns are just copies of others (for example `diff_hp_total` is `p1_hp_total - p2_hp_total`).

We compared everything with 5-fold CV repeated 3 times. CV accuracy:

| Model | CV |
|---|---|
| Logistic regression | 0.684 |
| HistGradientBoosting | 0.677 |
| Ensemble 2:1 | 0.688 |

## Submissions 2 and 3

These use Enes's feature set alone, with logistic regression and CatBoost. In submission 3 we added six Snorlax features and tuned the hyperparameters and the voting weights together with Optuna.

They use the first version of his feature code. There, five small move groups (recovery, setup, phazing, cleric, rapid spin) are written by hand and the species table also reads the test teams (no labels). In submission 1 we replaced both: the move groups are found from the training data and the species table uses the training set only.

## Files

- `src/submission_best_features.py` - our features (submission 1)
- `src/submission_best_features_teammate.py` - Enes's features, changed to use training data only (submission 1)
- `src/submission_2_features.py`, `src/submission_3_features.py` - Enes's features for submissions 2 and 3
- `src/validation.py` - the CV setup of submission 1
- `make_best_submission.py`, `make_submission_2.py`, `make_submission_3.py` - each one trains a model and writes one submission file
- `experiments_submission_1.ipynb` - the experiments behind submission 1
- `experiments_submission_2_and_3.ipynb` - the experiments behind submissions 2 and 3

## Running it

Put `train.jsonl`, `test.jsonl`, `sample_submission.csv` and `type_chart_gen2.csv` in a `data/` folder, then:

```
pip install -r requirements.txt
python make_best_submission.py
python make_submission_2.py --data-dir data
python make_submission_3.py --data-dir data
```

Each script takes a few minutes. For the first one you can also pass the data folder: `python make_best_submission.py /kaggle/input/<competition>`. The other two find the Kaggle folder by themselves.

## Development notebooks

The `development/` folder has the notebooks we worked in during the competition, with their outputs saved. They show the steps before the final code: the first features, the diagnosis and the error analysis. The text in them is in Turkish, so here is where to find the experiments from the report:

- `01_veri_anlama.ipynb` - looking at the data
- `02_baseline.ipynb` - the first 49 features and the first model comparison
- `03_features_v2.ipynb` - type and team strength features, ablation
- `04_teshis.ipynb` - train vs CV accuracy, learning curves, how sure the model is, error analysis
- `05_deneyler.ipynb` - the four features from the error analysis, the error check per Pokemon, the identity columns, random search, Naive Bayes and LDA, the ensemble and the threshold
- `experiments/hgb_variance.py` - fewer trees and bagging for HistGradientBoosting

These notebooks use the feature code as it was at that time (`development/src/`). To run them, start Jupyter inside `development/`, put the `data/` folder there too, and install `matplotlib`.
