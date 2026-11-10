# Pokemon Battle Prediction (FDS 2026)

Our code for the Sapienza Fundamentals of Data Science Kaggle challenge. Given the first 30 turns of a Gen 2 OU battle, predict whether player 1 wins.

Final result: 0.686 on the private leaderboard (0.696 public), 13th out of 47 teams.

## What we did

Each battle becomes one row of features. Most of them describe the state at turn 30: HP left, how many Pokemon fainted, status conditions, boosts of the active Pokemon, type matchups, and how strong the remaining team is. We also added 1/0 columns for the 30 most common Pokemon (is it still alive?) and the features our teammate Enes wrote (the `f_` columns).

Lookup tables like species stats or which moves heal are built from the training data only. The only extra file is the type chart that came with the competition.

The final model averages two models: logistic regression (weight 2) and HistGradientBoosting (weight 1). Logistic regression gets a smaller feature set, because many columns are just copies of others (for example `diff_hp_total` is `p1_hp_total - p2_hp_total`).

We compared everything with 5-fold CV repeated 3 times. CV accuracy:

| Model | CV |
|---|---|
| Logistic regression | 0.684 |
| HistGradientBoosting | 0.677 |
| Ensemble 2:1 | 0.688 |

## Files

- `src/submission_best_features.py` - our features
- `src/submission_best_features_teammate.py` - Enes's features
- `src/validation.py` - the CV setup
- `experiments.ipynb` - jupyter notebook about experiments
- `make_best_submission.py` - trains both models and writes `submission.csv`

## Running it

Put `train.jsonl`, `test.jsonl` and `type_chart_gen2.csv` in a `data/` folder, then:

```
pip install -r requirements.txt
python make_best_submission.py
```

It takes a few minutes. You can also pass the data folder: `python make_best_submission.py /kaggle/input/<competition>`.
