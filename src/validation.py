"""The cross-validation setup we used for every experiment.

It lives in one place so all experiments split the data the same way,
otherwise two scores wouldn't be comparable.
"""

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def make_cv():
    # 5-fold: train on 80% of train.jsonl, test on the other 20%, five times.
    # "Stratified" keeps the win/loss ratio the same in every fold.
    # One 5-fold run moved by about +-0.006 just from the random split, which is
    # bigger than the differences we wanted to compare, so we repeat it 3 times
    # (15 scores) with a fixed seed.
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)


def cv_accuracy(model, X, y):
    """Mean and standard deviation of accuracy over the 15 folds."""
    scores = cross_val_score(model, X, y, cv=make_cv(), scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std()
