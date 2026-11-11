"""Bütün deneylerde aynı CV düzeni kullanılsın diye tek yerde tanımlı."""

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

N_SPLITS = 5
N_REPEATS = 3
SEED = 42


def make_cv():
    # 5-fold × 3 tekrar: tek 5-fold'un gürültüsü (±0.006) küçük farkları ayırt etmeye yetmiyor
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)


def cv_accuracy(model, X, y):
    """(ortalama, standart sapma) accuracy."""
    scores = cross_val_score(model, X, y, cv=make_cv(), scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std()
