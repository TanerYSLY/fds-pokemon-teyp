import time, numpy as np, pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.features import *
from src.features_teammate import build_pokedex, load_setup_moves
from src.validation import make_cv
tr = pd.read_json("data/train.jsonl", lines=True)
sp, ch, hl, vocab = load_species([tr]), load_type_chart("data/type_chart_gen2.csv"), load_healing_moves([tr]), load_identity_vocab([tr])
X4 = build_features(tr, sp, ch, hl, vocab)
y = tr.set_index("battle_id").loc[X4.index, "player_won"]
# 05_deneyler'deki aramanın aynısı: yüklenen modelin yuvarlanmamış ayarları
space = {"learning_rate": loguniform(0.01, 0.2), "max_iter": randint(100, 600), "max_leaf_nodes": randint(4, 40),
         "min_samples_leaf": randint(20, 300), "l2_regularization": loguniform(1e-3, 10)}
s = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=42), space, n_iter=25,
                       cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy", random_state=42, n_jobs=-1).fit(select_features(X4, "tree"), y)
params = s.best_params_; print("ayarlar:", params, flush=True)
X5 = build_features(tr, sp, ch, hl, vocab, {"pokedex": build_pokedex(tr), "setup_moves": load_setup_moves([tr])})
XL, XT = select_features(X5, "linear"), select_features(X5, "tree")
def oof(est, Xs):
    t = time.time()
    r = [(va, clone(est).fit(Xs.iloc[a], y.iloc[a]).predict_proba(Xs.iloc[va])[:, 1]) for a, va in make_cv().split(Xs, y)]
    print(f"  bitti ({time.time() - t:.0f} sn)", flush=True); return r
H = lambda n: HistGradientBoostingClassifier(**{**params, "max_iter": n}, random_state=42)
Bag = lambda n: BaggingClassifier(H(n), n_estimators=10, bootstrap=True, random_state=42, n_jobs=-1)
LR = oof(make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=5000)), XL)
variants = {f"HistGB {params['max_iter']} ağaç (yüklenen)": H(params["max_iter"]), "HistGB 300 ağaç": H(300), "HistGB 200 ağaç": H(200),
            "Bagging 10 x 300 ağaç": Bag(300), f"Bagging 10 x {params['max_iter']} ağaç": Bag(params["max_iter"])}
P = {}
for k, e in variants.items():
    print(k, flush=True); P[k] = oof(e, XT)
ac = lambda p, k: ((p > 0.5) == y.values[LR[k][0]]).mean()
def sc(name, wl=2, wh=1):
    return np.array([ac((wl * LR[k][1] + wh * P[name][k][1]) / (wl + wh), k) for k in range(15)])
ref_name = next(iter(P)); ref = sc(ref_name)
rows = {}
for k in P:
    alone = np.array([ac(P[k][i][1], i) for i in range(15)])
    d = sc(k) - ref; e = sc(k, 1, 1)
    rows[k] = {"HistGB tek": alone.mean(), "ens 2:1": sc(k).mean(), "fark": d.mean(), "kazanç": f"{(d>0).sum()}/{(d<0).sum()}",
               "z": d.mean() / (d.std(ddof=1) / np.sqrt(15)) if d.std() else 0, "ens 1:1": e.mean()}
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).T.to_string())
