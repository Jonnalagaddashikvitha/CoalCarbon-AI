#!/usr/bin/env python3
"""CoalCarbon AI - ML pipeline (run: python coalcarbon_ml.py  -> writes ml_results.json)

PRIMARY   : state-level coal production forecasting (REAL data, Ministry of Coal / PIB)
SUPPORTING: emission-intensity regression, high-emitter classification, K-Means
            archetypes (all on a SYNTHETIC mine-level dataset - real mine data is not public).
"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

SEED = 42
r3 = lambda x: round(float(x), 3)

# ---------------------------------------------------------------- 1. PRIMARY: FORECASTING
def load_production(path="coal_production_state.csv"):
    wide = pd.read_csv(path)                                           # data collection
    df = wide.melt(id_vars="state", var_name="fy", value_name="mt")
    df["year"] = df.fy.str[2:6].astype(int)                            # FY2019-20 -> 2019
    return df.sort_values(["state", "year"]).reset_index(drop=True)

def engineer(df, states):
    df = df.copy()
    df["lag1"] = df.groupby("state").mt.shift(1)                       # previous-year production
    df["year_idx"] = df.year - df.year.min()                           # time index
    df["log_growth"] = np.log(df.mt / df.lag1)                         # TARGET handed to models
    for s in states:
        df["st_" + s] = (df.state == s).astype(int)                    # one-hot state
    return df

def make_models():
    return {"Random Forest": lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED),
            "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                                   learning_rate=.05, random_state=SEED),
            "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))}

def metrics(y, p):
    return dict(MAE=r3(mean_absolute_error(y, p)), RMSE=r3(np.sqrt(mean_squared_error(y, p))),
                MAPE=r3(np.mean(np.abs((y - p) / y)) * 100), R2=r3(r2_score(y, p)))

def forecasting():
    raw = load_production(); states = sorted(raw.state.unique())
    d = engineer(raw, states)
    X = ["year_idx", "lag1"] + ["st_" + s for s in states]
    last = int(d.year.max())
    # Rolling-origin validation: train only on years BEFORE the test year (no shuffling, no leakage)
    folds, preds = [last - 1, last], []
    for ty in folds:
        tr = d[(d.year < ty) & d.lag1.notna()]; te = d[d.year == ty]
        for name, mk in make_models().items():
            p = te.lag1 * np.exp(mk().fit(tr[X], tr.log_growth).predict(te[X]))
            preds += [dict(model=name, state=s, fy=f, year=int(y), actual=r3(a), pred=r3(q))
                      for s, f, y, a, q in zip(te.state, te.fy, te.year, te.mt, p)]
        preds += [dict(model="Persistence (naive)", state=s, fy=f, year=int(y), actual=r3(a), pred=r3(q))
                  for s, f, y, a, q in zip(te.state, te.fy, te.year, te.mt, te.lag1)]
    P = pd.DataFrame(preds); table = []
    for name, g in P.groupby("model"):
        m = metrics(g.actual.values, g.pred.values)
        m["model"] = name; m["n"] = len(g)
        m["per_fold_MAPE"] = {int(y): r3(np.mean(abs(x.actual - x.pred) / x.actual) * 100) for y, x in g.groupby("year")}
        m["log_err_sd"] = r3(np.log(g.actual / g.pred).std(ddof=1)); table.append(m)
    table.sort(key=lambda r: r["MAPE"])
    best_ml = next(r["model"] for r in table if r["model"] != "Persistence (naive)")
    # Final model: refit best ML model on ALL rows with a lag, recursive multi-step forecast
    full = d[d.lag1.notna()]; mdl = make_models()[best_ml]().fit(full[X], full.log_growth)
    sd = next(r["log_err_sd"] for r in table if r["model"] == best_ml); z = 1.2816  # ~80% band
    future = {}
    for s in states:
        prev, out = float(d[(d.state == s) & (d.year == last)].mt.iloc[0]), []
        for h, y in enumerate(range(last + 1, last + 6), 1):
            row = pd.DataFrame([[y - d.year.min(), prev] + [int(s == t) for t in states]], columns=X)
            nxt = prev * float(np.exp(mdl.predict(row)[0]))
            out.append([r3(nxt), r3(nxt * np.exp(-z * sd * h ** .5)), r3(nxt * np.exp(z * sd * h ** .5))]); prev = nxt
        future[s] = out
    imp = None
    if hasattr(mdl, "feature_importances_"):
        raw_imp = dict(zip(X, mdl.feature_importances_)); st = sum(v for k, v in raw_imp.items() if k.startswith("st_"))
        imp = {"State (one-hot, grouped)": r3(st), "Previous-year production (lag1)": r3(raw_imp["lag1"]),
               "Year index": r3(raw_imp["year_idx"])}
    return dict(best_model=best_ml, table=table, predictions=preds, future=future, future_start=last + 1,
                importance=imp, importance_method="Random Forest impurity importance (final model, all data)",
                dataset=dict(name="State-wise raw coal production (Mt)", source="Ministry of Coal / PIB (PIB2042652)",
                             records=len(raw), states=states, years=f"FY{raw.year.min()}-{str(raw.year.min()+1)[2:]} to FY{last}-{str(last+1)[2:]}",
                             features=X, n_features=len(X), target="log growth = ln(production_t / production_t-1) -> converted back to Mt",
                             trainable_rows=len(full),
                             folds=[dict(test_fy=f"FY{y}-{str(y+1)[2:]}", train_rows=int(((d.year < y) & d.lag1.notna()).sum()),
                                         test_rows=int((d.year == y).sum())) for y in folds]))

# ---------------------------------------------------------------- 2. SUPPORTING: SYNTHETIC MINES
def synthetic_mines(n=800):
    """SYNTHETIC. Intensity = diesel*2.68 + electricity*0.675 + 20 (CH4) + noise. Not real mines."""
    r = np.random.default_rng(SEED)
    m = pd.DataFrame(dict(haul_km=r.uniform(2, 25, n), strip_ratio=r.uniform(1, 6, n),
                          electrified_share=r.uniform(0, .6, n), fleet_age=r.uniform(2, 20, n)))
    diesel_l_t = (1 + .35 * m.strip_ratio + .06 * m.haul_km + .02 * m.fleet_age) * (1 - .5 * m.electrified_share)
    kwh_t = 12 + 25 * m.electrified_share + r.normal(0, 1.5, n)
    m["intensity"] = diesel_l_t * 2.68 + kwh_t * .675 + 20 + r.normal(0, 2, n)
    return m

def supporting():
    m = synthetic_mines(); F = ["haul_km", "strip_ratio", "electrified_share", "fleet_age"]
    Xtr, Xte, ytr, yte = train_test_split(m[F], m.intensity, test_size=.2, random_state=SEED)
    regs = {"Linear Regression": LinearRegression(), "Ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
            "Random Forest": RandomForestRegressor(300, random_state=SEED),
            "Gradient Boosting": GradientBoostingRegressor(random_state=SEED)}
    reg, fitted = [], {}
    for k, mdl in regs.items():
        cv = cross_val_score(mdl, Xtr, ytr, cv=5, scoring="r2").mean(); mdl.fit(Xtr, ytr); fitted[k] = mdl
        p = mdl.predict(Xte); reg.append(dict(model=k, CV_R2=r3(cv), **{("Test_" + a): b for a, b in metrics(yte.values, p).items()}))
    rf = fitted["Random Forest"]; pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=SEED)
    sample = [dict(**{f: round(float(Xte.iloc[i][f]), 2) for f in F}, actual=r3(yte.iloc[i]), predicted=r3(rf.predict(Xte.iloc[[i]])[0])) for i in range(6)]
    thr = float(m.intensity.quantile(2 / 3)); lab = (m.intensity > thr).astype(int)
    Ctr, Cte, ltr, lte = train_test_split(m[F], lab, test_size=.2, random_state=SEED, stratify=lab)
    clfs = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
            "Random Forest Classifier": RandomForestClassifier(300, random_state=SEED)}
    cls, cms = [], {}
    for k, c in clfs.items():
        p = c.fit(Ctr, ltr).predict(Cte); cms[k] = confusion_matrix(lte, p).tolist()
        cls.append(dict(model=k, Accuracy=r3(accuracy_score(lte, p)), Precision=r3(precision_score(lte, p)),
                        Recall=r3(recall_score(lte, p)), F1=r3(f1_score(lte, p))))
    bestc = max(cls, key=lambda r: r["F1"])["model"]
    Z = StandardScaler().fit_transform(m[F]); sil = {}
    for k in range(2, 7): sil[k] = r3(silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)))
    K = max(sil, key=sil.get); km = KMeans(K, n_init=10, random_state=SEED).fit(Z); m["cl"] = km.labels_
    rules = [("haul_km", 1, "Rail & conveyor logistics"), ("strip_ratio", 1, "Fuel optimisation, in-pit crushing & conveying"),
             ("electrified_share", -1, "Equipment electrification + renewable power"), ("fleet_age", 1, "Fleet renewal & fuel optimisation")]
    prof = []
    for c in range(K):
        g = m[m.cl == c]; zs = [(s * (g[f].mean() - m[f].mean()) / m[f].std(), lv) for f, s, lv in rules]
        top = max(zs); lever = top[1] if top[0] > .3 else "Near-median profile: renewable electricity & efficiency"
        prof.append(dict(cluster=c + 1, n=len(g), haul=r3(g.haul_km.mean()), strip=r3(g.strip_ratio.mean()),
                         elec=r3(g.electrified_share.mean()), age=r3(g.fleet_age.mean()), intensity=r3(g.intensity.mean()), lever=lever))
    pc = PCA(2, random_state=SEED).fit(Z); xy = pc.transform(Z)[::3]
    return dict(n=len(m), features=F, target="intensity (kg CO2e / t coal)", split="80/20 hold-out + 5-fold CV on train",
                regression=reg, importance={F[i]: r3(pi.importances_mean[i]) for i in np.argsort(-pi.importances_mean)},
                samples=sample, high_threshold=r3(thr), classification=cls, best_clf=bestc, confusion=cms[bestc],
                clusters=dict(k=K, silhouette=sil, profiles=prof, pca_var=[r3(v) for v in pc.explained_variance_ratio_],
                              points=[[r3(a), r3(b), int(c)] for (a, b), c in zip(xy, m.cl.values[::3])]))

if __name__ == "__main__":
    out = dict(forecast=forecasting(), supporting=supporting())
    json.dump(out, open("ml_results.json", "w"), indent=1)
    for r in out["forecast"]["table"]: print({k: r[k] for k in ("model", "MAE", "RMSE", "MAPE", "R2")})
    print("best:", out["forecast"]["best_model"])
