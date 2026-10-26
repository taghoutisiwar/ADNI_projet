import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

SEED = 42
FU_MONTHS = 24          # horizon de prédiction
FU_TOL = 6              # fenêtre de suivi : 18 à 30 mois
T0_LOOKBACK_DAYS = 180  # une mesure doit être faite entre T0-180 j et T0, jamais après


def load_table(prefix):
    files = sorted(glob.glob(str(RAW / f"{prefix}*.csv")))
    if not files:
        raise FileNotFoundError(f"Aucun fichier {prefix}*.csv dans {RAW}")
    print(f"  {prefix} <- {Path(files[-1]).name}")
    df = pd.read_csv(files[-1], low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df


def pick_date(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Aucune colonne date parmi {candidates}")


def to_number(s):
    # certaines valeurs LCR arrivent sous la forme "<8" ou ">1700"
    return pd.to_numeric(s.astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce")


def value_at_t0(df, date_col, t0, cols):
    """Mesure la plus proche de T0, jamais postérieure à T0."""
    d = df[["RID", date_col] + cols].dropna(subset=[date_col]).merge(t0, on="RID")
    d["gap"] = (d["T0"] - d[date_col]).dt.days
    d = d[(d.gap >= 0) & (d.gap <= T0_LOOKBACK_DAYS)]
    d = d.sort_values(["RID", "gap"]).drop_duplicates("RID")
    return d[["RID"] + cols]


def t0_block(prefix, date_candidates, cols, t0, rename=None, row_filter=None):
    df = load_table(prefix)
    if row_filter is not None:
        df = df[row_filter(df)].copy()
    date_col = pick_date(df, date_candidates)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for c in cols:
        df[c] = to_number(df[c])
    out = value_at_t0(df, date_col, t0, cols)
    return out.rename(columns=rename or {})


def make_model(name, y_train):
    # l'imputation est dans le pipeline : elle n'apprend que sur le train de chaque pli
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
        )
    if name == "xgb":
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=ratio, random_state=SEED,
                          eval_metric="logloss"),
        )
    raise ValueError(name)


def cv_scores(X, y, model_name, n_splits=5, n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    keys = ["auroc", "auprc", "sens", "spec", "f1"]
    res = {k: [] for k in keys}
    for tr, te in cv.split(X, y):
        model = make_model(model_name, y[tr])
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        tn, fp, fn, tp = confusion_matrix(y[te], (p >= 0.5).astype(int), labels=[0, 1]).ravel()
        res["auroc"].append(roc_auc_score(y[te], p))
        res["auprc"].append(average_precision_score(y[te], p))
        res["sens"].append(tp / (tp + fn) if (tp + fn) else np.nan)
        res["spec"].append(tn / (tn + fp) if (tn + fp) else np.nan)
        res["f1"].append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    out = {}
    for k in keys:
        out[f"{k}_mean"] = round(float(np.nanmean(res[k])), 3)
        out[f"{k}_std"] = round(float(np.nanstd(res[k])), 3)
    return out
