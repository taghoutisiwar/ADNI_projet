import itertools
import json

import numpy as np
import pandas as pd

from common import PROC, cv_scores
import warnings
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

feat = pd.read_csv(PROC / "features_t0.csv")
BLOCKS = json.load(open(PROC / "blocks.json"))
MIN_N = 50
MODELS = ["logreg", "rf", "xgb"]


def subset_for(group):
    # patients ayant au moins une valeur dans chaque modalité du groupe
    mask = np.ones(len(feat), dtype=bool)
    for b in group:
        mask &= feat[BLOCKS[b]].notna().any(axis=1).to_numpy()
    return feat[mask]


groups = [(b,) for b in BLOCKS] + list(itertools.combinations(BLOCKS, 2))

rows = []
for group in groups:
    d = subset_for(group)
    if len(d) < MIN_N or d.target.nunique() < 2:
        print("ignoré (effectif trop faible) :", group)
        continue
    cols = [c for b in group for c in BLOCKS[b]]
    X = d[cols].to_numpy(dtype=float)
    y = d.target.to_numpy()
    label = " + ".join(group)
    for name in MODELS:
        print(f"{label:28s} {name}", flush=True)
        scores = cv_scores(X, y, name)
        rows.append({"modalites": label, "modele": name,
                     "n": len(d), "converters": int(y.sum()), **scores})

res = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
res.to_csv(PROC / "rq1_resultats.csv", index=False)

cols_show = ["modalites", "modele", "n", "converters", "auroc_mean", "auroc_std",
             "auprc_mean", "sens_mean", "spec_mean", "f1_mean"]
print("\nMeilleurs résultats (AUROC) :")
print(res[cols_show].head(15).to_string(index=False))

print("\nMeilleur modèle pour chaque modalité seule :")
singles = res[~res.modalites.str.contains(r"\+")]
print(singles.loc[singles.groupby("modalites").auroc_mean.idxmax(), cols_show].to_string(index=False))
