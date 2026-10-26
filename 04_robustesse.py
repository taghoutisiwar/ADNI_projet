import json

import numpy as np
import pandas as pd

from common import PROC, SEED, cv_scores
import warnings
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

feat = pd.read_csv(PROC / "features_t0.csv")

BLOCKS = json.load(open(PROC / "blocks.json"))

# modèle retenu : à choisir d'après rq1_resultats.csv
MODEL = "xgb"
CORE = ["cognition", "clinique", "genetique", "irm", "pet", "lcr", "plasma"]
RATES = [0.2, 0.4]
MASKABLE = ["irm", "pet", "lcr", "plasma"]

y = feat.target.to_numpy()
rng = np.random.default_rng(SEED)


def cols_of(blocks):
    return [c for b in blocks for c in BLOCKS[b]]


rows = []

# 1) ablation : on retire une modalité et on réentraîne
s = cv_scores(feat[cols_of(CORE)].to_numpy(float), y, MODEL)
rows.append({"scenario": "toutes les modalités", "type": "ablation", **s})

for removed in CORE:
    used = [b for b in CORE if b != removed]
    s = cv_scores(feat[cols_of(used)].to_numpy(float), y, MODEL)
    rows.append({"scenario": f"sans {removed}", "type": "ablation", **s})
    print("ablation, sans", removed, "-> AUROC", s["auroc_mean"], flush=True)

# 2) données manquantes simulées : une modalité masquée au hasard
for block in MASKABLE:
    for rate in RATES:
        X_df = feat[cols_of(CORE)].copy()
        hidden = rng.random(len(X_df)) < rate
        X_df.loc[hidden, BLOCKS[block]] = np.nan
        s = cv_scores(X_df.to_numpy(float), y, MODEL)
        rows.append({"scenario": f"{block} absent pour {int(rate * 100)} %",
                     "type": "donnees_manquantes", **s})
        print(f"masqué {block} {rate:.0%} -> AUROC", s["auroc_mean"], flush=True)

res = pd.DataFrame(rows)
res.to_csv(PROC / "robustesse_resultats.csv", index=False)
print("\n", res[["scenario", "type", "auroc_mean", "auroc_std",
                 "auprc_mean", "sens_mean", "spec_mean", "f1_mean"]].to_string(index=False))
