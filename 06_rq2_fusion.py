import json
import numpy as np
import pandas as pd
from common import PROC, cv_scores
import warnings
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

f = pd.read_csv(PROC / "features_t0.csv")
B = json.load(open(PROC / "blocks.json"))

def run(name, mask, blocks, model="logreg"):
    d = f[mask]
    cols = [c for b in blocks for c in B[b]]
    s = cv_scores(d[cols].to_numpy(float), d.target.to_numpy(), model)
    return dict(population=name, modalites=" + ".join(blocks), modele=model,
                n=len(d), converters=int(d.target.sum()),
                auroc=s["auroc_mean"], auroc_std=s["auroc_std"], auprc=s["auprc_mean"])

has_irm = f[B["irm"]].notna().any(axis=1)
rows = []
# population commune : patients avec IRM (443)
for bl in [["cognition"], ["irm"], ["cognition", "irm"],
           ["cognition", "clinique", "genetique", "irm"]]:
    rows.append(run("IRM (443)", has_irm, bl))
# toute la cohorte (774)
for bl in [["cognition"], ["cognition", "clinique", "genetique"]]:
    rows.append(run("Cohorte (774)", np.ones(len(f), bool), bl))

r = pd.DataFrame(rows)
print(r.to_string(index=False))
r.to_csv(PROC / "rq2_resultats.csv", index=False)