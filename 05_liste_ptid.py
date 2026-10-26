from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
RAW = Path("data/raw")

coh = pd.read_csv(PROC / "cohort_mci_24m.csv")

ros = pd.read_csv(next(RAW.glob("ROSTER*.csv")), low_memory=False)
ros.columns = [c.upper() for c in ros.columns]

liste = coh.merge(ros[["RID", "PTID"]].drop_duplicates("RID"), on="RID", how="left")
liste.to_csv(PROC / "liste_ptid_cohorte.csv", index=False)

print(len(liste), "patients dans la cohorte")
print(liste.PTID.isna().sum(), "RID sans PTID")