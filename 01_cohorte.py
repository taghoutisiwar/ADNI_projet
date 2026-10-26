import pandas as pd

from common import FU_MONTHS, FU_TOL, PROC, load_table

dx = load_table("DXSUM")[["RID", "VISCODE", "EXAMDATE", "DIAGNOSIS"]].copy()
dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx = dx.dropna(subset=["EXAMDATE", "DIAGNOSIS"])

flow = []


def add_step(label, remaining, excluded=0):
    flow.append({"etape": label, "exclus": excluded, "restants": remaining})
    print(f"{remaining:6d} restants | {excluded:5d} exclus | {label}")


add_step("Patients avec un diagnostic daté", dx.RID.nunique())

# une visite = un patient à une date donnée : suppression des doublons éventuels
print("Doublons RID + date :", dx.duplicated(["RID", "EXAMDATE"]).sum())
dx = dx.sort_values(["RID", "EXAMDATE"]).drop_duplicates(["RID", "EXAMDATE"])

# baseline = première visite diagnostiquée
first = dx.drop_duplicates("RID", keep="first")
n_cn = int((first.DIAGNOSIS == 1).sum())
n_ad = int((first.DIAGNOSIS == 3).sum())
mci = first[first.DIAGNOSIS == 2].rename(columns={"EXAMDATE": "T0"})[["RID", "T0"]]

add_step("Exclus : CN au baseline", len(first) - n_cn, n_cn)
add_step("Exclus : AD au baseline", len(mci), n_ad)
n_mci = len(mci)

# visites de suivi dans la fenêtre 18 à 30 mois
fu = dx.merge(mci, on="RID")
fu["mois"] = (fu.EXAMDATE - fu.T0).dt.days / 30.44
lo, hi = FU_MONTHS - FU_TOL, FU_MONTHS + FU_TOL
win = fu[fu.mois.between(lo, hi)].copy()
win["ecart"] = (win.mois - FU_MONTHS).abs()
fu24 = win.sort_values(["RID", "ecart"]).drop_duplicates("RID")  # visite la plus proche de 24 mois

add_step(f"Exclus : aucune visite entre {lo} et {hi} mois", len(fu24), n_mci - len(fu24))

# retour à CN : évolution ambiguë, exclu
n_revert = int((fu24.DIAGNOSIS == 1).sum())
fu24 = fu24[fu24.DIAGNOSIS != 1]
add_step("Exclus : revenu CN à 24 mois (évolution ambiguë)", len(fu24), n_revert)

cohort = fu24.assign(target=(fu24.DIAGNOSIS == 3).astype(int))
cohort = cohort.rename(columns={"EXAMDATE": "DATE_FU", "mois": "MOIS_FU"})
cohort = cohort[["RID", "T0", "DATE_FU", "MOIS_FU", "target"]]

cohort.to_csv(PROC / "cohort_mci_24m.csv", index=False)
pd.DataFrame(flow).to_csv(PROC / "flow_cohort.csv", index=False)

print("\nClasses finales :")
print(cohort.target.value_counts().rename({0: "stables", 1: "converters"}))
print(f"Taux de converters : {cohort.target.mean():.1%}")
