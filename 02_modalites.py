import json

import pandas as pd

from common import PROC, load_table, t0_block

coh = pd.read_csv(PROC / "cohort_mci_24m.csv", parse_dates=["T0"])
t0 = coh[["RID", "T0"]]
feat = coh[["RID", "target"]].copy()

# cognition : valeur la plus proche de T0 (date de visite VISDATE)
mmse = t0_block("MMSE", ["VISDATE"], ["MMSCORE"], t0, rename={"MMSCORE": "mmse"})
cdr = t0_block("CDR", ["VISDATE"], ["CDRSB"], t0, rename={"CDRSB": "cdrsb"})
adas = t0_block("ADAS", ["VISDATE"], ["TOTAL13"], t0, rename={"TOTAL13": "adas13"})

# clinique et démographie : âge calculé à partir de l'année de naissance
pt = load_table("PTDEMOG")[["RID", "PTDOBYY", "PTGENDER", "PTEDUCAT"]].drop_duplicates("RID")
pt["birth_year"] = pd.to_datetime(pt["PTDOBYY"], errors="coerce").dt.year
clin = t0.merge(pt[["RID", "birth_year", "PTGENDER", "PTEDUCAT"]], on="RID", how="left")
clin["age"] = clin["T0"].dt.year - clin["birth_year"]
clin["sex_female"] = clin["PTGENDER"].map({1: 0, 2: 1})   # -4 (manquant) devient NaN
clin["educ"] = pd.to_numeric(clin["PTEDUCAT"], errors="coerce").where(lambda s: s >= 0)
clin = clin[["RID", "age", "sex_female", "educ"]]

# génétique : nombre d'allèles epsilon4
apoe = load_table("APOERES")[["RID", "GENOTYPE"]].drop_duplicates("RID")
apoe["apoe4"] = apoe.GENOTYPE.astype(str).str.count("4").where(apoe.GENOTYPE.notna())
apoe = apoe[["RID", "apoe4"]]

# IRM (FreeSurfer) : date d'examen EXAMDATE
mri = t0_block("UCSFFSX7", ["EXAMDATE"],
               ["ST29SV", "ST88SV", "ST24TA", "ST83TA", "ST10CV"], t0)
mri["hippo_vol"] = mri.ST29SV + mri.ST88SV
mri["hippo_norm"] = mri.hippo_vol / mri.ST10CV
mri["entorhinal"] = (mri.ST24TA + mri.ST83TA) / 2
mri["icv"] = mri.ST10CV
mri = mri[["RID", "hippo_vol", "hippo_norm", "entorhinal", "icv"]]

# PET : amyloïde (SCANDATE) et FDG (EXAMDATE)
# le tau est exclu : les scans sont postérieurs à T0 pour la cohorte
amy = t0_block("UCBERKELEY_AMY", ["SCANDATE"], ["CENTILOIDS"], t0,
               rename={"CENTILOIDS": "amy_cl"})
fdg = t0_block("UCBERKELEYFDG", ["EXAMDATE"], ["MEAN"], t0,
               rename={"MEAN": "fdg_suvr"},
               row_filter=lambda d: d["ROINAME"] == "MetaROI")
pet = amy.merge(fdg, on="RID", how="outer")

# biomarqueurs LCR (Roche Elecsys) et plasma : séparés (unités différentes)
lcr = t0_block("UPENNBIOMK_ROCHE", ["EXAMDATE"], ["ABETA42", "TAU", "PTAU"], t0,
               rename={"ABETA42": "csf_abeta42", "TAU": "csf_tau", "PTAU": "csf_ptau"})
plasma = t0_block("UPENN_PLASMA", ["EXAMDATE"], ["PT217_F", "AB42_AB40_F"], t0,
                  rename={"PT217_F": "pla_ptau217", "AB42_AB40_F": "pla_ab_ratio"})

for part in [mmse, cdr, adas, clin, apoe, mri, pet, lcr, plasma]:
    feat = feat.merge(part, on="RID", how="left")

BLOCKS = {
    "cognition": ["mmse", "cdrsb", "adas13"],
    "clinique": ["age", "sex_female", "educ"],
    "genetique": ["apoe4"],
    "irm": ["hippo_vol", "hippo_norm", "entorhinal", "icv"],
    "pet": ["amy_cl", "fdg_suvr"],
    "lcr": ["csf_abeta42", "csf_tau", "csf_ptau"],
    "plasma": ["pla_ptau217", "pla_ab_ratio"],
}

feat.to_csv(PROC / "features_t0.csv", index=False)
with open(PROC / "blocks.json", "w") as f:
    json.dump(BLOCKS, f, indent=2)

# comptage des modalités à T0
has = pd.DataFrame({b: feat[cols].notna().any(axis=1) for b, cols in BLOCKS.items()})
has["target"] = feat["target"].to_numpy()
n_total = len(has)

resume = []
for b in BLOCKS:
    m = has[b]
    resume.append({
        "modalite": b,
        "patients": int(m.sum()),
        "pct_cohorte": round(100 * m.mean(), 1),
        "converters": int(has.loc[m, "target"].sum()),
        "stables": int((m & (has.target == 0)).sum()),
    })
resume = pd.DataFrame(resume)
print(resume.to_string(index=False))

inter = pd.DataFrame(0, index=list(BLOCKS), columns=list(BLOCKS))
for a in BLOCKS:
    for b in BLOCKS:
        inter.loc[a, b] = int((has[a] & has[b]).sum())
print("\nIntersections deux à deux :")
print(inter.to_string())

combo = has.groupby(list(BLOCKS)).size().reset_index(name="patients")
combo["modalites"] = combo[list(BLOCKS)].apply(
    lambda r: " + ".join(b for b in BLOCKS if r[b]), axis=1)
combo = combo[combo.modalites != ""][["modalites", "patients"]]
combo = combo.sort_values("patients", ascending=False)
print("\nCombinaisons exactes (top 15) :")
print(combo.head(15).to_string(index=False))

all_blocks = int(has[list(BLOCKS)].all(axis=1).sum())
print(f"\nPatients avec toutes les modalités : {all_blocks} / {n_total}")

resume.to_csv(PROC / "modalites_resume.csv", index=False)
inter.to_csv(PROC / "modalites_intersections.csv")
combo.to_csv(PROC / "modalites_combinaisons.csv", index=False)
