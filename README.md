# Prédiction de la conversion MCI → Alzheimer (ADNI)

Pipeline reproductible : cohorte à 24 mois, comptage des modalités à T0,
RQ1 (modalités seules puis paires), robustesse (ablation et données manquantes).

## Installation

    pip install -r requirements.txt

## Données

Placer les fichiers CSV téléchargés depuis ADNI ARC Builder dans `data/raw/`.
Le script lit le premier fichier dont le nom commence par le préfixe demandé
(ex. `DXSUM*.csv`, `MMSE*.csv`), ce qui permet d'avoir une date dans le nom.

Fichiers attendus : DXSUM, MMSE, CDR, ADAS, PTDEMOG, APOERES, UCSFFSX7,
UCBERKELEY_AMY_6MM, UCBERKELEY_TAU_6MM, UCBERKELEYFDG_8mm,
UPENNBIOMK_ROCHE_ELECSYS, UPENN_PLASMA.

## Exécution (dans l'ordre)

    python 01_cohorte.py
    python 02_modalites.py
    python 03_rq1_unimodal.py
    python 04_robustesse.py

Sorties dans `data/processed/` : flux de cohorte, comptage des modalités,
résultats RQ1, robustesse.

## Structure

- `common.py` : chargement, T0, modèles, validation croisée
- `01_cohorte.py` : sélection MCI à T0, cible à 24 mois, tableau de flux
- `02_modalites.py` : variables à T0, comptage et intersections
- `03_rq1_unimodal.py` : modèles par modalité et par paire
- `04_robustesse.py` : ablation et données manquantes simulées

## Règles méthodologiques

- Aucune mesure postérieure à T0 n'est utilisée (fenêtre T0 − 180 jours à T0).
- L'imputation et la normalisation sont dans le pipeline, donc apprises uniquement sur le train de chaque pli.
- Validation croisée stratifiée répétée (5 plis × 3 répétitions).

## Limite connue

Les modalités n'ont pas le même nombre de patients. La comparaison RQ1 n'est donc
pas à effectif constant ; `modalites_combinaisons.csv` permet de refaire la
comparaison sur une sous-cohorte commune.

## Données

Ne pas redistribuer les données ADNI. Citer ADNI selon ses conditions d'accès.

## Notes de version (v2)

- `02_modalites.py` : MMSE, CDR et ADAS lisent la date dans `VISDATE` (et non `EXAMDATE`), sinon le script s'arrête.
- Les données ADNI ne sont pas incluses dans cette archive. Placez-les dans `data/raw/` (dossier non versionné).
