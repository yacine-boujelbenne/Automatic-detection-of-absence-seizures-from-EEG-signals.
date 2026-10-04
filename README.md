# Détection des crises d'absence EEG

Le livrable à présenter est **`notebooks/mini_projet_eeg_final.ipynb`**, entièrement
exécuté en français. **`notebooks/mini_projet_eeg_final.html`** permet de consulter
les mêmes résultats et figures sans lancer Python. Le rapport final est
**`reports/RAPPORT_FINAL_EEG.md`**.

Le notebook suit les douze sections du projet, utilise `Annotations MAJ.xlsx` et
les horaires relatifs à `DebutTrace`, et compare DT, RF, KNN et SVM sous quatre
stratégies : déséquilibre initial, sous-échantillonnage, duplication et SMOTE.
Le test reste déséquilibré et les enregistrements ne traversent pas les partitions.

La cohorte comprend 16 enregistrements et 65 crises après exclusions conservatrices
des doublons conflictuels et de l'annotation dépassant le signal. Les échantillons,
les temps et les noms de canaux des exports ont été comparés aux cellules
numériques des fichiers MATLAB sources. Les contrôles et exclusions sont visibles
dans le notebook et dans `artifacts/final/`.

## Organisation

```text
notebooks/       Notebook final exécuté et présentation HTML
reports/         Rapport final français
src/             Chargement, validation, modèles et figures
scripts/         Construction du notebook, exécution et entraînement
tests/           Vérifications du pipeline final
tools/           Lecteur des structures MATLAB sources
data/            Fichiers MATLAB originaux et exports numériques
artifacts/final/ Modèles entraînés, tableaux, contrôles et figures
```

À la racine : `Annotations MAJ.xlsx`, le PDF de l'énoncé, ce guide et
`requirements.txt`. Les dossiers cachés de consignes et les tests de référence
sont conservés. `venv/` utilise le Python local de `.runtime/python/` ; ces deux
dossiers sont nécessaires à l'exécution sur cette machine.

## Exécution

Depuis la racine, avec le Python du projet :

```powershell
venv/Scripts/python.exe scripts/execute_final_notebook.py
```

Ce script choisit explicitement l'interpréteur du projet, démarre un noyau propre,
exécute toutes les cellules et exporte le HTML uniquement après réussite.
Les fichiers temporaires du noyau sont automatiquement supprimés. Pour
ouvrir le notebook dans Jupyter ou VS Code, sélectionner `venv/Scripts/python.exe`
comme interpréteur. `RETRAIN = True`, dans la première cellule, force les 16
entraînements. Sinon, ils sont relus seulement si les empreintes des données, du
code, des partitions et des versions correspondent.

Le calcul autonome de la comparaison est également disponible :

```powershell
venv/Scripts/python.exe scripts/train_project.py
```

La construction de la mise en page, avant exécution, utilise :

```powershell
venv/Scripts/python.exe scripts/build_final_notebook.py
```

Cette dernière commande reconstruit les cellules et efface les sorties du
notebook. Il faut ensuite relancer le script d'exécution pour retrouver les
résultats intégrés.

## Résultats conservés

`artifacts/final/` contient le modèle sélectionné en validation interne
(`best_model.joblib`), les 16 configurations entraînées, les métriques des trois
plis internes, les scores du test commun, les prédictions, les intervalles et les
figures. Le choix du modèle ne change pas d'après le classement du test. Les
fausses alertes, les crises manquées et les limites de la petite cohorte sont
rapportées explicitement.

## Vérification

```powershell
venv/Scripts/python.exe -m ruff check .
venv/Scripts/python.exe -m ruff format . --check
venv/Scripts/python.exe -m pytest tests/test_eeg_project.py -q -p no:cacheprovider
venv/Scripts/python.exe .claude/skills/eeg-pipeline/reference/test_reference.py
```

Les dépendances exactes nécessaires au notebook sont conservées dans
`requirements.txt`. Les données sources ne sont ni modifiées ni remplacées
par des signaux simulés. L'expérience reste pédagogique et n'établit pas une
validation clinique externe.

## Nettoyage effectué

Les anciennes versions des notebooks et du rapport, les scripts abandonnés,
les anciennes figures et les caches ont été retirés. L'archive ZIP redondante
a été supprimée après comparaison SHA-256 de ses 45 fichiers utiles avec les
sources extraites conservées. Les modèles, résultats et données de tous les
enregistrements sont conservés, y compris ceux exclus de la cohorte.
La liste des suppressions et les empreintes des sources figurent dans
`artifacts/final/cleanup_manifest.json`.
