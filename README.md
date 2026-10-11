# Détection des crises d'absence EEG

Mini-projet de data mining : classer des fenêtres EEG de 2 s en « crise » ou
« hors crise », puis reconstruire les épisodes de crise d'absence sur des
enregistrements jamais vus.

| Livrable | Fichier |
|---|---|
| Notebook exécuté (12 sections, Q1–Q14) | `notebooks/mini_projet_eeg_final.ipynb` (+ `.html` lisible sans Python) |
| Rapport | `reports/RAPPORT_FINAL_EEG.md` (figures dans `reports/figures/`) |
| Application de lecture | `app/` — voir plus bas |

## En bref

- 16 enregistrements, 65 crises, 23 444 fenêtres (4 % positives), canal F7 à 256 Hz.
- Séparation **par enregistrement** : 12 tracés d'apprentissage, 4 tracés de test réservé.
- 38 configurations : 4 modèles × 5 stratégies de déséquilibre × 2 jeux de caractéristiques
  (les 12 de l'énoncé, et un jeu enrichi : passe-haut 0,5 Hz, log-puissances, puissances
  relatives, bande des pointes-ondes, normalisation robuste par tracé).
- Sélection par **AP moyenne en validation interne** ; post-traitement (lissage, seuil,
  durée minimale) réglé sur des prédictions **hors pli** ; test lu une seule fois.
- Validation « un enregistrement exclu » sur les 16 tracés pour mesurer la variabilité.

Résultat retenu : forêt aléatoire, caractéristiques enrichies. Sur le test réservé,
14 crises retrouvées sur 17 et 1,5 fausse alarme par heure, contre 134 par heure
pour le modèle choisi par l'ancienne procédure. En validation « un enregistrement
exclu » : 85 % des crises retrouvées, 4,1 fausses alarmes par heure.

## Installation

Python 3.12 ou plus récent.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt        # Windows : .venv\Scripts\python.exe
```

## Données

Les signaux ne sont pas versionnés (766 Mo). Placer le dossier fourni par
l'enseignant, `CRISES ABSENCES/` (44 fichiers `.mat` et le classeur initial), dans
`data/`, puis extraire les signaux :

```bash
.venv/bin/python tools/mcos_extract.py --export "data/CRISES ABSENCES" data/exports
```

Les `.mat` contiennent des objets MATLAB `timetable` que `scipy.io` ne lit pas ;
`tools/mcos_extract.py` les décode. Chaque export est ensuite comparé valeur par
valeur au fichier d'origine par `src/eeg_source_validation.py`. Le classeur utilisé
est `Annotations MAJ.xlsx`, à la racine.

## Exécution

```bash
.venv/bin/python scripts/build_final_notebook.py     # régénère les cellules (sans sorties)
.venv/bin/python scripts/execute_final_notebook.py   # exécute, écrit le notebook, le HTML et le rapport
```

Le premier passage entraîne les 38 configurations et la validation « un
enregistrement exclu » (environ 12 minutes sur 8 cœurs). Les résultats sont ensuite
relus d'un cache, invalidé dès que le code, les données, les partitions ou les
versions des bibliothèques changent. `RETRAIN = True`, dans la première cellule,
force le recalcul. `scripts/train_project.py` lance la même chaîne sans notebook.

## Application

Un lecteur local : on dépose un enregistrement (`_d.mat`, `.npz` exporté ou `.csv`
avec `Time` et `EEGF7`), et l'application affiche le tracé sur une grille de lecture,
le score de chaque fenêtre, les épisodes détectés et, pour un enregistrement de la
cohorte, les crises annotées.

```bash
app/lancer.sh            # Linux / macOS
app\lancer.bat           # Windows
```

puis ouvrir <http://127.0.0.1:8765>. Le premier lancement exporte le modèle choisi en
validation interne et contrôle les 22 exports contre leurs `.mat` d'origine
(`app/prepare_model.py`, environ une minute).

Aucun fichier n'est nécessaire pour les enregistrements du jeu de données : on tape leur
identifiant (`211104B-D`) ou on clique une ligne de la liste. Chaque enregistrement porte
son statut : *test* (jamais vu par le modèle), *apprentissage* (comparaison optimiste),
*sans annotation* ou *exclu*. Un lien direct ouvre un enregistrement :
<http://127.0.0.1:8765/?id=230718A-A>. `200625A-F` n'a pas de canal F7 et ne peut pas
être analysé. Navigation au clavier : `←` `→` pour se déplacer, `[` `]` pour
passer d'un épisode à l'autre, `+` `−` pour le gain.

## Organisation

```text
src/eeg_project.py          chargement, fenêtres, caractéristiques, modèles, post-traitement, évaluation
src/eeg_presentation.py     figures et rédaction du rapport à partir des résultats
src/eeg_source_validation.py  contrôle des exports contre les .mat d'origine
scripts/                    construction et exécution du notebook, entraînement autonome
app/                        serveur local et interface
tools/mcos_extract.py       lecteur des timetables MATLAB
tests/                      tests unitaires et contrôle de séparation sur les vraies données
```

## Vérification

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest tests -q -p no:cacheprovider
```

## Limites

16 enregistrements sans identifiant patient fiable, un seul canal, annotations
sans double relecture. Outil pédagogique : les épisodes signalés ne constituent pas
un diagnostic.
