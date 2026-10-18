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

Une interface Streamlit en huit pages, thème clair « clinique » : variables de couleur dans
`app/static/theme.css` (contrastes WCAG AA vérifiés), gabarit Plotly commun dans `app/figures.py`,
polices Space Grotesk, Manrope et JetBrains Mono servies depuis `app/static/fonts/` (licence OFL) :
l'interface fonctionne hors ligne. Le tracé EEG de la page Visualisation est un lecteur dessiné dans le
navigateur (`app/static/eeg_viewer.html`) : on s'y déplace à la souris ou au clavier sans recharger la page. Elle réutilise exactement les fonctions du notebook (`src/eeg_project.py`) : sur
les 4 enregistrements du test réservé, la forêt retrouve les mêmes 14 crises sur 17 et les
mêmes 2 fausses alarmes.

| Section | Page | Contenu |
|---|---|---|
| Comprendre | **Vue d'ensemble** | Ce qu'est une absence à côté d'un extrait réel, chiffres clés, parcours, catalogue avec mini-frises |
| | **Qu'est-ce qu'une absence ?** | Fond normal et crise à la même échelle (fréquence dominante et amplitude mesurées), carte des électrodes, durée des crises |
| | **Méthode** | La chaîne appliquée pas à pas à un enregistrement : filtrage, fenêtres et règle des 50 %, spectres, classement, post-traitement, appariement IoU |
| Explorer | **Visualisation** | Voies EEG avec curseur de plage, spectrogramme, carte du scalp (MNE) |
| | **Détection** | Forêt (F7) ou CNN (19 voies), résultat en une phrase, confrontation aux annotations (retards), saillance du CNN, export |
| | **Erreurs** | Chaque crise manquée et chaque fausse alarme du test, avec son extrait et un lien vers la visualisation |
| Évaluer | **Modèles** | 38 configurations et le CNN, matrice de confusion, ROC / précision-rappel, calibration, validation par patient |
| | **Limites** | Ce que les résultats permettent de dire, intervalles de confiance de Wilson, pistes |

```bash
app/lancer.sh            # Linux / macOS
app\lancer.bat           # Windows
```

puis ouvrir <http://127.0.0.1:8501>. Le premier lancement exporte le modèle choisi en
validation interne, les scores du test et contrôle les 22 exports contre leurs `.mat`
d'origine (`app/prepare_model.py`, environ une minute). Liens directs :
<http://127.0.0.1:8501/detection?id=230718A-A&modele=cnn>.

Chaque enregistrement porte son statut : *test réservé* (jamais vu par les modèles),
*apprentissage* (comparaison optimiste), *sans annotation* ou *exclu*. On peut aussi déposer
un fichier pour la forêt (`_d.mat`, `.npz` exporté ou `.csv` avec `Time` et `EEGF7`).
`200625A-F` n'a pas de canal F7 et ne peut pas être analysé.

### Deep learning (optionnel)

```bash
.venv/bin/python scripts/train_cnn.py    # environ 4 minutes sur CPU
```

Un CNN 1D (`src/eeg_deep.py`, environ 48 000 paramètres) apprend directement sur les 19 voies
filtrées, rééchantillonnées à 128 Hz et normalisées par enregistrement, avec une perte
pondérée et de l'augmentation (décalage, amplitude, bruit). Même séparation que la forêt :
3 plis groupés sur les 12 tracés d'apprentissage (arrêt précoce sur un tracé de
l'apprentissage), post-traitement réglé sur les scores hors pli, modèle final réentraîné sur
les 12 tracés, test lu une seule fois. Sorties dans `app/model/` (`cnn.pt`,
`cnn_metadata.json`, `cnn_test_scores.csv`) ; sans elles, l'interface n'affiche que la forêt.

Résultat sur le test réservé : AP des fenêtres 0,823 (forêt : 0,828), F1 brut au seuil 0,5
de 0,75, mais seulement 8 crises sur 17 après post-traitement. Le seuil réglé hors pli
(0,88) ne se transpose pas au modèle final réentraîné, dont les probabilités sont décalées :
c'est une limite du protocole, signalée telle quelle plutôt que corrigée après lecture du test.

### Docker

```bash
docker build -t absences-eeg .
docker run --rm -p 8501:8501 -v "$PWD/data:/app/data:ro" \
  -v "$PWD/artifacts:/app/artifacts:ro" -v "$PWD/app/model:/app/app/model" absences-eeg
```

## Organisation

```text
src/eeg_project.py          chargement, fenêtres, caractéristiques, modèles, post-traitement, évaluation
src/eeg_presentation.py     figures et rédaction du rapport à partir des résultats
src/eeg_source_validation.py  contrôle des exports contre les .mat d'origine
scripts/                    construction et exécution du notebook, entraînement autonome, train_cnn.py
src/eeg_deep.py             CNN 1D, préparation des 19 voies, Monte-Carlo dropout, saillance
app/                        interface Streamlit (streamlit_app.py, service.py, figures.py) et export du modèle
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
