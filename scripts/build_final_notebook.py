"""Construit le notebook de présentation, sans résultats rédigés à la main."""

import ast
import textwrap
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build():
    """Génère les douze sections, leurs cellules exécutables et les réponses françaises."""
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # Détection automatique des crises d'absence à partir de l'EEG
    **Mini-projet de data mining — notebook final exécuté**

    Nous classons des fenêtres EEG de 2 s en « crise » ou « hors crise », comparons
    quatre classifieurs (arbre de décision, forêt aléatoire, KNN, SVM) et cinq façons
    de traiter le déséquilibre, puis reconstruisons les épisodes de crise.

    Deux jeux de caractéristiques sont évalués côte à côte : les **12 caractéristiques
    de l'énoncé**, et un jeu **enrichi** motivé par l'analyse exploratoire (section 6).
    La sélection se fait uniquement en validation interne ; le test réservé n'est lu
    qu'une fois, à la fin.

    | Sections | Contenu |
    |---|---|
    | 1–4 | données, contrôles, visualisation, annotations (Q1–Q4) |
    | 5–6 | fenêtres, étiquettes, caractéristiques (Q5–Q11) |
    | 7–9 | séparation, entraînement, évaluation, comparaison (Q12–Q13) |
    | 10–11 | reconstruction des épisodes et analyse des erreurs (Q14) |
    | 12 | améliorations mesurées et validation « un enregistrement exclu » |
    """)

    md("""
    ## 1. Chargement des données
    Les fonctions de calcul sont dans `src/eeg_project.py` ; leur code est affiché
    dans le notebook là où il sert. Les entraînements sont mis en cache avec une
    empreinte du code, des données, des partitions et des versions : `RETRAIN = True`
    force leur recalcul.
    """)
    code("""
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from IPython.display import Markdown, display

    ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
                if (path / 'Annotations MAJ.xlsx').exists() and (path / 'src/eeg_project.py').exists())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import eeg_presentation as presentation
    from src import eeg_project as eeg
    from src.eeg_source_validation import validate_exports

    OUT = ROOT / 'artifacts/final'
    FIGURES = OUT / 'figures'
    OUT.mkdir(parents=True, exist_ok=True)
    RETRAIN = False
    presentation.apply_style()
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.width', 160)
    pd.set_option('display.float_format', lambda value: f'{value:.3f}')
    """)
    code("""
    cohort, events, audit, all_events, unannotated = eeg.load_annotations(ROOT)
    audit.to_csv(OUT / 'cohort_audit.csv', index=False)
    events.to_csv(OUT / 'annotations_seconds.csv', index=False)
    display(audit.rename(columns={
        'record_id': 'Enregistrement', 'excel_row': 'Ligne Excel', 'included': 'Retenu',
        'reason': 'Décision', 'actual_events': 'Paires réelles', 'declared_events': 'Nombre déclaré',
        'count_mismatch': 'Écart de comptage', 'age': 'Âge', 'sex': 'Sexe', 'duration_s': 'Durée du tracé (s)'
    }))
    display(Markdown(f'**Cohorte retenue : {len(cohort)} enregistrements, {len(events)} crises.** '
                     f'Exports sans ligne dans le classeur mis à jour, non utilisés : {", ".join(unannotated)}.'))
    """)
    md("""
    Trois exclusions, décidées avant tout apprentissage :
    `230406B-A` et `230515B-G` apparaissent deux fois avec des annotations
    contradictoires ; `210504B-C` contient une crise à 21 608 s dans un tracé de
    3 121 s. `210914B-A` déclare `NombreCE = 6` mais décrit 11 paires : les 11 sont
    conservées. Les colonnes mal orthographiées `DebutCE5` et `DebtCE11` sont reconnues
    par leur numéro.
    """)

    md("""
    ## 2. Exploration
    **Q1.** Les `.mat` contiennent des objets MATLAB `timetable` (MCOS) que `scipy.io`
    et `h5py` ne lisent pas. Les exports NPZ sont comparés **valeur par valeur** aux
    cellules numériques des fichiers d'origine : échantillons, noms des canaux, temps.
    La fréquence est mesurée à partir du nombre d'échantillons par bloc et de l'écart
    entre les lignes temporelles, puis vérifiée sur l'axe développé.
    """)
    code("""
    validation = validate_exports(ROOT, cohort.record_id, OUT)
    display(validation[['record_id', 'n_channels', 'n_samples', 'samples_per_time_row',
                        'time_step_s', 'fs_hz', 'duration_s', 'source_values_match',
                        'source_names_match', 'source_times_match']].rename(columns={
        'record_id': 'Enregistrement', 'n_channels': 'Canaux', 'n_samples': 'Échantillons',
        'samples_per_time_row': 'Échantillons/bloc', 'time_step_s': 'Pas des blocs (s)',
        'fs_hz': 'Fréquence mesurée (Hz)', 'duration_s': 'Durée (s)',
        'source_values_match': 'Valeurs conformes', 'source_names_match': 'Canaux conformes',
        'source_times_match': 'Temps conformes'
    }))
    DEMO = '190304A-E'
    t_demo, signal_demo, fs_demo = eeg.load_trace(ROOT, DEMO, validation)
    demo_intervals = list(events.loc[events.record_id == DEMO, ['start_s', 'end_s']]
                          .itertuples(index=False, name=None))
    info = validation.set_index('record_id').loc[DEMO]
    display(pd.Series({
        'Enregistrement': DEMO, 'Lignes': len(signal_demo), 'Variables (Time + canaux)': int(info.n_channels) + 1,
        'Fréquence mesurée (Hz)': fs_demo, 'Durée couverte (s)': len(signal_demo) / fs_demo,
        'Valeurs manquantes ou non finies': int((~np.isfinite(signal_demo)).sum())
    }).to_frame('Q1'))
    display(pd.DataFrame({'Time (s)': t_demo[:6], eeg.CHANNEL: signal_demo[:6]}))
    """)
    md("""
    Le canal **F7** (`EEGF7`), présent dans tous les tracés, est fixé avant toute
    comparaison : les absences sont des crises généralisées, visibles sur l'ensemble
    du scalp. Les amplitudes sont dans l'unité du fichier, qui n'est pas documentée.
    """)

    md("""
    ## 3. Visualisations
    **Q2.** Le tracé complet, puis un zoom de 10 s autour de la crise annotée.
    """)
    code("""
    presentation.signal_plot(t_demo, signal_demo, demo_intervals, DEMO, FIGURES, name='signal_annote')
    a, b = demo_intervals[0]
    presentation.signal_plot(t_demo, signal_demo, demo_intervals, DEMO, FIGURES,
                             name='signal_zoom', bounds=(a - 10, b + 10))
    """)
    md("""
    Pendant l'intervalle annoté, le tracé montre une activité rythmique ample et
    régulière, nettement différente de l'activité de fond. Le tracé complet montre aussi
    une **dérive lente et un décalage continu** : le signal n'est pas centré sur zéro.
    Ce point aura des conséquences en section 6.
    """)

    md("""
    ## 4. Traitement des annotations
    **Q3–Q4.** Les débuts et fins sont des **horaires Excel** (fractions de jour).
    Ils sont convertis en secondes depuis le début du tracé :

    `début_s = (DebCE − DebutTrace) × 86 400`, `fin_s = (FinCE − DebutTrace) × 86 400`.

    Pour `190304A-E` : tracé commencé à 10:13:35, crise de 10:35:06 à 10:35:13,
    soit **[1 291 s ; 1 298 s]** et une durée de **7 s**.
    """)
    code("""
    display(events[['record_id', 'event', 'start_s', 'end_s', 'duration_s']]
            .rename(columns={'record_id': 'Enregistrement', 'event': 'Crise',
                             'start_s': 'Début (s)', 'end_s': 'Fin (s)', 'duration_s': 'Durée (s)'}))
    gaps = presentation.duration_plots(events, FIGURES)
    display(pd.Series({
        'Nombre de crises': len(events), 'Durée moyenne (s)': events.duration_s.mean(),
        'Durée médiane (s)': events.duration_s.median(), 'Durée minimale (s)': events.duration_s.min(),
        'Durée maximale (s)': events.duration_s.max(),
        'Intervalle moyen entre deux crises (s)': gaps.mean()
    }).to_frame('Q4'))
    """)
    md("""
    Les durées ne sont pas identiques : de 3 à 49 s, avec une médiane de 11 s. Une
    crise de 3 s ne couvre qu'une ou deux fenêtres de 2 s, alors qu'une crise de 49 s
    en couvre une cinquantaine. La reconstruction des épisodes (section 10) doit en tenir compte.
    """)

    md("""
    ## 5. Segmentation
    **Q5–Q7.** Chaque échantillon reçoit la classe 1 s'il tombe dans un intervalle
    annoté **[début, fin]** (bornes incluses, comme dans l'énoncé), 0 sinon. Les
    fenêtres durent **2 s** (512 échantillons) avec un recouvrement de **50 %** (pas
    de 256). Une fenêtre est de classe 1 si **au moins 50 %** de ses échantillons sont
    en crise. Les tracés ne sont jamais concaténés.
    """)
    code("""
    presentation.show_source(eeg.sample_mask, eeg.segment_signal)
    """)
    code("""
    demo_windows, demo_starts, demo_labels, demo_coverage, demo_mask = eeg.segment_signal(
        t_demo, signal_demo, demo_intervals)
    display(pd.DataFrame({'Classe': ['Hors crise', 'Crise'],
                          'Échantillons': [int((~demo_mask).sum()), int(demo_mask.sum())],
                          'Fenêtres': [int((demo_labels == 0).sum()), int((demo_labels == 1).sum())]}))
    display(pd.DataFrame({'Début fenêtre (s)': t_demo[demo_starts], 'Part de crise': demo_coverage,
                          'Class': demo_labels}).loc[demo_coverage > 0])
    """)
    md("""
    La crise de 7 s produit 8 fenêtres positives. La fenêtre qui commence à 1 297 s
    n'a que 50,2 % de ses échantillons en crise : elle est positive de justesse.
    """)

    md("""
    ## 6. Extraction des caractéristiques
    **Q8–Q11.** Une ligne par fenêtre. Les 12 prédicteurs demandés : moyenne,
    écart-type, variance, minimum, maximum, amplitude (max − min), RMS, énergie (Σx²),
    et les puissances FFT unilatérales en Delta [0,5 ; 4[, Theta [4 ; 8[, Alpha [8 ; 13[
    et Beta [13 ; 30[ Hz.
    """)
    code("""
    presentation.show_source(eeg.one_sided_spectrum, eeg.extract_features)
    """)
    code("""
    frame, counts = eeg.build_features(ROOT, cohort, events, validation)
    frame.to_csv(OUT / 'features.csv', index=False)
    counts.to_csv(OUT / 'class_counts.csv', index=False)
    assert np.isfinite(frame[eeg.FEATURES + eeg.IMPROVED_FEATURES].to_numpy()).all()
    display(frame[['record_id', 'start_s', *eeg.FEATURES, 'Class']].head(5))
    display(counts.rename(columns={'record_id': 'Enregistrement', 'samples': 'Échantillons',
                                   'positive_samples': 'Échantillons crise', 'windows': 'Fenêtres',
                                   'positive_windows': 'Fenêtres crise', 'fs_hz': 'Fréquence (Hz)',
                                   'duration_s': 'Durée (s)', 'channel': 'Canal',
                                   'missing_values': 'Valeurs manquantes'}))
    display(Markdown(
        f'**Cohorte :** {counts.samples.sum():,} échantillons, dont '
        f'{100 * counts.positive_samples.sum() / counts.samples.sum():.2f} % en crise ; '
        f'{len(frame):,} fenêtres, dont {int(frame.Class.sum())} positives ({100 * frame.Class.mean():.2f} %). '
        'Un classifieur qui répond toujours « hors crise » aurait donc environ 96 % d\\'accuracy.'
    ))
    presentation.imbalance_plot(counts, FIGURES)
    """)
    code("""
    presentation.spectra_plot(demo_windows, demo_labels, fs_demo, DEMO, FIGURES)
    demo_features = eeg.extract_features(demo_windows, fs_demo)
    display(demo_features.groupby(demo_labels)[['power_delta', 'power_theta', 'power_alpha', 'power_beta']]
            .mean().rename_axis('Classe').rename(index={0: 'Hors crise', 1: 'Crise'}))
    """)
    md("""
    **Q9.** Sur ce tracé, la puissance augmente dans toutes les bandes pendant la crise,
    surtout en Delta (environ ×8) et en Theta : c'est la décharge de pointes-ondes
    autour de 3 Hz, typique des absences.
    """)
    code("""
    feature_summary, correlation = presentation.feature_plots(frame, FIGURES)
    display(feature_summary.T.rename(columns={0: 'Hors crise', 1: 'Crise'}))
    correlation.to_csv(OUT / 'feature_correlations.csv')
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = upper.stack().rename('Corrélation').reset_index()
    pairs.columns = ['Variable 1', 'Variable 2', 'Corrélation']
    display(pairs.loc[pairs['Corrélation'].abs() >= 0.85].sort_values('Corrélation', ascending=False))
    effect = presentation.effect_sizes(frame, eeg.FEATURES).sort_values(ascending=False)
    display(effect.rename('Écart standardisé |d| entre classes').to_frame())
    """)
    md("""
    **Q10–Q11.** Variance et écart-type, énergie et RMS sont liés par construction
    (variance = écart-type², énergie = 512 × RMS²) : ils sont redondants, même si leur
    corrélation de Pearson n'est pas exactement 1. Les caractéristiques qui séparent le
    mieux les classes sont l'amplitude, le minimum et l'écart-type.

    Deux résultats sont toutefois inquiétants :

    1. **La puissance Delta ne sépare presque pas les classes** sur la cohorte
       (|d| ≈ 0), alors qu'elle est multipliée par 8 pendant la crise du tracé de
       démonstration. Ses valeurs ont une queue extrêmement lourde : l'écart-type
       dépasse 20 fois la moyenne, à cause d'artefacts.
    2. **Le minimum et la moyenne figurent parmi les « meilleures » caractéristiques**,
       alors qu'une décharge à 3 Hz ne déplace pas la moyenne d'une fenêtre de 2 s.
    """)
    code("""
    presentation.offset_plot(frame, FIGURES)
    """)
    md("""
    Le signal brut contient un **décalage continu propre à chaque enregistrement**,
    de plusieurs centaines d'unités. La moyenne, le minimum et le maximum indiquent
    donc surtout *de quel tracé* vient une fenêtre. Comme les tracés ont des
    proportions de crises très différentes (de 0,5 % à 15,5 % des fenêtres), un modèle
    peut apprendre l'identité du tracé au lieu de la crise, ce qui ne se généralise pas.

    D'où un **second jeu de caractéristiques** :

    - filtrage passe-haut à 0,5 Hz (Butterworth d'ordre 4, sans déphasage), qui retire le décalage et la dérive ;
    - logarithmes de l'écart-type, de l'amplitude, de la longueur de ligne et des quatre puissances, contre les queues lourdes ;
    - puissances relatives, part de la bande 2,5–4,5 Hz des pointes-ondes, rapport Delta / (Alpha + Beta), asymétrie, kurtosis, fréquence dominante ;
    - centrage-réduction par la médiane et l'IQR **de chaque enregistrement**, sans étiquette, puis bornage à ±10.

    Les 12 caractéristiques de l'énoncé restent calculées et évaluées telles quelles.
    """)
    code("""
    presentation.show_source(eeg.highpass, eeg.extract_improved_features, eeg.robust_normalize)
    """)
    code("""
    base_effect, improved_effect = presentation.effect_comparison_plot(frame, FIGURES)
    display(Markdown(
        f'**Meilleure séparation :** {base_effect.max():.2f} avec les 12 caractéristiques, '
        f'{improved_effect.max():.2f} avec le jeu enrichi. Puissance Delta : '
        f'{base_effect["power_delta"]:.3f} en brut, {improved_effect["log_power_delta"]:.2f} en log normalisé.'
    ))
    """)

    md("""
    ## 7. Entraînement des modèles
    **Q12.** Les fenêtres se recouvrent de 50 % et celles d'un même tracé se
    ressemblent : une séparation fenêtre par fenêtre mettrait des quasi-doublons des
    deux côtés. La séparation 80/20 se fait donc **par enregistrement**. Parmi 256
    tirages groupés, on retient celui qui approche le mieux 80/20 et la proportion de
    crises, **sans regarder aucun score**.
    """)
    code("""
    train_indices, test_indices, folds = eeg.choose_split(frame)
    train_records = set(frame.iloc[train_indices].record_id)
    test_records = set(frame.iloc[test_indices].record_id)
    assert train_records.isdisjoint(test_records)
    split_table = pd.DataFrame([
        {'Partition': name, 'Enregistrements': subset.record_id.nunique(), 'Fenêtres': len(subset),
         'Part des fenêtres (%)': 100 * len(subset) / len(frame),
         'Fenêtres crise': int(subset.Class.sum()), 'Crises (%)': 100 * subset.Class.mean()}
        for name, subset in [('Apprentissage', frame.iloc[train_indices]), ('Test réservé', frame.iloc[test_indices])]
    ])
    display(split_table)
    display(Markdown(f'**Test réservé :** {", ".join(sorted(test_records))}.'))
    """)
    md("""
    Chaque configuration est un pipeline **normalisation → rééchantillonnage éventuel →
    modèle**, réajusté sur chaque pli : le rééchantillonnage ne touche jamais les
    fenêtres évaluées. Hyperparamètres fixés à l'avance : arbre (profondeur 8, feuilles
    d'au moins 5 fenêtres), forêt (200 arbres), KNN (5 voisins pondérés par la
    distance), SVM RBF (C = 1).

    Stratégies : les quatre de l'énoncé (aucune, sous-échantillonnage, duplication,
    SMOTE), plus la **pondération des classes** (`class_weight="balanced"`), qui ne
    s'applique pas au KNN. 2 jeux × 19 couples = **38 configurations**.

    **Règle de sélection, fixée avant le calcul :** la **précision moyenne (AP)**
    moyenne sur trois plis groupés de l'apprentissage. L'AP résume toute la courbe
    précision-rappel sans dépendre du seuil de 0,5, que rien ne justifie avec 4 % de
    positifs. Une configuration est dite *ex aequo* si son AP est à moins d'un
    écart-type de la meilleure.
    """)
    code("""
    presentation.show_source(eeg.make_model)
    """)
    code("""
    experiment = eeg.run_training(frame, train_indices, test_indices, folds, OUT, force=RETRAIN)
    display(presentation.cv_table(experiment))
    presentation.cv_ranking_plot(experiment, FIGURES)
    best = experiment['best']
    display(Markdown(f'**Configuration retenue en validation interne : {eeg.config_label(best)}.**'))
    """)
    md("""
    Avec les 12 caractéristiques de l'énoncé, les configurations sont très proches et
    leurs écarts-types sont du même ordre que leurs différences : choisir entre elles
    revient presque à tirer au sort. Avec le jeu enrichi, la forêt aléatoire se détache
    nettement, quelle que soit la stratégie d'équilibrage.
    """)

    md("""
    ## 8. Évaluation
    **Q13.** Le test garde toutes ses fenêtres et son déséquilibre naturel. La classe
    positive est la crise. Le seuil par défaut est utilisé ici (0,5 pour les
    probabilités, 0 pour la marge du SVM).
    """)
    code("""
    y_test = frame.iloc[test_indices].Class.to_numpy()
    baseline = eeg.score_predictions(y_test, np.zeros(len(y_test), dtype=int))
    display(pd.DataFrame([baseline], index=['Toujours « hors crise »'])
            .rename(columns={'accuracy': 'Accuracy', 'precision': 'Précision', 'recall': 'Rappel', 'f1': 'F1'}))
    display(presentation.metrics_table(experiment, strategy='original'))
    presentation.confusion_grid(experiment, FIGURES)
    """)
    md("""
    **Pourquoi l'accuracy est trompeuse.** La baseline atteint près de 96 %
    d'accuracy sans détecter une seule crise. Le **rappel** compte les crises
    manquées, la **précision** les fausses alertes, et le **F1** combine les deux.
    Dans un contexte de détection, une crise manquée coûte plus qu'une fausse
    alerte, mais trop de fausses alertes rendent l'outil inutilisable : il faut
    regarder les deux.
    """)

    md("""
    ## 9. Comparaison
    Les 38 configurations sur le même test. Le choix est déjà fait (section 7) ; ce
    tableau sert à comprendre, pas à choisir.
    """)
    code("""
    display(presentation.metrics_table(experiment))
    presentation.comparison_plot(experiment, FIGURES)
    """)
    code("""
    res = experiment['results'].set_index('configuration')

    def pick(name, metric):
        return res.loc[name, metric]

    display(Markdown(
        f"**Lecture.** Au seuil par défaut, sur ces quatre tracés, le jeu enrichi n'améliore pas tout. "
        f"Il transforme le KNN (F1 {pick('base__original__knn', 'f1'):.2f} → {pick('improved__original__knn', 'f1'):.2f}) "
        f"et élève l'AP de la forêt ({pick('base__original__rf', 'average_precision'):.2f} → "
        f"{pick('improved__original__rf', 'average_precision'):.2f}), mais la forêt sur les 12 caractéristiques garde "
        f"le meilleur F1 brut ({pick('base__original__rf', 'f1'):.2f} contre {pick('improved__original__rf', 'f1'):.2f}). "
        "Avec 4 tracés, ce classement est fragile : la section 12 le vérifie sur les 16 tracés. "
        "Les stratégies d'équilibrage déplacent le compromis (plus de rappel, moins de précision) sans ajouter d'information."
    ))
    """)

    md("""
    ## 10. Détection temporelle
    **Q14.** Les fenêtres positives qui se touchent ou se recouvrent forment un
    épisode `[début de la première fenêtre ; fin de la dernière]`. Un épisode prédit
    est apparié à une crise annotée si leur **IoU** (intersection / union) atteint
    **0,3**, avec un appariement un à un.

    La sortie brute du classifieur produit beaucoup d'épisodes d'une ou deux fenêtres
    isolées. Trois réglages les filtrent : un **lissage** du score sur 1, 3 ou 5
    fenêtres, un **seuil**, et une **durée minimale** de 1 à 5 fenêtres consécutives.
    Ils maximisent le F1 événementiel sur les **prédictions hors pli de
    l'apprentissage**, puis sont figés avant d'être appliqués au test.
    """)
    code("""
    presentation.show_source(eeg.smooth_scores, eeg.drop_short_runs, eeg.intervals_from_labels)
    """)
    code("""
    temporal = eeg.temporal_analysis(frame, events, experiment, OUT)
    params = temporal['params']
    display(Markdown(f'**Réglages retenus hors pli :** lissage sur {params["smoothing"]} fenêtre(s), '
                     f'seuil {params["threshold"]:.3f}, au moins {params["min_windows"]} fenêtre(s) consécutive(s).'))
    presentation.postprocessing_plot(temporal, FIGURES)
    display(presentation.comparison_table(temporal))
    """)
    code("""
    display(temporal['details'].sort_values(['record_id', 'true_start_s', 'pred_start_s']).rename(columns={
        'record_id': 'Enregistrement', 'status': 'Résultat', 'true_start_s': 'Début réel (s)',
        'true_end_s': 'Fin réelle (s)', 'pred_start_s': 'Début prédit (s)', 'pred_end_s': 'Fin prédite (s)'
    }))
    presentation.temporal_plots(ROOT, events, validation, temporal, FIGURES)
    """)

    md("""
    ## 11. Analyse des erreurs
    Fenêtres et épisodes sont comptés séparément : une crise de 20 s compte une
    vingtaine de fenêtres mais un seul épisode.
    """)
    code("""
    display(temporal['windows'].rename(columns={'record_id': 'Enregistrement', 'accuracy': 'Accuracy',
                                               'precision': 'Précision', 'recall': 'Rappel', 'f1': 'F1'}))
    display(temporal['events'].rename(columns={
        'record_id': 'Enregistrement', 'true_events': 'Crises annotées', 'raw_predicted_events': 'Épisodes bruts',
        'predicted_events': 'Épisodes post-traités', 'precision': 'Précision', 'recall': 'Rappel', 'f1': 'F1',
        'false_alarms_per_hour': 'Fausses alarmes / h'}))
    details = temporal['details'].assign(duration=lambda d: d.true_end_s - d.true_start_s,
                                         predicted=lambda d: d.pred_end_s - d.pred_start_s)
    missed = details[details.status == 'FN']
    found = details[details.status == 'TP']
    false = details[details.status == 'FP']
    raw = temporal['events'].set_index('record_id').raw_predicted_events
    lines = [f"**{len(missed)} crise(s) manquée(s)** sur {len(missed) + len(found)}, "
             f"{len(false)} fausse(s) alarme(s)."]
    if len(missed):
        by_record = missed.groupby('record_id').duration.agg(['count', 'min', 'max'])
        lines.append('Crises manquées par tracé : ' + ' ; '.join(
            f"{rid} : {int(row['count'])} ({row['min']:.0f}–{row['max']:.0f} s, "
            f"{int(raw[rid])} épisode(s) en sortie brute)" for rid, row in by_record.iterrows()) + '.')
        lines.append(f"Plus courte crise retrouvée : {found.duration.min():.0f} s. "
                     f"Plus longue crise manquée : {missed.duration.max():.0f} s.")
    if len(false):
        lines.append('Fausses alarmes : ' + ' ; '.join(
            f"{r.record_id} à {r.pred_start_s:.0f}–{r.pred_end_s:.0f} s ({r.predicted:.0f} s)" for r in false.itertuples()) + '.')
    display(Markdown(' '.join(lines)))
    """)
    md("""
    Les omissions sont concentrées dans un seul tracé, où les scores restent bas
    même pendant les crises : le modèle y sous-estime l'activité de crise, ce qui
    tient davantage à l'enregistrement qu'à la durée des crises, puisqu'une crise
    plus courte est retrouvée ailleurs. Les fausses alarmes durent aussi longtemps que des
    crises réelles ; seule une relecture du tracé dirait s'il s'agit d'artefacts ou
    d'absences non annotées. La validation « un enregistrement exclu » (section 12)
    montre la même hétérogénéité d'un tracé à l'autre.
    """)

    md("""
    ## 12. Améliorations
    **Rééchantillonnage.** Le sous-échantillonnage garde toutes les fenêtres de
    crise de l'apprentissage et autant de fenêtres normales tirées au hasard. Le
    suréchantillonnage garde les normales et multiplie les crises, par duplication
    ou par SMOTE (interpolation entre crises voisines dans l'espace des
    caractéristiques). La règle exacte est `N_normales_retenues = N_crises`.
    """)
    code("""
    resampling = experiment['results'].drop_duplicates('strategy')[[
        'strategy', 'train_positive_before', 'train_negative_before',
        'train_positive_after', 'train_negative_after']].copy()
    resampling.strategy = resampling.strategy.map(eeg.STRATEGIES)
    resampling.columns = ['Stratégie', 'Crises avant', 'Normales avant', 'Crises après', 'Normales après']
    display(resampling)
    results = experiment['results']
    reference = results.loc[results.strategy == 'original']
    before_after = results.merge(reference[['feature_set', 'model', 'precision', 'recall', 'f1']],
                                 on=['feature_set', 'model'], suffixes=('', '_ref'))
    for metric in ('precision', 'recall', 'f1'):
        before_after['Δ ' + metric] = before_after[metric] - before_after[metric + '_ref']
    before_after = before_after.loc[before_after.strategy != 'original']
    pivot = before_after.pivot_table(index=['feature_set', 'strategy'], values=['Δ precision', 'Δ recall', 'Δ f1'],
                                     aggfunc='mean').rename(index=eeg.FEATURE_SET_LABELS, level=0).rename(
                                     index=eeg.STRATEGIES, level=1)
    display(Markdown('**Variation moyenne sur les quatre modèles, par rapport à « sans équilibrage » (test).**'))
    display(pivot[['Δ precision', 'Δ recall', 'Δ f1']])
    """)
    md("""
    Le rééchantillonnage augmente le rappel et fait chuter la précision : il déplace
    le seuil implicite du modèle sans lui apprendre quoi que ce soit de nouveau. Ce
    déplacement s'obtient plus proprement en réglant le seuil du score, comme en
    section 10. Les gains mesurés dans ce projet, confirmés ci-dessous sur les
    16 tracés, viennent de trois sources :

    1. **Les caractéristiques** (section 6) : retirer le décalage continu et passer en log.
    2. **La sélection par AP** plutôt que par F1 au seuil 0,5.
    3. **Le raisonnement par épisodes** (section 10) : lissage, seuil et durée minimale.

    **Validation « un enregistrement exclu ».** Le test ne compte que 4 tracés. Pour
    mesurer la variabilité, la configuration retenue et trois références fixées à
    l'avance (l'ancien choix arbre + SMOTE, la forêt sur les 12 caractéristiques, la
    forêt enrichie) sont réévaluées 16 fois, chaque tracé étant exclu à tour de rôle.
    À chaque itération, le post-traitement est réglé sur les prédictions hors pli des
    15 autres tracés.
    """)
    code("""
    loro = eeg.leave_one_record_out(frame, events, eeg.loro_shortlist(best), OUT, force=RETRAIN)
    display(presentation.loro_table(loro))
    presentation.loro_plot(loro, FIGURES)
    """)
    md("""
    ### Conclusion et limites
    Avec un seul canal et une forêt aléatoire, la chaîne retrouve la grande majorité
    des crises de cette cohorte avec quelques fausses alarmes par heure, à condition
    de retirer le décalage continu, d'utiliser des caractéristiques indépendantes de
    l'échelle de chaque tracé et de raisonner en épisodes. L'équilibrage des classes
    pèse peu face à ces trois choix.

    **Limites.** 16 enregistrements sans identifiant patient fiable ; un seul canal ;
    annotations sans double relecture ; hyperparamètres non optimisés ; la normalisation
    par enregistrement suppose un tracé long, dominé par l'activité de fond. Les
    chiffres LORO, plus prudents que ceux du test réservé, sont ceux à retenir.
    """)
    code("""
    report_path = presentation.write_report(ROOT, cohort, events, counts, frame, experiment, temporal, loro, gaps)
    assert train_records.isdisjoint(test_records)
    assert len(experiment['results']) == 38 and len(experiment['cv']) == 38 * 3
    assert validation[['source_values_match', 'source_names_match', 'source_times_match']].all().all()
    display(Markdown(f'**Rapport régénéré :** `{report_path.relative_to(ROOT)}`, figures dans `reports/figures/`.'))
    """)

    notebook = nbformat.v4.new_notebook(cells=cells)
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cellule_{index + 1}")
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = ROOT / "notebooks/mini_projet_eeg_final.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Notebook construit : {len(cells)} cellules, douze sections.")


if __name__ == "__main__":
    build()
