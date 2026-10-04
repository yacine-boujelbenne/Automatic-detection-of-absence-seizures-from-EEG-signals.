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
    **Mini-projet de Data Mining — notebook final exécuté — 4 octobre 2026**

    **Objectif.** Comparer un arbre de décision, une forêt aléatoire, un KNN et un
    SVM, puis reconstruire les périodes de crises. Nous comparons les données
    initiales, le sous-échantillonnage des fenêtres normales, le suréchantillonnage
    des crises par duplication et SMOTE dans l'espace des caractéristiques.

    Le classeur utilisé est **Annotations MAJ.xlsx**. Les horaires sont convertis
    par rapport à **DebutTrace**, conformément au message de l'enseignant. Les
    signaux restent réels : aucun EEG ou horaire de crise n'est inventé.

    **Lecture pour la présentation :** données et annotations (1–4), fenêtres et
    caractéristiques (5–6), modèles et comparaison (7–9), détection et erreurs
    (10–11), bilan des améliorations et conclusion (12).
    """)
    md("""
    ## 1. Chargement des données
    Le notebook utilise l'interpréteur du projet et les modules `src/` fournis avec
    lui. Il s'exécute depuis la racine ou depuis `notebooks/`. Les entraînements sont
    conservés avec une empreinte des données, du code, des partitions et des versions.
    Mettre `RETRAIN = True` force leur recalcul ; le cache vérifié accélère la présentation.
    """)
    code("""
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import seaborn as sns
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
    sns.set_theme(style='whitegrid', context='notebook', font_scale=1.05)
    pd.set_option('display.max_columns', 18)
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.float_format', lambda value: f'{value:.3f}')
    display(Markdown(f'**Projet :** `{ROOT.name}` — **Graine :** {eeg.SEED} — **Canal fixé :** {eeg.CHANNEL}'))
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
                     f'Exports sans annotation mise à jour, exclus : {", ".join(unannotated)}.'))
    """)
    md("""
    Les deux doublons conflictuels `230406B-A` et `230515B-G` sont exclus. Tout
    `210504B-C` est exclu parce qu'une crise de 21 608 s dépasse la durée du tracé :
    la tronquer ferait presque tout classer en crise. Les onze paires de
    `210914B-A` sont conservées malgré `NombreCE = 6`. Les noms `DebutCE5` et
    `DebtCE11` sont reconnus par leur numéro. L'absence d'annotation ne signifie
    pas absence de crise. Ces décisions sont fixées avant l'entraînement.
    """)
    md("""
    ## 2. Exploration
    **Q1.** Les fichiers MCOS ne sont pas directement lus comme des tableaux EEG
    par `scipy.io` ou `h5py`. Les exports déjà présents sont contrôlés contre les
    **cellules numériques structurées** du fichier MATLAB original : tous les
    blocs, leurs dimensions, les noms ordonnés et les temps doivent correspondre.
    Les empreintes SHA-256 invalident le contrôle si un fichier change.

    La propriété temporelle MATLAB `millis` identifie les millisecondes. La
    fréquence est dérivée du nombre réel d'échantillons par bloc et de l'écart
    entre les lignes temporelles, puis vérifiée sur le vecteur `Time` développé.
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
        'Type EEG exporté': 'float32, calculs en float64', 'Fréquence mesurée (Hz)': fs_demo,
        'Durée couverte (s)': len(signal_demo) / fs_demo,
        'Valeurs manquantes / non finies dans le canal': int((~np.isfinite(signal_demo)).sum())
    }).to_frame('Q1'))
    display(pd.DataFrame({'Time (s)': t_demo[:8], eeg.CHANNEL: signal_demo[:8]}))
    """)
    md("""
    Le canal commun **EEGF7** est fixé avant toute comparaison. Aucun canal n'est
    choisi selon les performances du test. Les amplitudes sont exprimées dans
    l'unité du fichier : une unité physique non fournie n'est pas inventée.
    La durée couverte est `N / fs`, tandis que le dernier temps d'échantillon est
    `(N − 1) / fs`. Le contrôle de source est technique, sans validation clinique.
    """)
    md("""
    ## 3. Visualisations
    **Q2.** Le signal complet et un zoom présentent les périodes annotées.
    Les titres identifient le tracé et le canal ; les axes indiquent le temps et
    l'unité disponible. L'affichage du tracé complet peut être réduit en points,
    mais les calculs utilisent tous les échantillons.
    """)
    code("""
    presentation.signal_plot(t_demo, signal_demo, demo_intervals, DEMO, FIGURES, name='signal_annote')
    a, b = demo_intervals[0]
    presentation.signal_plot(t_demo, signal_demo, demo_intervals, DEMO, FIGURES,
                             name='signal_zoom', bounds=(a - 10, b + 10))
    """)
    md("""
    **Observation visuelle.** Le zoom permet de comparer l'amplitude et la
    régularité du tracé pendant l'intervalle annoté et autour de lui. Une décharge
    rythmique ou une variation d'amplitude constitue une observation locale ;
    elle ne démontre pas à elle seule que les quatre modèles généraliseront.
    La FFT et les distributions de caractéristiques complètent cette lecture.
    """)
    md("""
    ## 4. Traitement des annotations
    **Q3–Q4.** Les valeurs sont des **horaires Excel**, et non des durées ou des
    fractions de la durée de l'enregistrement. La précision de l'enseignant
    fournit l'origine temporelle désormais disponible.

    `début_s = (DebCE − DebutTrace) × 86400` ;
    `fin_s = (FinCE − DebutTrace) × 86400` ;
    `durée_s = fin_s − début_s`.

    Pour `190304A-E` : 10:13:35 → début du tracé ; 10:35:06 → début de crise ;
    10:35:13 → fin de crise. La crise est **[1291 s, 1298 s]**, soit **7 s**.
    """)
    code("""
    display(events[['record_id', 'event', 'start_s', 'end_s', 'duration_s']]
            .rename(columns={'record_id': 'Enregistrement', 'event': 'Crise',
                             'start_s': 'Début (s)', 'end_s': 'Fin (s)', 'duration_s': 'Durée (s)'}))
    display(events.groupby('record_id').agg(Nombre=('event', 'count'),
                                            Duree_totale_s=('duration_s', 'sum')))
    gaps = presentation.duration_plots(events, FIGURES)
    display(pd.Series({
        'Nombre de crises': len(events), 'Durée moyenne (s)': events.duration_s.mean(),
        'Durée médiane (s)': events.duration_s.median(), 'Durée minimale (s)': events.duration_s.min(),
        'Durée maximale (s)': events.duration_s.max(),
        'Intervalle libre moyen fin → début suivant (s)': gaps.mean()
    }).to_frame('Q4'))
    display(Markdown(f'**Les durées ne sont pas identiques :** de {events.duration_s.min():.0f} '
                     f'à {events.duration_s.max():.0f} s, avec une médiane de {events.duration_s.median():.0f} s. '
                     'Les intervalles entre crises sont calculés dans un même enregistrement.'))
    """)
    md("""
    ## 5. Segmentation
    **Q5–Q7.** Chaque échantillon reçoit la classe 0 (hors crise) ou 1 (crise).
    Le PDF, p. 4, imprime **[début, fin]**. Nous suivons cette convention fermée,
    conformément à la demande de fidélité au PDF ; la référence locale utilise
    `(début, fin]`, soit un échantillon de différence par début aligné.

    Une fenêtre dure **2 s** et recouvre la précédente à **50 %**. Sa classe vaut
    1 lorsque **au moins 50 %** de ses échantillons sont en crise. Les fenêtres
    incomplètes en fin de tracé sont omises ; les tracés ne sont jamais concaténés.
    """)
    code("""
    demo_windows, demo_starts, demo_labels, demo_coverage, demo_mask = eeg.segment_signal(
        t_demo, signal_demo, demo_intervals)
    display(pd.DataFrame({'Classe': ['Hors crise', 'Crise'],
                          'Échantillons': [int((~demo_mask).sum()), int(demo_mask.sum())],
                          'Fenêtres': [int((demo_labels == 0).sum()), int((demo_labels == 1).sum())]}))
    display(pd.DataFrame({'Début fenêtre (s)': t_demo[demo_starts], 'Couverture de crise': demo_coverage,
                          'Class': demo_labels}).loc[demo_coverage > 0])
    display(Markdown(f'**Fenêtre :** {demo_windows.shape[1]} échantillons ; '
                     f'**pas :** {demo_starts[1] - demo_starts[0]} échantillons.'))
    """)
    md("""
    ## 6. Extraction des caractéristiques
    **Q8–Q11.** Une ligne est une fenêtre et les prédicteurs sont exactement les
    huit statistiques temporelles et les quatre puissances demandées.

    | Temporelles | Définition |
    |---|---|
    | Moyenne, écart-type, variance | Statistiques de population (`ddof = 0`) |
    | Minimum, maximum, amplitude | Amplitude = maximum − minimum |
    | RMS, énergie | RMS = √moyenne(x²) ; énergie = Σx² |

    La FFT est unilatérale, avec normalisation `|rFFT|²/N²` et doublement hors
    composantes DC et Nyquist. La somme de ses puissances est la moyenne du carré
    du signal. Les bandes sont Delta [0,5;4[, Theta [4;8[, Alpha [8;13[ et
    Beta [13;30[ Hz ; leurs frontières ne sont pas comptées deux fois.
    """)
    code("""
    frame, counts = eeg.build_features(ROOT, cohort, events, validation)
    frame.to_csv(OUT / 'features.csv', index=False)
    counts.to_csv(OUT / 'class_counts.csv', index=False)
    assert frame[eeg.FEATURES].shape[1] == 12
    assert np.isfinite(frame[eeg.FEATURES].to_numpy()).all()
    display(frame[['record_id', 'start_s', *eeg.FEATURES, 'Class']].head(6))
    display(counts.rename(columns={'record_id': 'Enregistrement', 'samples': 'Échantillons',
                                   'positive_samples': 'Échantillons crise', 'windows': 'Fenêtres',
                                   'positive_windows': 'Fenêtres crise', 'fs_hz': 'Fréquence (Hz)',
                                   'duration_s': 'Durée (s)', 'channel': 'Canal',
                                   'missing_values': 'Valeurs manquantes'}))
    display(Markdown(
        f'**Q6/Q8 — Cohorte :** {counts.samples.sum():,} échantillons ; '
        f'{100 * counts.positive_samples.sum() / counts.samples.sum():.2f} % en crise. '
        f'**Fenêtres :** {len(frame):,}, dont {int(frame.Class.sum()):,} crises '
        f'({100 * frame.Class.mean():.2f} %) et {int((frame.Class == 0).sum()):,} hors crise. '
        'Ces deux pourcentages portent sur des unités différentes ; le déséquilibre est important.'
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
    **Q9.** Le spectre et les quatre puissances ci-dessus sont calculés sur le
    canal réel. Leur comparaison indique dans quelles bandes l'énergie varie
    pour cet enregistrement. Une bande pédagogique, notamment Delta 0,5–4 Hz,
    ne doit pas être confondue avec un diagnostic clinique fondé uniquement sur
    une fréquence ou avec un résultat commun à tous les tracés.
    """)
    code("""
    feature_summary, correlation = presentation.feature_plots(frame, FIGURES)
    display(feature_summary.T.rename(columns={0: 'Hors crise', 1: 'Crise'}))
    correlation.to_csv(OUT / 'feature_correlations.csv')
    feature_summary.to_csv(OUT / 'features_by_class.csv')
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    pairs = upper.stack().rename('Corrélation').reset_index()
    pairs.columns = ['Variable 1', 'Variable 2', 'Corrélation']
    display(pairs.loc[pairs['Corrélation'].abs() >= 0.85].sort_values('Corrélation', ascending=False))
    pooled_std = np.sqrt((frame.loc[frame.Class == 0, eeg.FEATURES].var()
                         + frame.loc[frame.Class == 1, eeg.FEATURES].var()) / 2)
    effect = ((frame.loc[frame.Class == 1, eeg.FEATURES].mean()
               - frame.loc[frame.Class == 0, eeg.FEATURES].mean()) / pooled_std.replace(0, np.nan)).abs()
    display(effect.sort_values(ascending=False).rename('Écart standardisé absolu entre classes').to_frame())
    display(Markdown('**Q11 — Réponses.** '
        '1. Les paires fortement corrélées sont celles du tableau ci-dessus. '
        '2. Variance et écart-type, ainsi que énergie et RMS, ont des relations déterministes : '
        'il existe une redondance, même si la corrélation de Pearson ne vaut pas exactement 1. '
        f'3. Selon cet écart descriptif, les caractéristiques les plus discriminantes sont '
        f'**{", ".join(effect.sort_values(ascending=False).head(3).index)}**. '
        'Cette EDA décrit la cohorte ; elle ne sert pas à sélectionner des variables à partir du test. '
        'Les 12 prédicteurs sont conservés pour toutes les configurations.'))
    """)
    md("""
    ## 7. Entraînement des modèles
    **Q12.** Une séparation groupée empêche les fenêtres partageant des
    échantillons de se retrouver dans les deux ensembles. Des partitions fixées
    avec la graine 42 sont comparées **uniquement selon les effectifs et proportions**
    pour approcher 80/20 et une stratification des classes. Les groupes indivisibles
    rendent le ratio exact impossible ; les proportions réalisées sont publiées.

    Trois plis de validation stratifiée groupée, **dans l'apprentissage seulement**,
    servent à sélectionner le F1 moyen. Le rappel et la précision départagent les
    égalités. Les hyperparamètres sont fixés à l'avance : arbre (profondeur 8,
    feuille minimale 5), forêt (200 arbres, feuille minimale 2), KNN (5 voisins,
    poids de distance), SVM RBF (`C = 1`). Ils ne sont pas ajustés sur le test.
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
    display(pd.DataFrame({'Apprentissage': pd.Series(sorted(train_records)),
                          'Test réservé': pd.Series(sorted(test_records))}).fillna(''))
    split_table.to_csv(OUT / 'split_summary.csv', index=False)
    """)
    md("""
    Chaque configuration est un pipeline **StandardScaler → rééchantillonnage
    éventuel → modèle**. Normalisation et équilibrage sont réajustés uniquement
    sur chaque pli d'apprentissage. L'égalité des effectifs concerne l'entraînement,
    jamais le test. Les identifiants et horaires servent à regrouper et reconstruire
    les fenêtres ; ils ne font pas partie des prédicteurs.

    SMOTE interpole des caractéristiques de crises réelles de l'apprentissage ;
    ses lignes ne sont ni de nouveaux patients, ni des crises réellement annotées.
    Les deux stratégies proposées sont ainsi comparées à une référence sans
    équilibrage, avec la duplication comme contrôle de l'interpolation.
    """)
    code("""
    experiment = eeg.run_training(frame, train_indices, test_indices, folds, OUT, force=RETRAIN)
    cv_display = experiment['cv_summary'].copy()
    cv_display['strategy'] = cv_display.strategy.map(eeg.STRATEGIES)
    cv_display['model'] = cv_display.model.map(eeg.MODELS)
    display(cv_display[['strategy', 'model', 'precision_mean', 'recall_mean', 'f1_mean', 'f1_std']]
            .rename(columns={'strategy': 'Stratégie', 'model': 'Modèle', 'precision_mean': 'Précision CV',
                             'recall_mean': 'Rappel CV', 'f1_mean': 'F1 moyen CV', 'f1_std': 'Écart-type F1 CV'}))
    selected_strategy, selected_model = experiment['best'].split('__')
    display(Markdown(f'**Choix fixé en validation interne : {eeg.MODELS[selected_model]} '
                     f'avec {eeg.STRATEGIES[selected_strategy].lower()}.** '
                     'Le classement du test ne change pas ce choix.'))
    """)
    md("""
    ## 8. Évaluation
    **Q13.** Le test garde toutes ses fenêtres réelles et son déséquilibre naturel.
    Les quatre métriques demandées et les matrices de confusion sont présentées
    pour chaque configuration. La classe positive est toujours la crise.
    """)
    code("""
    y_test = frame.iloc[test_indices].Class.to_numpy()
    baseline = eeg.score_predictions(y_test, np.zeros(len(y_test), dtype=int))
    display(pd.DataFrame([baseline], index=['Toujours hors crise'])
            .rename(columns={'accuracy': 'Accuracy', 'precision': 'Précision', 'recall': 'Rappel', 'f1': 'F1'}))
    display(presentation.metrics_table(experiment, strategy='original'))
    presentation.confusion_plots(experiment, FIGURES)
    """)
    md("""
    **Pourquoi l'accuracy est trompeuse.** La baseline peut avoir une accuracy
    élevée tout en manquant toutes les crises. Le **rappel** est essentiel pour
    rendre ces omissions visibles ; la **précision** mesure la fiabilité des
    fenêtres signalées. Un rappel parfait en prédisant « crise » partout ne
    suffirait pas. Le **F1** sert ici de compromis pour sélectionner le modèle,
    accompagné des deux métriques et du nombre de fausses alertes. Une précision
    n'est pas un nombre d'alarmes par heure.
    """)
    md("""
    ## 9. Comparaison
    Le tableau contient **16 configurations**, évaluées sur les mêmes fenêtres.
    Les performances de validation interne et du test sont distinctes.
    Aucune configuration n'est retouchée après lecture des résultats du test.
    """)
    code("""
    display(presentation.metrics_table(experiment))
    presentation.comparison_plot(experiment, FIGURES)
    selected_test = experiment['results'].set_index('configuration').loc[experiment['best']]
    best_observed = experiment['results'].sort_values('f1', ascending=False).iloc[0]
    observed_strategy, observed_model = best_observed.configuration.split('__')
    display(Markdown(
        f'Le modèle **sélectionné en CV** obtient sur le test une précision de '
        f'**{selected_test.precision:.3f}**, un rappel de **{selected_test.recall:.3f}** '
        f'et un F1 de **{selected_test.f1:.3f}**. '
        f'Le plus grand F1 **observé sur ce test** est {best_observed.f1:.3f}, '
        f'pour {eeg.MODELS[observed_model]} / {eeg.STRATEGIES[observed_strategy]}. '
        'Cette différence illustre la variabilité entre enregistrements ; elle ne conduit '
        'pas à sélectionner rétrospectivement le modèle sur le test.'
    ))
    """)
    md("""
    ## 10. Détection temporelle
    **Q14.** Les supports temporels des fenêtres positives sont fusionnés lorsqu'ils
    se touchent ou se recouvrent, dans un même tracé. Les bornes sont celles des
    fenêtres, sans déplacer l'origine ni fixer une durée de crise arbitraire.
    Cette reconstruction a une résolution liée au pas de 1 s et n'est pas une
    estimation clinique précise des limites.

    Pour l'analyse événementielle complémentaire, l'appariement est un à un,
    avec **IoU ≥ 0,3** fixée avant calcul : intersection des intervalles divisée
    par leur union. Le meilleur modèle de validation interne est utilisé.
    """)
    code("""
    temporal = eeg.temporal_analysis(frame, events, experiment, OUT)
    display(temporal['predicted'].rename(columns={'record_id': 'Enregistrement', 'event': 'Crise détectée',
                                                  'start_s': 'Début prédit (s)', 'end_s': 'Fin prédite (s)'}))
    display(temporal['details'].loc[temporal['details'].status != 'FP'].rename(columns={
        'record_id': 'Enregistrement', 'status': 'Résultat', 'true_start_s': 'Début réel (s)',
        'true_end_s': 'Fin réelle (s)', 'pred_start_s': 'Début prédit (s)', 'pred_end_s': 'Fin prédite (s)'
    }))
    presentation.temporal_plots(ROOT, events, validation, temporal, FIGURES)
    """)
    md("""
    ## 11. Analyse des erreurs
    Les TP, FP, FN et TN au niveau des fenêtres sont affichés par enregistrement.
    Les événements correctement appariés, manqués ou non appariés sont comptés
    séparément. Un événement réel peut produire plusieurs fenêtres positives ;
    les deux niveaux ne doivent pas être confondus. Il n'existe pas de nombre
    naturel de TN au niveau des événements continus.
    """)
    code("""
    display(temporal['windows'].rename(columns={'record_id': 'Enregistrement', 'accuracy': 'Accuracy',
                                               'precision': 'Précision', 'recall': 'Rappel', 'f1': 'F1'}))
    display(temporal['events'].rename(columns={'record_id': 'Enregistrement', 'true_events': 'Crises réelles',
                                              'predicted_events': 'Crises prédites', 'precision': 'Précision',
                                              'recall': 'Rappel', 'f1': 'F1'}))
    totals = temporal['events'][['TP', 'FP', 'FN']].sum()
    event_recall = totals.TP / (totals.TP + totals.FN) if totals.TP + totals.FN else 0
    event_precision = totals.TP / (totals.TP + totals.FP) if totals.TP + totals.FP else 0
    test_hours = validation.loc[validation.record_id.isin(test_records), 'duration_s'].sum() / 3600
    display(Markdown(f'**Événements, IoU ≥ 0,3 :** {totals.TP} TP, {totals.FP} FP, {totals.FN} FN ; '
                     f'rappel {event_recall:.3f}, précision {event_precision:.3f}. '
                     f'**Événements non appariés par heure : {totals.FP / test_hours:.1f}.**'))
    display(temporal['details'].loc[temporal['details'].status == 'FP'].head(10))
    """)
    md("""
    Un événement non apparié peut être un artefact, une détection trop fragmentée,
    un intervalle mal borné ou une anomalie non couverte par les annotations. Sans
    revue clinique indépendante, sa cause n'est pas affirmée. Les fenêtres FN
    peuvent correspondre à un début, une fin ou une portion d'une crise malgré un
    événement finalement apparié. Les chronologies ci-dessus permettent de
    distinguer ces situations ; les moyennes globales ne les remplacent pas.
    """)
    md("""
    ## 12. Améliorations
    Les deux solutions proposées ont été **réellement mesurées** sur le test
    commun. Le sous-échantillonnage garde toutes les fenêtres de crises de
    l'apprentissage et un nombre égal de fenêtres normales. Le suréchantillonnage
    conserve les normales et multiplie les positives, par duplication ou SMOTE.

    Prélever « 3 % des normales » ne donne pas exactement le même effectif que
    « 3 % du total » : la règle correcte est `N_normal_retenu = N_crise`.
    Ces opérations concernent des fenêtres déjà construites, pas un signal
    artificiel obtenu en concaténant des morceaux.
    """)
    code("""
    resampling = experiment['results'].drop_duplicates('strategy')[[
        'strategy', 'train_positive_before', 'train_negative_before',
        'train_positive_after', 'train_negative_after']].copy()
    resampling.strategy = resampling.strategy.map(eeg.STRATEGIES)
    resampling.columns = ['Stratégie', 'Crises avant', 'Normales avant', 'Crises après', 'Normales après']
    display(resampling)
    reference = experiment['results'].loc[experiment['results'].strategy == 'original']
    before_after = experiment['results'].merge(reference[['model', 'precision', 'recall', 'f1']],
                                               on='model', suffixes=('', '_reference'))
    for metric in ('precision', 'recall', 'f1'):
        before_after['delta_' + metric] = before_after[metric] - before_after[metric + '_reference']
    before_after = before_after.loc[before_after.strategy != 'original', [
        'strategy', 'model', 'precision', 'delta_precision', 'recall', 'delta_recall', 'f1', 'delta_f1']]
    before_after.strategy = before_after.strategy.map(eeg.STRATEGIES)
    before_after.model = before_after.model.map(eeg.MODELS)
    before_after.columns = ['Stratégie', 'Modèle', 'Précision', 'Δ précision', 'Rappel', 'Δ rappel', 'F1', 'Δ F1']
    display(before_after)
    before_after.to_csv(OUT / 'improvements_before_after.csv', index=False)
    display(Markdown('**Interprétation :** un gain de rappel doit être lu avec le changement de '
                     "précision et de F1. L'équilibrage ne crée pas d'information indépendante "
                     'et ne garantit pas un meilleur résultat pour chaque modèle.'))
    """)
    md("""
    ### Conclusion et limites
    Le notebook compare quatre familles de modèles et quatre stratégies avec des
    horaires corrigés et une séparation sans partage d'enregistrement. Le choix
    du modèle repose sur la validation interne et ses erreurs restent visibles.

    La cohorte de **16 tracés / 65 crises** est petite ; le test comprend quatre
    tracés. Les plis groupés et leurs écarts-types rendent cette variabilité
    visible. La séparation par enregistrement ne garantit pas une séparation
    par patient, faute d'identifiant patient fiable. Le canal unique, les artefacts,
    les annotations exclues, les unités d'amplitude et la plausibilité des
    interpolations SMOTE limitent la portée des conclusions. Les données ne sont
    pas une validation clinique externe.

    Parmi les perspectives : résoudre les annotations conflictuelles, valider sur
    une autre cohorte, explorer d'autres canaux ou fenêtres et ajuster les seuils
    uniquement en validation sur l'apprentissage. Aucune performance de ces
    perspectives n'est inventée.

    **Sources :** énoncé `mini-prjet-data-mining.pdf`, classeur `Annotations MAJ.xlsx`
    et message de l'enseignant ; [pipeline imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html),
    [rééchantillonnage et fuites](https://imbalanced-learn.org/stable/common_pitfalls.html),
    [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
    [SMOTE, Chawla et al.](https://arxiv.org/abs/1106.1813).
    """)
    code("""
    report_path = presentation.write_report(ROOT, cohort, events, counts, frame, experiment, temporal, gaps)
    assert train_records.isdisjoint(test_records)
    assert len(experiment['results']) == 16 and len(experiment['cv']) == 48
    assert validation[['source_values_match', 'source_names_match', 'source_times_match']].all().all()
    display(Markdown('**Livrables générés :** notebook exécuté, rapport français `reports/RAPPORT_FINAL_EEG.md`, '
                     'figures, tableaux CSV, contrôles source et modèle sélectionné dans `artifacts/final/`.'))
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cellule_{index + 1}")
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python (DM-project)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13.7"},
        "eeg_project_final": True,
    }
    path = ROOT / "notebooks/mini_projet_eeg_final.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Notebook construit : {len(cells)} cellules, douze sections.")


if __name__ == "__main__":
    build()
