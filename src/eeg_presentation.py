"""Figures et rédaction française produites à partir des résultats mesurés."""

import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import eeg_project as eeg

COLORS = {"original": "#2874a6", "under": "#ca6f1e", "over": "#148f77", "smote": "#884ea0"}


def save_figure(fig, folder, name):
    """Sauvegarde la figure présentée sans modifier ses données."""
    fig.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def signal_plot(t, signal, intervals, record_id, folder, name="signal", bounds=None):
    """Affiche l'EEG et les crises sur leurs véritables coordonnées temporelles."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    if bounds is None:
        bounds = (float(t[0]), float(t[-1]))
    keep = (t >= bounds[0]) & (t <= bounds[1])
    indices = np.flatnonzero(keep)
    hop = max(1, len(indices) // 18000)
    indices = indices[::hop]
    ax.plot(t[indices], signal[indices], color="#2874a6", lw=0.7, label="Signal EEG")
    for i, (a, b) in enumerate(intervals):
        ax.axvspan(a, b, color="#c0392b", alpha=0.25, label="Crise annotée" if i == 0 else None)
    ax.set(
        xlabel="Temps depuis le début du tracé (s)",
        ylabel="Amplitude EEG (unité du fichier)",
        title=f"EEG — {record_id}, canal {eeg.CHANNEL}",
        xlim=bounds,
    )
    ax.legend(loc="upper right")
    save_figure(fig, folder, name)


def duration_plots(events, folder):
    """Présente les durées et les intervalles libres entre deux crises successives."""
    sorted_events = events.sort_values(["record_id", "start_s"])
    gaps = sorted_events.start_s - sorted_events.groupby("record_id").end_s.shift()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(events.duration_s, bins=15, color="#2874a6", edgecolor="white")
    axes[0].set(
        xlabel="Durée de crise (s)",
        ylabel="Nombre de crises",
        title="Durées réelles des crises retenues",
    )
    axes[1].hist(gaps.dropna(), bins=15, color="#148f77", edgecolor="white")
    axes[1].set(
        xlabel="Temps libre entre crises (s)",
        ylabel="Nombre d'intervalles",
        title="Intervalle fin → début suivant, même tracé",
    )
    save_figure(fig, folder, "durees_annotations")
    return gaps.dropna()


def imbalance_plot(counts, folder):
    """Compare la proportion de fenêtres de crises entre les enregistrements."""
    fig, ax = plt.subplots(figsize=(11, 5))
    rates = 100 * counts.positive_windows / counts.windows
    ax.barh(counts.record_id, rates, color="#c0392b")
    ax.set(
        xlabel="Fenêtres de crises (%)",
        ylabel="Enregistrement",
        title="Déséquilibre mesuré avant tout rééchantillonnage",
    )
    save_figure(fig, folder, "desequilibre")


def spectra_plot(windows, labels, fs, record_id, folder):
    """Compare une fenêtre positive et une fenêtre négative avec leur FFT."""
    positive = windows[np.flatnonzero(labels == 1)[0]]
    negative = windows[np.flatnonzero(labels == 0)[0]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for row, (window, name, color) in enumerate(
        ((negative, "hors crise", "#2874a6"), (positive, "crise", "#c0392b"))
    ):
        n = len(window)
        axes[row, 0].plot(np.arange(n) / fs, window, color=color, lw=0.8)
        axes[row, 0].set(
            xlabel="Temps dans la fenêtre (s)",
            ylabel="Amplitude (unité du fichier)",
            title=f"{record_id} / {eeg.CHANNEL} — {name}",
        )
        power = np.abs(np.fft.rfft(window)) ** 2 / n**2
        power[1:-1] *= 2
        axes[row, 1].plot(np.fft.rfftfreq(n, 1 / fs), power, color=color)
        axes[row, 1].set(
            xlabel="Fréquence (Hz)",
            ylabel="Puissance par fréquence (unité²)",
            title=f"Spectre FFT unilatéral — {name}",
            xlim=(0, 40),
        )
    save_figure(fig, folder, "spectres")


def feature_plots(frame, folder):
    """Affiche les douze distributions réelles, sans normalisation globale apprise."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax, feature, label in zip(axes.ravel(), eeg.FEATURES, eeg.FEATURE_LABELS, strict=True):
        samples = [frame.loc[frame.Class == k, feature].to_numpy() for k in (0, 1)]
        ax.boxplot(samples, tick_labels=["Hors crise", "Crise"], showfliers=False)
        if feature not in ("mean", "min", "max"):
            ax.set_yscale("symlog", linthresh=1)
        squared = feature == "variance" or feature == "energy" or feature.startswith("power_")
        ax.set(
            xlabel="Classe",
            ylabel="Valeur (unité du fichier²)" if squared else "Valeur (unité du fichier)",
            title=label,
        )
    fig.suptitle(
        "Comparaison des caractéristiques par classe — fenêtres réelles", y=1.01, fontsize=16
    )
    save_figure(fig, folder, "caracteristiques")
    corr = frame[eeg.FEATURES].corr()
    corr.index = eeg.FEATURE_LABELS
    corr.columns = eeg.FEATURE_LABELS
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
        cbar_kws={"label": "Corrélation de Pearson (sans unité)"},
    )
    ax.set(xlabel="Caractéristique", ylabel="Caractéristique", title="Matrice de corrélation — Q11")
    save_figure(fig, folder, "correlations")
    return frame.groupby("Class")[eeg.FEATURES].agg(["mean", "std"]), corr


def confusion_plots(experiment, folder):
    """Présente les matrices des quatre modèles pour chacune des stratégies."""
    for strategy, strategy_name in eeg.STRATEGIES.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, (key, model_name) in zip(axes.ravel(), eeg.MODELS.items(), strict=True):
            matrix = experiment["matrices"][f"{strategy}__{key}"]
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Hors crise", "Crise"],
                yticklabels=["Hors crise", "Crise"],
                ax=ax,
            )
            ax.set(xlabel="Classe prédite", ylabel="Classe réelle", title=model_name)
        fig.suptitle(f"Test réservé — {strategy_name}", fontsize=16)
        save_figure(fig, folder, f"matrices_{strategy}")


def comparison_plot(experiment, folder):
    """Compare rappel, précision et F1 sur le même test naturel."""
    results = experiment["results"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, label in zip(
        axes, ("precision", "recall", "f1"), ("Précision", "Rappel", "F1"), strict=True
    ):
        for i, strategy in enumerate(eeg.STRATEGIES):
            subset = results[results.strategy == strategy].set_index("model").loc[list(eeg.MODELS)]
            ax.bar(
                np.arange(4) + (i - 1.5) * 0.18,
                subset[metric],
                width=0.18,
                color=COLORS[strategy],
                label=eeg.STRATEGIES[strategy],
            )
        ax.set_xticks(np.arange(4), ["Arbre", "Forêt", "KNN", "SVM"])
        ax.set(xlabel="Modèle", ylabel=f"{label} (0 à 1)", title=label, ylim=(0, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4)
    save_figure(fig, folder, "comparaison_modeles")


def temporal_plots(root, events, validation, temporal, folder):
    """Affiche les périodes réelles et prédites séparément pour chaque tracé de test."""
    for rid in sorted(temporal["test_frame"].record_id.unique()):
        t, signal, _ = eeg.load_trace(root, rid, validation)
        truth = events[events.record_id == rid]
        detected = temporal["predicted"][temporal["predicted"].record_id == rid]
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1, 2]})
        hop = max(1, len(t) // 14000)
        axes[0].plot(t[::hop], signal[::hop], lw=0.5, color="#2874a6")
        axes[0].set(
            xlabel="Temps depuis le début (s)",
            ylabel="Amplitude (unité du fichier)",
            title=f"Tracé complet — {rid}, {eeg.CHANNEL}",
            xlim=(0, t[-1]),
        )
        for row, color, _label, data in (
            (1, "#148f77", "Réelles", truth),
            (0, "#c0392b", "Prédites", detected),
        ):
            for event in data.itertuples(index=False):
                axes[1].broken_barh(
                    [(event.start_s, event.end_s - event.start_s)],
                    (row - 0.3, 0.6),
                    facecolors=color,
                )
        axes[1].set_yticks([0, 1], ["Prédites", "Réelles"])
        axes[1].set(
            xlabel="Temps depuis le début (s)",
            ylabel="Périodes de crises",
            title="Annotations et reconstruction du modèle sélectionné",
            xlim=(0, t[-1]),
            ylim=(-0.6, 1.6),
        )
        first = truth.iloc[0]
        a, b = max(0, first.start_s - 8), min(t[-1], first.end_s + 8)
        keep = (t >= a) & (t <= b)
        axes[2].plot(t[keep], signal[keep], lw=0.6, color="#2874a6")
        axes[2].axvspan(
            first.start_s, first.end_s, color="#148f77", alpha=0.25, label="Crise réelle"
        )
        prediction_label_used = False
        for event in detected.itertuples(index=False):
            if event.end_s >= a and event.start_s <= b:
                axes[2].axvspan(
                    event.start_s,
                    event.end_s,
                    color="#c0392b",
                    alpha=0.2,
                    label="Crise prédite" if not prediction_label_used else None,
                )
                prediction_label_used = True
        axes[2].set(
            xlabel="Temps depuis le début (s)",
            ylabel="Amplitude (unité du fichier)",
            title="Zoom sur la première crise réelle du tracé",
            xlim=(a, b),
        )
        axes[2].legend(loc="upper right")
        save_figure(fig, folder, f"detection_{rid}")


def metrics_table(experiment, strategy=None):
    """Prépare le tableau de présentation sans masquer les faibles scores."""
    frame = experiment["results"]
    if strategy is not None:
        frame = frame[frame.strategy == strategy]
    result = frame[
        ["strategy", "model", "accuracy", "precision", "recall", "f1", "TP", "FP", "FN", "TN"]
    ].copy()
    result.strategy = result.strategy.map(eeg.STRATEGIES)
    result.model = result.model.map(eeg.MODELS)
    result.columns = [
        "Stratégie",
        "Modèle",
        "Accuracy",
        "Précision",
        "Rappel",
        "F1",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    return result


def markdown_table(frame):
    """Rédige une table Markdown sans dépendance supplémentaire."""
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(root, cohort, events, counts, frame, experiment, temporal, gaps):
    """Produit le rapport final uniquement à partir des résultats exécutés."""
    best = experiment["results"].set_index("configuration").loc[experiment["best"]]
    strategy, model = experiment["best"].split("__")
    train, test = experiment["train_indices"], experiment["test_indices"]
    event_totals = temporal["events"][["TP", "FP", "FN"]].sum()
    strongest_test = experiment["results"].sort_values("f1", ascending=False).iloc[0]
    strongest_strategy, strongest_model = strongest_test.configuration.split("__")
    baseline = float((frame.iloc[test].Class == 0).mean())
    duration_stats = events.duration_s
    text = f"""# Détection automatique des crises d'absence à partir de l'EEG

Rapport final — 4 octobre 2026. Les nombres proviennent du notebook exécuté `notebooks/mini_projet_eeg_final.ipynb`.

## Introduction

Les crises d'absence (CE) correspondent à des épisodes annotés dans les enregistrements EEG du projet. Nous avons construit une chaîne de classification des fenêtres, comparé quatre modèles et reconstruit les périodes prédites. Le déséquilibre a été traité comme une variable expérimentale, en conservant un test composé de données réelles.

## Problématique

Peut-on détecter automatiquement les CE dans des enregistrements réservés à l'évaluation, et quel modèle et quelle stratégie d'équilibrage obtiennent le meilleur compromis entre rappel et précision ?

## Méthodologie

Le pipeline a suivi : chargement → audit → segmentation → étiquetage → extraction des caractéristiques → séparation → normalisation/rééchantillonnage de l'apprentissage → entraînement → évaluation → reconstruction → analyse des erreurs.

```mermaid
flowchart LR
 A[EEG et horaires] --> B[Audit et contrôles source]
 B --> C[Fenêtres de 2 s, recouvrement 50 %]
 C --> D[12 caractéristiques et classe]
 D --> E[Séparation par enregistrement]
 E --> F[Apprentissage et validation interne]
 F --> G[DT, RF, KNN, SVM]
 G --> H[Test naturel réservé]
 H --> I[Intervalles et erreurs]
```

Nous avons retenu le canal commun {eeg.CHANNEL} avant l'entraînement. La fréquence mesurée est de {counts.fs_hz.min():.0f} Hz sur les tracés retenus : chaque bloc original contient {int(counts.fs_hz.min())} échantillons et les lignes temporelles sources sont espacées d'une seconde. Les exports ont été comparés aux cellules numériques des fichiers MATLAB, avec égalité des échantillons après conversion float32, des noms ordonnés et des temps. Ce contrôle technique ne constitue pas une validation clinique.

Les horaires Excel ont été convertis par `(DebCE - DebutTrace) × 86400` et `(FinCE - DebutTrace) × 86400`. Aucune durée de crise n'a été inventée. Le masque suit l'intervalle fermé [début, fin] imprimé dans le PDF ; la différence avec la convention ouverte à gauche de la référence locale est explicitée dans le notebook.

Chaque fenêtre comporte 512 échantillons et le pas est de 256. Elle est positive si au moins 50 % de ses échantillons sont en crise. Les huit caractéristiques temporelles sont moyenne, écart-type, variance, minimum, maximum, amplitude, RMS et énergie (somme des carrés). Les quatre puissances FFT unilatérales couvrent Delta [0,5;4[, Theta [4;8[, Alpha [8;13[ et Beta [13;30[ Hz.

Nous avons réservé {len(test)} fenêtres ({100 * len(test) / len(frame):.2f} %) de {frame.iloc[test].record_id.nunique()} enregistrements pour le test et utilisé {len(train)} fenêtres de {frame.iloc[train].record_id.nunique()} enregistrements pour l'apprentissage. Une sélection déterministe de partitions groupées a approché le ratio 80/20 et les proportions des classes, sans comparer les scores des modèles. Aucun enregistrement ne figure dans les deux ensembles. Trois plis de validation stratifiée groupée, uniquement dans l'apprentissage, ont servi à sélectionner le meilleur F1 moyen ; le rappel puis la précision départagent les égalités.

Les quatre modèles ont été entraînés avec les mêmes caractéristiques et partitions pour quatre stratégies : déséquilibre initial, sous-échantillonnage aléatoire, suréchantillonnage par duplication et SMOTE. Les transformations sont dans des pipelines ajustés sur chaque pli d'apprentissage. SMOTE crée des caractéristiques interpolées, pas des EEG ou des crises réelles. Les hyperparamètres fixés et la graine 42 sont conservés dans le notebook et le module du projet.

## Données

Le classeur mis à jour comporte 21 lignes d'identifiants et 19 identifiants uniques. Nous avons exclu les deux doublons conflictuels `230406B-A` et `230515B-G`, ainsi que `210504B-C`, dont la première crise dépasse la durée du tracé. Trois exports sans ligne dans le classeur mis à jour n'ont pas été assimilés à des enregistrements normaux. L'écart de `NombreCE` pour `210914B-A` a été documenté et les onze paires explicites ont été conservées.

La cohorte finale comprend {len(cohort)} enregistrements et {len(events)} CE, de durée moyenne {duration_stats.mean():.2f} s, médiane {duration_stats.median():.0f} s, minimale {duration_stats.min():.0f} s et maximale {duration_stats.max():.0f} s. L'intervalle libre moyen entre deux crises successives du même tracé est de {gaps.mean():.2f} s. La segmentation fournit {len(frame)} fenêtres, dont {int(frame.Class.sum())} positives ({100 * frame.Class.mean():.2f} %). La proportion d'échantillons en crise est {100 * counts.positive_samples.sum() / counts.samples.sum():.2f} % ; elle est distincte de celle des fenêtres.

Figure 1 — EEG et annotations : `../artifacts/final/figures/signal_annote.png`.

Figure 2 — Durées des crises : `../artifacts/final/figures/durees_annotations.png`.

## Résultats

Tableau 1 — Quatre modèles sans équilibrage, sur le test réservé.

{markdown_table(metrics_table(experiment, "original"))}

Tableau 2 — Comparaison complète des 16 configurations, sur le même test.

{markdown_table(metrics_table(experiment))}

Figure 3 — Caractéristiques par classe : `../artifacts/final/figures/caracteristiques.png`.

Figure 4 — Corrélations : `../artifacts/final/figures/correlations.png`.

Figure 5 — Comparaison de la précision, du rappel et du F1 : `../artifacts/final/figures/comparaison_modeles.png`.

La variance est le carré de l'écart-type ; l'énergie vaut 512 fois le carré du RMS. Ces identités expliquent une redondance structurelle, sans imposer une corrélation de Pearson exactement égale à un. Les moyennes et écarts par classe ainsi que les corrélations effectivement mesurées sont affichés en Q10–Q11 ; les douze caractéristiques de référence ont été conservées pour isoler l'effet de l'équilibrage.

## Discussion et analyse des erreurs

Le choix fixé en validation interne est **{eeg.MODELS[model]} avec {eeg.STRATEGIES[strategy].lower()}**. Sur le test réservé, ce modèle atteint une précision de {best.precision:.3f}, un rappel de {best.recall:.3f} et un F1 de {best.f1:.3f}, avec TP={int(best.TP)}, FP={int(best.FP)}, FN={int(best.FN)} et TN={int(best.TN)} au niveau des fenêtres. Le choix n'a pas été modifié en fonction du classement sur le test.

Le plus grand F1 observé sur ce test est {strongest_test.f1:.3f}, obtenu par {eeg.MODELS[strongest_model]} / {eeg.STRATEGIES[strongest_strategy]}. Ce résultat descriptif ne remplace pas la sélection antérieure sur les plis internes. L'écart entre ces classements souligne la variabilité de la petite cohorte.

La baseline « toujours hors crise » obtient {baseline:.3f} d'accuracy, avec rappel et F1 nuls. Une accuracy élevée peut donc masquer une absence de détection. Le rappel rend visibles les crises manquées ; la précision et le F1 complètent l'analyse pour éviter de favoriser un modèle qui déclencherait partout. La comparaison avant/après n'affirme pas que l'équilibrage améliore toutes les métriques : elle expose les résultats de chaque stratégie.

Les fenêtres positives successives ont été reconstruites par leur support temporel. Un appariement un à un avec seuil IoU ≥ 0,3, fixé avant l'évaluation, donne {int(event_totals.TP)} événements correctement appariés, {int(event_totals.FP)} événements prédits non appariés et {int(event_totals.FN)} crises non appariées. Cette définition est plus exigeante qu'une simple intersection. Les tables des bornes réelles/prédites et les figures par tracé sont dans le notebook ; les événements ne doivent pas être confondus avec les fenêtres.

## Conclusion

Le projet fournit une comparaison exécutée de quatre familles de modèles et de quatre stratégies sur des annotations horaires corrigées, avec une séparation stricte par enregistrement. Le modèle retenu est celui choisi en validation interne. Ses scores et ses erreurs temporelles, présentés ci-dessus, constituent le résultat de cette expérience ; ils ne démontrent pas une performance clinique sur de nouveaux patients.

## Limites et perspectives

La cohorte de {len(cohort)} tracés demeure petite et le test ne comprend que {frame.iloc[test].record_id.nunique()} enregistrements. Le choix conservateur d'exclure les annotations conflictuelles réduit l'effectif et peut modifier la population étudiée. La correspondance des exports aux valeurs sources a été contrôlée, mais aucune validation clinique indépendante des annotations ou des unités d'amplitude n'a été réalisée. Les fenêtres se recouvrent ; les plis groupés empêchent leur partage entre entraînement et évaluation, sans faire des fenêtres d'un même tracé des observations indépendantes. Aucun identifiant patient fiable ne permet de garantir une séparation par patient. Le canal unique, le montage, les artefacts et la généralisation à d'autres acquisitions restent des limites. Les interpolations SMOTE ne créent aucune nouvelle diversité de patients et peuvent rompre les identités entre caractéristiques. Une validation externe et la résolution des annotations exclues sont des prolongements prioritaires.

## Bibliographie

1. Énoncé local, `mini-prjet-data-mining.pdf`, parties I–XII et questions 1–14.
2. Message de l'enseignant fourni par l'utilisateur : interprétation des débuts et fins comme horaires et exemple `190304A-E`.
3. Classeur source, `Annotations MAJ.xlsx`, feuille `Feuil1`.
4. imbalanced-learn, [Pipeline](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html) et [erreurs fréquentes](https://imbalanced-learn.org/stable/common_pitfalls.html).
5. scikit-learn, [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
6. Chawla et al., [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813), 2002.
"""
    text = "\n".join(
        re.sub(r"(?<=\d)\.(?=\d)", ",", line)
        if not line.startswith("|") and "https://" not in line
        else line
        for line in text.splitlines()
    )
    text = re.sub(
        r"Figure (\d+) — ([^\n]+) : `([^`]+)`\.", r"Figure \1 — \2.\n\n![Figure \1 — \2](\3)", text
    )
    path = root / "reports/RAPPORT_FINAL_EEG.md"
    path.write_text(text, encoding="utf-8")
    return path
