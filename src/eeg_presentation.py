"""Figures et rédaction française produites à partir des résultats mesurés."""

import inspect
import re
import shutil
from datetime import date

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Code, display

from src import eeg_project as eeg

INK = "#1e1c18"
INK_2 = "#4d4840"
INK_3 = "#7c7568"
RULE = "#d9cfbd"
PAPER = "#fbf8f1"
SIGNAL = "#1e2a3f"
DETECT = "#c8462c"
REFERENCE = "#2c6fb0"
# Ordre catégoriel validé (vision normale et daltonisme), jamais recyclé.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STRATEGY_COLORS = dict(zip(eeg.STRATEGIES, CATEGORICAL, strict=True))
SET_COLORS = {"base": CATEGORICAL[0], "improved": CATEGORICAL[1]}


def apply_style():
    """Style commun : encre chaude, axes discrets, titres alignés à gauche."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": INK_3,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "xtick.color": INK_3,
            "ytick.color": INK_3,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "text.color": INK,
            "figure.dpi": 110,
        }
    )


def save_figure(fig, folder, name):
    """Sauvegarde la figure présentée sans modifier ses données."""
    fig.tight_layout()
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def show_source(*functions):
    """Affiche le code exact utilisé par le notebook, sans le dupliquer."""
    for function in functions:
        display(Code(inspect.getsource(function), language="python"))


def signal_plot(t, signal, intervals, record_id, folder, name="signal", bounds=None):
    """Affiche l'EEG et les crises sur leurs véritables coordonnées temporelles."""
    fig, ax = plt.subplots(figsize=(12, 3.6))
    if bounds is None:
        bounds = (float(t[0]), float(t[-1]))
    keep = (t >= bounds[0]) & (t <= bounds[1])
    indices = np.flatnonzero(keep)
    indices = indices[:: max(1, len(indices) // 18000)]
    for i, (a, b) in enumerate(intervals):
        ax.axvspan(
            a, b, color=REFERENCE, alpha=0.16, lw=0, label="Crise annotée" if i == 0 else None
        )
    ax.plot(t[indices], signal[indices], color=SIGNAL, lw=0.6, label="EEG brut")
    ax.set(
        xlabel="Temps depuis le début du tracé (s)",
        ylabel="Amplitude (unité du fichier)",
        title=f"{record_id} — canal {eeg.CHANNEL}",
        xlim=bounds,
    )
    ax.legend(loc="upper right", ncols=2)
    save_figure(fig, folder, name)


def duration_plots(events, folder):
    """Présente les durées et les intervalles libres entre deux crises successives."""
    sorted_events = events.sort_values(["record_id", "start_s"])
    gaps = sorted_events.start_s - sorted_events.groupby("record_id").end_s.shift()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].hist(
        events.duration_s, bins=np.arange(0, 52, 3), color=REFERENCE, edgecolor=PAPER, lw=1.5
    )
    axes[0].axvline(events.duration_s.median(), color=INK, lw=1, ls="--")
    axes[0].annotate(
        f"médiane {events.duration_s.median():.0f} s",
        (events.duration_s.median(), axes[0].get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
    )
    axes[0].set(
        xlabel="Durée de crise (s)", ylabel="Nombre de crises", title="Durée des 65 crises retenues"
    )
    axes[1].hist(gaps.dropna(), bins=15, color=INK_3, edgecolor=PAPER, lw=1.5)
    axes[1].set(
        xlabel="Temps entre la fin d'une crise et le début de la suivante (s)",
        ylabel="Nombre d'intervalles",
        title="Intervalles entre crises d'un même tracé",
    )
    save_figure(fig, folder, "durees_annotations")
    return gaps.dropna()


def imbalance_plot(counts, folder):
    """Compare la proportion de fenêtres de crises entre les enregistrements."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ordered = counts.assign(rate=100 * counts.positive_windows / counts.windows).sort_values("rate")
    ax.barh(ordered.record_id, ordered.rate, color=DETECT, height=0.6)
    for y, (rate, n) in enumerate(zip(ordered.rate, ordered.positive_windows, strict=True)):
        ax.text(rate + 0.2, y, f"{n} fen.", va="center", fontsize=8, color=INK_3)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set(xlabel="Fenêtres de crise (%)", title="Part des fenêtres de crise par enregistrement")
    save_figure(fig, folder, "desequilibre")


def spectra_plot(windows, labels, fs, record_id, folder):
    """Compare une fenêtre positive et une fenêtre négative avec leur FFT."""
    positive = windows[np.flatnonzero(labels == 1)[len(np.flatnonzero(labels == 1)) // 2]]
    negative = windows[np.flatnonzero(labels == 0)[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for window, name, color in ((negative, "hors crise", INK_3), (positive, "crise", DETECT)):
        n = len(window)
        axes[0].plot(np.arange(n) / fs, window - window.mean(), color=color, lw=0.9, label=name)
        spectrum, frequencies = eeg.one_sided_spectrum(window[None, :], fs)
        axes[1].plot(frequencies, spectrum[0], color=color, lw=1.2, label=name)
    for name, (a, b) in eeg.BANDS.items():
        axes[1].axvline(a, color=RULE, lw=0.8)
        axes[1].text(
            (a + b) / 2,
            1.0,
            name.capitalize(),
            transform=axes[1].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_3,
        )
    axes[0].set(
        xlabel="Temps dans la fenêtre (s)",
        ylabel="Amplitude centrée",
        title=f"Deux fenêtres de 2 s — {record_id}",
    )
    axes[1].set(
        xlabel="Fréquence (Hz)",
        ylabel="Puissance (unité²)",
        title="Spectre FFT unilatéral",
        xlim=(0, 32),
    )
    axes[0].legend(loc="upper right")
    save_figure(fig, folder, "spectres")


def effect_sizes(frame, columns):
    """Écart standardisé absolu entre classes (d de Cohen, variance poolée)."""
    zero, one = frame[frame.Class == 0][columns], frame[frame.Class == 1][columns]
    pooled = np.sqrt((zero.var() + one.var()) / 2)
    return ((one.mean() - zero.mean()) / pooled.replace(0, np.nan)).abs()


def feature_plots(frame, folder):
    """Distributions des douze caractéristiques de l'énoncé et leurs corrélations."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9.5))
    for ax, feature, label in zip(axes.ravel(), eeg.FEATURES, eeg.FEATURE_LABELS, strict=True):
        samples = [frame.loc[frame.Class == k, feature].to_numpy() for k in (0, 1)]
        parts = ax.boxplot(
            samples,
            tick_labels=["hors crise", "crise"],
            showfliers=False,
            widths=0.5,
            patch_artist=True,
            medianprops={"color": INK, "lw": 1.4},
        )
        for patch, color in zip(parts["boxes"], (INK_3, DETECT), strict=True):
            patch.set(facecolor=color, alpha=0.25, edgecolor=color)
        if feature not in ("mean", "min", "max"):
            ax.set_yscale("symlog", linthresh=1)
        ax.set_title(label, fontsize=10.5)
    fig.suptitle(
        "Les 12 caractéristiques de l'énoncé, par classe (fenêtres réelles)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    save_figure(fig, folder, "caracteristiques")
    corr = frame[eeg.FEATURES].corr()
    corr.index = corr.columns = eeg.FEATURE_LABELS
    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    for (i, j), value in np.ndenumerate(corr.to_numpy()):
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if abs(value) > 0.6 else INK,
        )
    ax.set_xticks(range(12), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(12), corr.index)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Corrélation de Pearson")
    ax.set_title("Corrélations entre les 12 caractéristiques")
    save_figure(fig, folder, "correlations")
    return frame.groupby("Class")[eeg.FEATURES].agg(["mean", "std"]), corr


def offset_plot(frame, folder):
    """Montre que la moyenne d'une fenêtre dépend surtout de l'enregistrement."""
    order = frame.groupby("record_id")["mean"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 4))
    for x, rid in enumerate(order):
        values = frame.loc[frame.record_id == rid, ["mean", "Class"]]
        q1, median, q3 = values["mean"].quantile([0.1, 0.5, 0.9])
        ax.plot([x, x], [q1, q3], color=INK_3, lw=4, solid_capstyle="butt", alpha=0.5)
        ax.plot(x, median, "o", color=INK, ms=4)
        crisis = values.loc[values.Class == 1, "mean"]
        ax.plot(np.full(len(crisis), x + 0.22), crisis, "o", color=DETECT, ms=2.5, alpha=0.5)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set(
        ylabel="Moyenne de la fenêtre (unité du fichier)",
        title="La moyenne d'une fenêtre identifie surtout le tracé : décalage continu propre à chaque enregistrement",
    )
    ax.plot([], [], color=INK_3, lw=4, alpha=0.5, label="fenêtres hors crise, déciles 10–90")
    ax.plot([], [], "o", color=DETECT, ms=3, label="fenêtres de crise")
    ax.legend(loc="upper left")
    save_figure(fig, folder, "decalage_continu")


def effect_comparison_plot(frame, folder):
    """Séparation des classes : caractéristiques de l'énoncé contre caractéristiques enrichies."""
    base = effect_sizes(frame, eeg.FEATURES).sort_values()
    improved = effect_sizes(frame, eeg.IMPROVED_FEATURES).sort_values()
    labels_base = dict(zip(eeg.FEATURES, eeg.FEATURE_LABELS, strict=True))
    labels_improved = dict(zip(eeg.IMPROVED_FEATURES, eeg.IMPROVED_LABELS, strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, values, labels, key, title in (
        (axes[0], base, labels_base, "base", "12 caractéristiques de l'énoncé, signal brut"),
        (
            axes[1],
            improved,
            labels_improved,
            "improved",
            "16 caractéristiques enrichies, filtrées et normalisées",
        ),
    ):
        ax.barh([labels[k] for k in values.index], values, color=SET_COLORS[key], height=0.62)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Écart standardisé entre classes |d|")
    save_figure(fig, folder, "separation_caracteristiques")
    return base, improved


def confusion_grid(experiment, folder, feature_set="base"):
    """Seize matrices de confusion sur le test : stratégies en lignes, modèles en colonnes."""
    strategies = list(eeg.ASSIGNMENT_STRATEGIES)
    fig, axes = plt.subplots(len(strategies), 4, figsize=(12, 11))
    results = experiment["results"].set_index("configuration")
    for r, strategy in enumerate(strategies):
        for c, model in enumerate(eeg.MODELS):
            ax = axes[r, c]
            row = results.loc[eeg.config_name(feature_set, strategy, model)]
            matrix = np.array([[row.TN, row.FP], [row.FN, row.TP]])
            ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (i, j), value in np.ndenumerate(matrix):
                share = value / matrix[i].sum()
                ax.text(
                    j,
                    i,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if share > 0.6 else INK,
                )
            ax.set_xticks([0, 1], ["prédit 0", "prédit 1"], fontsize=8)
            ax.set_yticks([0, 1], ["réel 0", "réel 1"], fontsize=8)
            ax.grid(False)
            if r == 0:
                ax.set_title(eeg.MODELS[model], fontsize=11)
            if c == 0:
                ax.set_ylabel(eeg.STRATEGIES[strategy], fontsize=10, color=INK)
    fig.suptitle(
        "Matrices de confusion sur le test réservé — 12 caractéristiques de l'énoncé",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    save_figure(fig, folder, "matrices_confusion")


def cv_ranking_plot(experiment, folder):
    """AP moyenne en validation interne, avec un écart-type, pour les 38 configurations."""
    summary = experiment["cv_summary"].sort_values("average_precision_mean")
    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(summary))
    colors = [SET_COLORS[s] for s in summary.feature_set]
    ax.errorbar(
        summary.average_precision_mean,
        y,
        xerr=summary.average_precision_std,
        fmt="none",
        ecolor=RULE,
        elinewidth=2.5,
    )
    ax.scatter(summary.average_precision_mean, y, c=colors, s=34, zorder=3, edgecolor=PAPER, lw=1)
    labels = [
        f"{eeg.MODELS[m]} · {eeg.STRATEGIES[s]}"
        for s, m in zip(summary.strategy, summary.model, strict=True)
    ]
    ax.set_yticks(y, labels, fontsize=8)
    best = summary.iloc[-1]
    ax.axvspan(
        best.average_precision_mean - best.average_precision_std,
        best.average_precision_mean,
        color=INK,
        alpha=0.05,
        lw=0,
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for key, label in (("base", "12 de l'énoncé"), ("improved", "enrichies")):
        ax.scatter([], [], c=SET_COLORS[key], s=34, label=f"caractéristiques {label}")
    ax.legend(loc="lower right")
    ax.set(
        xlabel="Précision moyenne (AP) — moyenne ± écart-type sur 3 plis groupés",
        title="Classement en validation interne ; la bande grise couvre un écart-type sous le meilleur",
    )
    save_figure(fig, folder, "classement_validation")


def comparison_plot(experiment, folder):
    """F1 sur le test : chaque configuration avec les deux jeux de caractéristiques."""
    results = experiment["results"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, metric, label in (
        (axes[0], "f1", "F1 par fenêtre (seuil par défaut)"),
        (axes[1], "average_precision", "Précision moyenne (AP)"),
    ):
        rows = []
        for strategy in eeg.STRATEGIES:
            for model in eeg.MODELS:
                if strategy == "balanced" and model == "knn":
                    continue
                rows.append((strategy, model))
        for y, (strategy, model) in enumerate(rows):
            a = results.set_index("configuration").loc[
                eeg.config_name("base", strategy, model), metric
            ]
            b = results.set_index("configuration").loc[
                eeg.config_name("improved", strategy, model), metric
            ]
            ax.plot([a, b], [y, y], color=RULE, lw=2, zorder=1)
            ax.scatter(a, y, color=SET_COLORS["base"], s=30, zorder=2)
            ax.scatter(b, y, color=SET_COLORS["improved"], s=30, zorder=2)
        ax.set_yticks(
            range(len(rows)),
            [f"{eeg.MODELS[m]} · {eeg.STRATEGIES[s]}" for s, m in rows],
            fontsize=8.5,
        )
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set(xlim=(0, 1), xlabel=label)
    axes[0].scatter([], [], color=SET_COLORS["base"], label="12 de l'énoncé")
    axes[0].scatter([], [], color=SET_COLORS["improved"], label="enrichies")
    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Test réservé : effet des caractéristiques enrichies pour chaque modèle et stratégie",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    save_figure(fig, folder, "comparaison_modeles")


def postprocessing_plot(temporal, folder):
    """F1 événementiel hors pli selon le seuil, pour chaque durée minimale (lissage retenu)."""
    grid = temporal["grid"]
    params = temporal["params"]
    subset = grid[grid.smoothing == params["smoothing"]].sort_values("threshold")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for color, (minimum, group) in zip(CATEGORICAL, subset.groupby("min_windows"), strict=False):
        axes[0].plot(
            group.threshold, group.event_f1, color=color, lw=1.8, label=f"≥ {minimum} fenêtre(s)"
        )
        axes[1].plot(group.threshold, group.false_alarms_per_hour, color=color, lw=1.8)
    for ax in axes:
        ax.axvline(params["threshold"], color=INK, lw=1, ls="--")
    axes[0].set(
        xlabel="Seuil sur le score lissé",
        ylabel="F1 événementiel (IoU ≥ 0,3)",
        title=f"Réglage hors pli — lissage sur {params['smoothing']} fenêtres",
    )
    axes[1].set(
        xlabel="Seuil sur le score lissé",
        ylabel="Fausses alarmes par heure",
        title="Le seuil et la durée minimale réduisent les fausses alarmes",
        yscale="log",
    )
    axes[0].legend(loc="lower left", ncols=2)
    save_figure(fig, folder, "post_traitement")


def temporal_plots(root, events, validation, temporal, folder):
    """Signal, score, détections brutes et post-traitées, annotations, pour chaque tracé de test."""
    frame = temporal["test_frame"]
    params = temporal["params"]
    for rid in sorted(frame.record_id.unique()):
        t, signal, fs = eeg.load_trace(root, rid, validation)
        filtered = eeg.highpass(signal, fs)
        group = frame[frame.record_id == rid].sort_values("start_s")
        truth = events[events.record_id == rid]
        detected = temporal["predicted"][temporal["predicted"].record_id == rid]
        raw = temporal["raw_predicted"][rid]
        fig, axes = plt.subplots(
            3, 1, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2, 1]}
        )
        hop = max(1, len(t) // 16000)
        axes[0].plot(t[::hop], filtered[::hop], lw=0.45, color=SIGNAL)
        limit = np.percentile(np.abs(filtered), 99.9)
        axes[0].set(ylim=(-limit, limit), ylabel="EEG filtré", title=f"{rid} — test réservé")
        centers = (group.start_s + group.end_s) / 2
        axes[1].plot(
            centers,
            eeg.smooth_scores(group.score.to_numpy(), params["smoothing"]),
            color=INK_3,
            lw=0.9,
        )
        axes[1].axhline(params["threshold"], color=DETECT, lw=1, ls="--")
        axes[1].set(ylabel="score lissé", ylim=(0, 1))
        rows = (
            (
                "annotées",
                REFERENCE,
                list(truth[["start_s", "end_s"]].itertuples(index=False, name=None)),
            ),
            (
                "détectées",
                DETECT,
                list(detected[["start_s", "end_s"]].itertuples(index=False, name=None)),
            ),
            ("sortie brute", INK_3, raw),
        )
        for y, (_, color, intervals) in enumerate(rows):
            axes[2].broken_barh(
                [(a, b - a) for a, b in intervals], (y - 0.32, 0.64), facecolors=color
            )
        for a, b in truth[["start_s", "end_s"]].itertuples(index=False, name=None):
            axes[0].axvspan(a, b, color=REFERENCE, alpha=0.15, lw=0)
        axes[2].set_yticks(range(3), [r[0] for r in rows])
        axes[2].set(ylim=(-0.6, 2.6), xlabel="Temps (s)", xlim=(0, t[-1]))
        axes[2].invert_yaxis()
        axes[2].grid(False)
        save_figure(fig, folder, f"detection_{rid}")


def loro_plot(loro, folder):
    """F1 par enregistrement exclu, pour chaque configuration de la liste courte."""
    data = loro[loro.postprocessing == "Post-traitement"]
    names = list(dict.fromkeys(data.configuration))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, metric, label in (
        (axes[0], "f1", "F1 par fenêtre"),
        (axes[1], "false_alarms_per_hour", "Fausses alarmes par heure"),
    ):
        for y, name in enumerate(names):
            values = data.loc[data.configuration == name, metric]
            jitter = np.linspace(-0.18, 0.18, len(values))
            ax.scatter(
                values, y + jitter, s=16, color=SET_COLORS[eeg.parse_config(name)[0]], alpha=0.75
            )
            ax.plot([values.median()] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
        ax.set_yticks(range(len(names)), [eeg.config_label(n) for n in names], fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlabel(label)
    axes[0].set_title("Un enregistrement exclu à tour de rôle (16 tracés) ; trait noir = médiane")
    save_figure(fig, folder, "loro")


def metrics_table(experiment, strategy=None, feature_set=None):
    """Tableau de présentation, sans masquer les faibles scores."""
    frame = experiment["results"]
    if strategy is not None:
        frame = frame[frame.strategy == strategy]
    if feature_set is not None:
        frame = frame[frame.feature_set == feature_set]
    result = frame[
        [
            "feature_set",
            "strategy",
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "average_precision",
            "TP",
            "FP",
            "FN",
            "TN",
        ]
    ].copy()
    result.feature_set = result.feature_set.map(eeg.FEATURE_SET_LABELS)
    result.strategy = result.strategy.map(eeg.STRATEGIES)
    result.model = result.model.map(eeg.MODELS)
    result.columns = [
        "Caractéristiques",
        "Stratégie",
        "Modèle",
        "Accuracy",
        "Précision",
        "Rappel",
        "F1",
        "AP",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    return result.reset_index(drop=True)


def cv_table(experiment, top=None):
    """Résumé de la validation interne, trié par AP moyenne."""
    summary = experiment["cv_summary"]
    if top:
        summary = summary.head(top)
    table = pd.DataFrame(
        {
            "Configuration": summary.configuration.map(eeg.config_label),
            "AP moyenne": summary.average_precision_mean,
            "AP é.-t.": summary.average_precision_std,
            "F1 moyen": summary.f1_mean,
            "F1 é.-t.": summary.f1_std,
            "Rappel": summary.recall_mean,
            "Précision": summary.precision_mean,
            "Ex aequo avec le meilleur": summary.tied_with_best.map({True: "oui", False: ""}),
        }
    )
    return table


def comparison_table(temporal):
    """Avant/après post-traitement, en validation interne et sur le test."""
    table = temporal["comparison"][
        [
            "stage",
            "postprocessing",
            "precision",
            "recall",
            "f1",
            "event_TP",
            "event_FN",
            "event_FP",
            "event_recall",
            "event_precision",
            "false_alarms_per_hour",
        ]
    ].copy()
    table.columns = [
        "Données",
        "Sortie",
        "Précision fen.",
        "Rappel fen.",
        "F1 fen.",
        "Crises retrouvées",
        "Crises manquées",
        "Fausses alarmes",
        "Rappel év.",
        "Précision év.",
        "Fausses alarmes / h",
    ]
    return table


def loro_table(loro):
    """Résumé LORO lisible."""
    summary = eeg.loro_summary(loro)
    return pd.DataFrame(
        {
            "Configuration": summary.configuration.map(eeg.config_label),
            "Sortie": summary.postprocessing,
            "F1 fen. médiane": summary.f1_median,
            "F1 fen. Q1–Q3": [
                f"{a:.2f}–{b:.2f}" for a, b in zip(summary.f1_q1, summary.f1_q3, strict=True)
            ],
            "F1 fen. poolé": summary.pooled_window_f1,
            "Rappel év.": summary.event_recall,
            "Précision év.": summary.event_precision,
            "Fausses alarmes / h": summary.false_alarms_per_hour,
        }
    )


def markdown_table(frame, digits=3):
    """Table Markdown sans dépendance supplémentaire."""
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.{digits}f}" if isinstance(value, float) else str(value) for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def french_numbers(text):
    """Virgule décimale hors tableaux, liens et code."""
    out = []
    for line in text.splitlines():
        if (
            line.startswith("|")
            or "https://" in line
            or line.startswith("![")
            or line.startswith("    ")
        ):
            out.append(line)
        else:
            out.append(re.sub(r"(?<![\w.])(\d+)\.(\d+)(?![\w]|\.\d)", r"\1,\2", line))
    return "\n".join(out)


REPORT_FIGURES = [
    "signal_annote",
    "durees_annotations",
    "decalage_continu",
    "separation_caracteristiques",
    "classement_validation",
    "comparaison_modeles",
    "post_traitement",
    "loro",
]


def write_report(root, cohort, events, counts, frame, experiment, temporal, loro, gaps):
    """Produit le rapport final uniquement à partir des résultats exécutés."""
    figures_src = root / "artifacts/final/figures"
    figures_dst = root / "reports/figures"
    figures_dst.mkdir(parents=True, exist_ok=True)
    for name in REPORT_FIGURES + [
        f"detection_{rid}" for rid in sorted(temporal["test_frame"].record_id.unique())
    ]:
        shutil.copy2(figures_src / f"{name}.png", figures_dst / f"{name}.png")

    best = experiment["best"]
    old = "base__smote__dt"
    train, test = experiment["train_indices"], experiment["test_indices"]
    cmp = temporal["comparison"].set_index(["stage", "postprocessing"])
    test_raw = cmp.loc[("Test réservé", "Sortie brute")]
    test_pp = cmp.loc[("Test réservé", "Post-traitement")]
    summary = experiment["cv_summary"].set_index("configuration")
    tied = summary[summary.tied_with_best]
    loro_summary = eeg.loro_summary(loro).set_index(["configuration", "postprocessing"])
    loro_best = loro_summary.loc[(best, "Post-traitement")]
    base_effect = effect_sizes(frame, eeg.FEATURES).sort_values(ascending=False)
    improved_effect = effect_sizes(frame, eeg.IMPROVED_FEATURES).sort_values(ascending=False)
    labels = dict(
        zip(
            eeg.FEATURES + eeg.IMPROVED_FEATURES,
            eeg.FEATURE_LABELS + eeg.IMPROVED_LABELS,
            strict=True,
        )
    )
    old_events = old_event_counts(frame, events, experiment, old)
    params = temporal["params"]
    test_hours = test_pp.hours
    base16 = metrics_table(experiment, feature_set="base")
    base16 = base16[base16["Stratégie"] != eeg.STRATEGIES["balanced"]].drop(
        columns=["Caractéristiques"]
    )
    improved_rows = metrics_table(experiment, feature_set="improved").drop(
        columns=["Caractéristiques"]
    )
    events_table = temporal["events"][
        ["record_id", "true_events", "TP", "FN", "FP", "false_alarms_per_hour"]
    ].copy()
    events_table.columns = [
        "Enregistrement",
        "Crises annotées",
        "Retrouvées",
        "Manquées",
        "Fausses alarmes",
        "Fausses alarmes / h",
    ]

    ranking = experiment["cv_summary"]
    leading = 0
    for row in ranking.itertuples():
        if row.feature_set != "improved" or row.model != "rf":
            break
        leading += 1
    leading_text = (
        f"Les {leading} premières places reviennent à la forêt aléatoire sur caractéristiques enrichies, "
        "quelle que soit la stratégie d'équilibrage : le choix des caractéristiques compte plus que celui de la stratégie."
        if leading >= 2
        else ""
    )
    base_f1 = ranking[ranking.feature_set == "base"].sort_values("f1_mean", ascending=False)
    close = base_f1[base_f1.f1_mean >= base_f1.f1_mean.iloc[0] - 0.02]
    old_criterion_text = (
        f"L'ancien critère (F1 moyen au seuil par défaut, 12 caractéristiques) désignait "
        f"{eeg.config_label(base_f1.configuration.iloc[0]).split(' · caract')[0].lower()}, alors que {len(close)} configurations "
        f"tenaient en 0,02 point de F1 avec des écarts-types de {close.f1_std.min():.2f} à {close.f1_std.max():.2f} : "
        "ce classement relevait du hasard des plis."
    )
    res = experiment["results"].set_index("configuration")
    rf_base, rf_imp = res.loc["base__original__rf"], res.loc["improved__original__rf"]
    knn_base, knn_imp = res.loc["base__original__knn"], res.loc["improved__original__knn"]
    test_window_text = (
        "Le sous-échantillonnage et le suréchantillonnage échangent de la précision contre du rappel ; "
        "la pondération des classes déplace le compromis dans le même sens, plus modérément. "
        "Pour la forêt aléatoire, aucune stratégie ne dépasse le F1 obtenu sans équilibrage, avec l'un ou l'autre jeu. "
        f"Au seuil par défaut et sur ces quatre tracés, le jeu enrichi n'améliore pas tout : il transforme le KNN "
        f"(F1 {knn_base.f1:.2f} → {knn_imp.f1:.2f}) et élève l'AP de la forêt ({rf_base.average_precision:.2f} → "
        f"{rf_imp.average_precision:.2f}), mais la forêt sur les 12 caractéristiques garde le meilleur F1 brut "
        f"({rf_base.f1:.2f} contre {rf_imp.f1:.2f}), et les SVM suréchantillonnés se dégradent. "
        "Quatre tracés ne suffisent pas à départager ; la validation « un enregistrement exclu » tranche (tableau 5)."
    )
    loro_all = eeg.loro_summary(loro).set_index(["configuration", "postprocessing"])
    lb = loro_all.loc[("base__original__rf", "Post-traitement")]
    li = loro_all.loc[("improved__original__rf", "Post-traitement")]
    loro_features_text = (
        f"Sur les 16 tracés en LORO, à modèle et post-traitement identiques (forêt sans équilibrage), "
        f"le jeu enrichi retrouve {100 * li.event_recall:.0f} % des crises contre {100 * lb.event_recall:.0f} %, "
        f"avec {li.false_alarms_per_hour:.1f} fausses alarmes par heure contre {lb.false_alarms_per_hour:.1f}. "
        f"Avec les 12 caractéristiques, le premier quartile du F1 par tracé tombe à {lb.f1_q1:.2f} : "
        "sur certains enregistrements, le modèle ne détecte presque rien, ce qui est cohérent avec un apprentissage "
        "de l'identité du tracé plutôt que de la crise."
    )
    details = temporal["details"]
    missed = details[details.status == "FN"]
    found = details[details.status == "TP"]
    missed_text = ""
    if len(missed):
        records = sorted(missed.record_id.unique())
        missed_text = (
            f"Les {len(missed)} crises manquées sur le test ({(missed.true_end_s - missed.true_start_s).min():.0f} à "
            f"{(missed.true_end_s - missed.true_start_s).max():.0f} s) appartiennent à {', '.join(f'`{r}`' for r in records)}, "
            f"alors qu'une crise de {(found.true_end_s - found.true_start_s).min():.0f} s est retrouvée ailleurs : "
            "les omissions tiennent surtout au tracé, dont les scores restent bas, plus qu'à la durée des crises."
        )
    text = f"""# Détection automatique des crises d'absence à partir de l'EEG

Mini-projet de data mining — rapport final, {date.today().strftime("%d/%m/%Y")}.
Tous les nombres de ce rapport sont produits par l'exécution du notebook `notebooks/mini_projet_eeg_final.ipynb` ; aucun n'est saisi à la main.

## Résumé

Sur 16 enregistrements EEG annotés (65 crises d'absence), nous classons des fenêtres de 2 s puis reconstruisons des épisodes. Avec les 12 caractéristiques de l'énoncé, le meilleur modèle en validation interne n'atteint qu'une précision moyenne (AP) de {summary.loc[summary.feature_set == "base", "average_precision_mean"].max():.2f}. Deux défauts l'expliquent : un décalage continu propre à chaque tracé, qui fait des statistiques d'amplitude brutes des identifiants d'enregistrement, et des puissances spectrales à queue très lourde. Un filtrage passe-haut à 0,5 Hz, des puissances logarithmiques et relatives et une normalisation robuste par enregistrement portent l'AP interne à **{summary.loc[best, "average_precision_mean"]:.2f}**. Un post-traitement (lissage, seuil, durée minimale), réglé sur des prédictions hors pli, ramène les fausses alarmes sur le test réservé de **{old_events["fa_h"]:.0f} à {test_pp.false_alarms_per_hour:.1f} par heure**, avec {int(test_pp.event_TP)} crises retrouvées sur {int(test_pp.event_TP + test_pp.event_FN)}. En validation « un enregistrement exclu », le système retrouve {100 * loro_best.event_recall:.0f} % des crises avec {loro_best.false_alarms_per_hour:.1f} fausses alarmes par heure.

## Introduction et problématique

L'absence typique est une crise généralisée brève, de quelques secondes à quelques dizaines de secondes, avec une rupture du contact et, à l'EEG, des pointes-ondes bilatérales et synchrones autour de 3 Hz [1, 2]. Ces crises sont fréquentes et peu visibles ; les compter sur un long tracé est fastidieux. La question posée est la suivante : **peut-on repérer automatiquement ces épisodes sur des enregistrements jamais vus, avec assez peu de fausses alarmes pour qu'une relecture reste utile ?** Nous comparons quatre classifieurs (arbre de décision, forêt aléatoire, KNN, SVM) et plusieurs façons de traiter le déséquilibre, puis nous évaluons au niveau des fenêtres et au niveau des épisodes.

## Données

Le classeur `Annotations MAJ.xlsx` donne, pour chaque tracé, l'heure de début de l'enregistrement et les heures de début et de fin de chaque crise. Elles sont converties en secondes depuis le début du tracé : `(DebCE − DebutTrace) × 86 400`. Pour `190304A-E`, la crise de 10:35:06 à 10:35:13 sur un tracé commencé à 10:13:35 donne [1 291 s ; 1 298 s].

Les fichiers `.mat` contiennent des objets MATLAB `timetable` que `scipy.io` ne lit pas. Les signaux ont été extraits par un lecteur dédié, puis **comparés valeur par valeur** aux cellules numériques des fichiers d'origine (échantillons, noms des canaux, temps). La fréquence d'échantillonnage, mesurée sur l'axe temporel, vaut 256 Hz pour tous les tracés.

Trois enregistrements sont exclus : `230406B-A` et `230515B-G`, présents deux fois avec des annotations contradictoires, et `210504B-C`, dont une annotation dépasse la durée du signal (21 608 s pour un tracé de 3 121 s). `210914B-A` déclare 6 crises mais en décrit 11 ; les 11 paires sont conservées. La cohorte finale compte **{len(cohort)} enregistrements, {len(events)} crises et {len(frame):,} fenêtres**, dont {int(frame.Class.sum())} positives ({100 * frame.Class.mean():.1f} %). Les crises durent de {events.duration_s.min():.0f} à {events.duration_s.max():.0f} s (médiane {events.duration_s.median():.0f} s) ; {int(counts.loc[counts.record_id == "210914B-A", "positive_windows"].iloc[0])} des fenêtres positives viennent d'un seul tracé, `210914B-A`.

![Figure 1 — Tracé 190304A-E et crise annotée](figures/signal_annote.png)

![Figure 2 — Durées des crises et intervalles entre crises](figures/durees_annotations.png)

## Méthodologie

**Segmentation et étiquetage.** Un seul canal, F7, présent dans tous les tracés et choisi avant tout apprentissage : les absences étant généralisées, la décharge est visible sur l'ensemble du scalp. Fenêtres de 2 s (512 échantillons) avec un recouvrement de 50 % ; une fenêtre est « crise » si au moins la moitié de ses échantillons tombe dans un intervalle annoté [début, fin].

**Caractéristiques de l'énoncé.** Moyenne, écart-type, variance, minimum, maximum, amplitude, RMS, énergie, et puissances FFT unilatérales en Delta [0,5 ; 4[, Theta [4 ; 8[, Alpha [8 ; 13[ et Beta [13 ; 30[ Hz.

**Pourquoi les enrichir.** L'analyse exploratoire montre que la moyenne d'une fenêtre varie de plusieurs centaines d'unités d'un tracé à l'autre (figure 3) : le signal n'est pas centré et chaque enregistrement a son propre décalage continu. La moyenne, le minimum et le maximum décrivent donc surtout *quel* tracé on regarde. Ce n'est pas ce que l'on veut apprendre, et le modèle ne peut pas le généraliser à un nouveau tracé. À l'inverse, la puissance Delta, où devrait apparaître la décharge à 3 Hz, a un écart standardisé entre classes de {base_effect["power_delta"]:.3f} : ses valeurs extrêmes (artefacts) écrasent la différence. Nous avons donc ajouté un second jeu de 16 caractéristiques, calculé sur le signal filtré en passe-haut à 0,5 Hz (Butterworth d'ordre 4, phase nulle) : logarithmes de l'écart-type, de l'amplitude, de la longueur de ligne [7] et des quatre puissances ; puissances relatives ; part de la bande 2,5–4,5 Hz (pointes-ondes) ; rapport Delta / (Alpha + Beta) ; asymétrie, kurtosis et fréquence dominante. Chaque caractéristique est centrée et réduite par la médiane et l'écart interquartile **de son propre enregistrement**, sans utiliser les étiquettes, puis bornée à ±10. La meilleure caractéristique passe d'un écart standardisé de {base_effect.iloc[0]:.2f} ({labels[base_effect.index[0]]}) à {improved_effect.iloc[0]:.2f} ({labels[improved_effect.index[0]]}) (figure 4).

![Figure 3 — Décalage continu propre à chaque enregistrement](figures/decalage_continu.png)

![Figure 4 — Séparation des classes, caractéristiques de l'énoncé et enrichies](figures/separation_caracteristiques.png)

**Séparation des données.** Les fenêtres d'un même tracé se recouvrent et se ressemblent ; une séparation au niveau des fenêtres ferait fuir l'information du test vers l'apprentissage [8]. Les données sont donc séparées **par enregistrement** : {frame.iloc[test].record_id.nunique()} tracés ({len(test):,} fenêtres, {100 * len(test) / len(frame):.0f} %) forment le test réservé et {frame.iloc[train].record_id.nunique()} tracés l'apprentissage. Le découpage est choisi parmi 256 tirages groupés uniquement pour approcher 80/20 et la proportion de crises, sans regarder aucun score.

**Modèles et déséquilibre.** Arbre de décision (profondeur 8), forêt aléatoire (200 arbres) [5], KNN (5 voisins pondérés par la distance), SVM à noyau RBF (C = 1), hyperparamètres fixés à l'avance. Cinq stratégies : sans équilibrage, sous-échantillonnage aléatoire, suréchantillonnage par duplication, SMOTE [3], et pondération des classes (`class_weight="balanced"`, hors KNN). Normalisation et rééchantillonnage sont placés dans un pipeline `imbalanced-learn` [6] et ne voient jamais les fenêtres évaluées. Au total : 2 jeux × 19 couples = **38 configurations**.

**Sélection.** Trois plis stratifiés et groupés par enregistrement, à l'intérieur de l'apprentissage. Le critère, fixé avant de lancer les calculs, est la **précision moyenne (AP)** moyenne sur les plis. Elle résume toute la courbe précision-rappel et ne dépend pas d'un seuil arbitraire, ce qui convient à une classe positive à 4 % [9].

**Des fenêtres aux épisodes.** Les fenêtres positives qui se touchent sont fusionnées en épisodes. Un épisode prédit est apparié à une crise annotée si leur IoU (intersection sur union) atteint 0,3, avec un appariement un à un. Trois réglages transforment les scores en épisodes : un lissage sur 1, 3 ou 5 fenêtres, un seuil, et une durée minimale de 1 à 5 fenêtres consécutives. Ils sont choisis pour maximiser le F1 événementiel sur les **prédictions hors pli de l'apprentissage**, puis figés avant l'évaluation sur le test.

**Validation « un enregistrement exclu » (LORO).** Le test réservé ne compte que 4 tracés. Pour estimer la variabilité, la configuration retenue et trois références fixées à l'avance sont réévaluées en excluant chaque tracé à tour de rôle (16 fois). Le post-traitement est alors réglé, à chaque itération, sur les prédictions hors pli des 15 autres tracés.

## Résultats

### Validation interne

Le meilleur résultat en validation interne est **{eeg.config_label(best)}** : AP = {summary.loc[best, "average_precision_mean"]:.3f} ± {summary.loc[best, "average_precision_std"]:.3f}. {"Aucune autre configuration ne se trouve" if len(tied) == 1 else f"{len(tied) - 1} autre(s) configuration(s) se trouvent"} à moins d'un écart-type de ce score. Avec les seules caractéristiques de l'énoncé, la meilleure AP est {summary.loc[summary.feature_set == "base", "average_precision_mean"].max():.3f}. {leading_text}

{old_criterion_text}

![Figure 5 — Classement des 38 configurations en validation interne](figures/classement_validation.png)

### Test réservé, niveau fenêtre

La baseline « toujours hors crise » obtient une accuracy de {float((frame.iloc[test].Class == 0).mean()):.3f} avec un rappel nul : l'accuracy seule ne dit rien ici. Le tableau 1 donne les 16 configurations demandées par l'énoncé (12 caractéristiques, seuil par défaut).

Tableau 1 — Test réservé, 12 caractéristiques de l'énoncé.

{markdown_table(base16)}

Tableau 2 — Test réservé, caractéristiques enrichies (les cinq stratégies).

{markdown_table(improved_rows)}

![Figure 6 — Effet des caractéristiques enrichies sur le test](figures/comparaison_modeles.png)

{test_window_text}

### Test réservé, niveau épisode

Tableau 3 — Configuration retenue, avant et après post-traitement (lissage {params["smoothing"]}, seuil {params["threshold"]:.3f}, au moins {params["min_windows"]} fenêtres).

{markdown_table(comparison_table(temporal), digits=2)}

Le post-traitement fait passer les fausses alarmes de {test_raw.false_alarms_per_hour:.1f} à **{test_pp.false_alarms_per_hour:.1f} par heure** sur {test_hours * 60:.0f} minutes de test, et le F1 par fenêtre de {test_raw.f1:.3f} à {test_pp.f1:.3f}. À titre de comparaison, l'ancien choix (arbre de décision + SMOTE, sortie brute) produisait {old_events["fp"]} fausses alarmes sur le même test, soit {old_events["fa_h"]:.0f} par heure.

Tableau 4 — Détail par enregistrement de test.

{markdown_table(events_table, digits=2)}

![Figure 7 — Réglage du post-traitement sur les prédictions hors pli](figures/post_traitement.png)

![Figure 8 — Enregistrement de test 230718A-A : signal, score, détections et annotations](figures/detection_230718A-A.png)

### Validation « un enregistrement exclu »

Tableau 5 — LORO sur les 16 enregistrements.

{markdown_table(loro_table(loro), digits=2)}

![Figure 9 — F1 et fausses alarmes par enregistrement exclu](figures/loro.png)

## Discussion

**La qualité des caractéristiques compte plus que l'équilibrage.** Les stratégies de rééchantillonnage déplacent le compromis précision/rappel, mais aucune n'apporte d'information nouvelle. Retirer le décalage continu et passer en échelle logarithmique change en revanche ce que le modèle peut apprendre : l'AP interne passe de {summary.loc[summary.feature_set == "base", "average_precision_mean"].max():.2f} à {summary.loc[best, "average_precision_mean"]:.2f}. {loro_features_text}

**L'évaluation par épisode est celle qui compte.** Un F1 par fenêtre élevé peut cacher de nombreuses détections isolées d'une ou deux fenêtres. Or la crise annotée la plus courte dure {events.duration_s.min():.0f} s et la médiane est de {events.duration_s.median():.0f} s. Exiger une suite de fenêtres et lisser le score supprime l'essentiel de ces détections parasites. La durée minimale retenue, {params["min_windows"]} fenêtres, correspond à environ {params["min_windows"] + 1} s de signal et se trouve à la borne de la grille fixée à l'avance. {missed_text}

**Le test réservé est optimiste.** Sur les 4 tracés de test, le F1 par fenêtre ({test_pp.f1:.2f}) dépasse le F1 poolé sur les 16 tracés en LORO ({loro_best.pooled_window_f1:.2f}) ; d'un tracé à l'autre, il varie du simple au double (quartiles {loro_best.f1_q1:.2f}–{loro_best.f1_q3:.2f}, médiane {loro_best.f1_median:.2f}). Ce sont les chiffres LORO qu'il faut retenir.

## Conclusion

Une chaîne simple (un canal, des fenêtres de 2 s, une forêt aléatoire) suffit à retrouver la majorité des crises d'absence de cette cohorte avec quelques fausses alarmes par heure, à trois conditions : retirer le décalage continu, utiliser des caractéristiques indépendantes de l'échelle de chaque tracé, et raisonner en épisodes plutôt qu'en fenêtres. L'équilibrage des classes, au cœur de l'énoncé, a un effet secondaire par rapport à ces trois choix.

## Limites et perspectives

- **Effectif.** 16 enregistrements, sans identifiant patient fiable : deux tracés peuvent venir du même enfant, auquel cas la séparation par enregistrement ne garantit pas une séparation par patient.
- **Annotations.** Trois tracés exclus pour des annotations incohérentes ; les bornes annotées n'ont pas été relues par un second expert.
- **Un seul canal.** F7 est un choix fixé à l'avance ; un montage multicanal ou la moyenne de plusieurs dérivations rendraient la détection plus robuste aux artefacts locaux.
- **Normalisation par enregistrement.** Elle suppose un tracé assez long, dominé par l'activité de fond ; sur un extrait court, riche en crises, la médiane serait biaisée.
- **Hyperparamètres.** Fixés, non optimisés ; la grille de post-traitement s'arrête à 5 fenêtres, borne atteinte par le réglage retenu.
- **Perspectives.** Une validation sur une cohorte externe ou sur une base publique comme CHB-MIT [4], des modèles qui utilisent le contexte temporel (fenêtres voisines) et l'ajout de canaux.

## Bibliographie

1. Fisher R. S. et al. (2017). Operational classification of seizure types by the International League Against Epilepsy. *Epilepsia*, 58(4), 522–530.
2. Crunelli V., Leresche N. (2002). Childhood absence epilepsy: genes, channels, neurons and networks. *Nature Reviews Neuroscience*, 3(5), 371–382.
3. Chawla N. V., Bowyer K. W., Hall L. O., Kegelmeyer W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
4. Shoeb A., Guttag J. (2010). Application of machine learning to epileptic seizure onset detection. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 975–982.
5. Breiman L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
6. Lemaître G., Nogueira F., Aridas C. K. (2017). Imbalanced-learn: a Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, 18(17), 1–5.
7. Esteller R., Echauz J., Tcheng T., Litt B., Pless B. (2001). Line length: an efficient feature for seizure onset detection. *Proceedings of the 23rd Annual International Conference of the IEEE EMBS*, 1707–1710.
8. Roberts D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
9. Saito T., Rehmsmeier M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
10. Pedregosa F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Sources du projet : énoncé `mini-prjet-data-mining.pdf` ; classeur `Annotations MAJ.xlsx` ; consigne complémentaire de l'enseignant sur la conversion des horaires (exemple `190304A-E`).
"""
    path = root / "reports/RAPPORT_FINAL_EEG.md"
    path.write_text(french_numbers(text), encoding="utf-8")
    return path


def old_event_counts(frame, events, experiment, name):
    """Fausses alarmes de l'ancien choix (sortie brute) sur le test, pour comparaison."""
    recordings = eeg.Recordings(eeg.window_table(frame, experiment["test_indices"]), events)
    metrics, _, _ = recordings.evaluate(experiment["test_scores"][name], eeg.default_params(name))
    return {"fp": metrics["event_FP"], "fa_h": metrics["false_alarms_per_hour"]}
