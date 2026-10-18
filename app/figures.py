"""Figures Plotly (interactives) et Matplotlib (topographie, export PNG) de l'interface.

Un seul gabarit sombre partagé ; les couleurs reprennent les variables de app/static/theme.css.
Code couleur des données, doublé d'une forme : crise annotée = bande verte pleine ;
crise détectée = bande orange à bord pointillé ; erreurs (manquées, fausses alarmes) = rouge.
"""

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

THEME = {
    "bg": "#F5F7FA",
    "surface": "#FFFFFF",
    "raised": "#EEF2F6",
    "hairline": "#E3E8EF",
    "text": "#0F172A",
    "text2": "#475467",
    "muted": "#5E6878",
    "accent": "#0B6280",
    "amber": "#B4480F",
    "green": "#157347",
    "red": "#B42318",
    "trace": "#1F2937",
    "neutral": "#98A2B3",
}
ACCENT, ANNOTATED, DETECTED, ERROR = THEME["accent"], THEME["green"], THEME["amber"], THEME["red"]
INK = THEME["text"]
THRESHOLD = THEME["text2"]


def tint(color, alpha):
    """Couleur du thème en transparence (les aplats dérivent tous de la palette)."""
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


ANNOTATED_FILL = tint(THEME["green"], 0.13)
DETECTED_FILL = tint(THEME["amber"], 0.13)
ACCENT_FILL = tint(THEME["accent"], 0.14)
GRID = THEME["hairline"]
GRID_MINOR = tint(THEME["neutral"], 0.25)
NONE = "rgba(0,0,0,0)"
SERIES = [ACCENT, THEME["amber"], "#6941C6", THEME["green"], "#C11574", THEME["text2"]]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
SYMBOLS = ["circle", "diamond", "square", "triangle-up", "x", "star"]
OUTCOME_COLORS = {
    "Vrai négatif": "#C3CAD5",
    "Vrai positif": THEME["green"],
    "Fausse alarme": THEME["red"],
    "Crise manquée": THEME["red"],
}
OUTCOME_SYMBOLS = {
    "Vrai négatif": "circle",
    "Hors crise": "circle",
    "Vrai positif": "circle",
    "Crise": "diamond",
    "Fausse alarme": "diamond",
    "Crise manquée": "x",
}
SEQUENTIAL = [[0, "#F7FAFC"], [0.35, "#A9D3E1"], [0.75, ACCENT], [1, "#052F3D"]]
MONO = "JetBrains Mono, ui-monospace, monospace"

pio.templates["nuit"] = go.layout.Template(
    layout=dict(
        font=dict(family="Manrope, system-ui, sans-serif", color=THEME["text2"], size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", color=THEME["text"], size=14)),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=THEME["hairline"], linecolor=THEME["hairline"], zerolinecolor=THEME["hairline"],
                   tickfont=dict(family=MONO, size=11), title=dict(font=dict(color=THEME["text2"]))),
        yaxis=dict(gridcolor=THEME["hairline"], linecolor=THEME["hairline"], zerolinecolor=THEME["hairline"],
                   tickfont=dict(family=MONO, size=11), title=dict(font=dict(color=THEME["text2"]))),
        legend=dict(font=dict(color=THEME["text2"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=THEME["raised"], bordercolor=THEME["hairline"],
                        font=dict(family=MONO, color=THEME["text"], size=12)),
        colorscale=dict(sequential=SEQUENTIAL),
        transition=dict(duration=300, easing="cubic-in-out"),
    )
)  # fmt: skip
pio.templates.default = "nuit"


def _layout(fig, height, **kwargs):
    settings = {
        "margin": dict(l=60, r=20, t=40, b=40),
        "hovermode": "x unified",
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    }
    fig.update_layout(height=height, **(settings | kwargs))
    return fig


def _legend_square(fig, label, color, fill):
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(size=12, symbol="square", color=fill, line=dict(color=color, width=2)),
        )
    )  # fmt: skip


def _bands(fig, intervals, kind, label, row=None):
    """Bandes translucides : verte pleine (annotée) ou orange à bord pointillé (détectée)."""
    color, fill = (ANNOTATED, ANNOTATED_FILL) if kind == "annotated" else (DETECTED, DETECTED_FILL)
    for a, b in intervals:
        fig.add_vrect(
            x0=a, x1=b, fillcolor=fill, layer="below",
            line=dict(color=color, width=1.2, dash="dot") if kind == "detected" else dict(width=0),
            row=row, col=1 if row else None,
        )  # fmt: skip
    if intervals:
        _legend_square(fig, label, color, fill)


def multichannel_trace(
    time_values, signals, names, annotated, detected=(), view=None, slider=True, height=None
):
    """Visionneuse EEG : voies empilées, grille temporelle, bandes, curseur de plage en bas."""
    n = len(names)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.004)
    for i, (name, values) in enumerate(zip(names, signals, strict=True), 1):
        label = name.removeprefix("EEG")
        fig.add_trace(
            go.Scattergl(
                x=time_values, y=values, name=label, line=dict(width=0.8, color=THEME["trace"]),
                showlegend=False, hovertemplate=f"<b>{label}</b> %{{y:.1f}}<extra></extra>",
            ),
            row=i, col=1,
        )  # fmt: skip
        fig.update_yaxes(
            title_text=label, title_standoff=6, title_font=dict(family=MONO, size=11, color=THEME["text2"]),
            showticklabels=False, zeroline=False, showgrid=False, row=i, col=1,
        )  # fmt: skip
    # Grille de lecture : repère toutes les 5 s, grille fine chaque seconde.
    fig.update_xaxes(showgrid=True, gridcolor=THEME["hairline"], dtick=5,
                     minor=dict(dtick=1, showgrid=True, gridcolor=GRID_MINOR, griddash="dot"))  # fmt: skip
    _bands(fig, annotated, "annotated", "Crise annotée", row="all")
    _bands(fig, detected, "detected", "Crise détectée", row="all")
    fig.update_xaxes(
        title_text="Temps (s)", row=n, col=1, hoverformat=".2f", range=view,
        rangeslider=dict(visible=slider, thickness=0.07, bgcolor=THEME["surface"], bordercolor=THEME["hairline"], borderwidth=1),
    )  # fmt: skip
    return _layout(
        fig,
        height or max(300, 74 * n + 150),
        margin=dict(l=56, r=16, t=36, b=20),
        showlegend=slider,
    )


def spectrogram_figure(f, t, power, annotated, view=None):
    fig = go.Figure(
        go.Heatmap(
            x=t, y=f, z=power, colorscale=SEQUENTIAL, colorbar=dict(title="dB", thickness=10, outlinewidth=0),
            zmin=np.percentile(power, 5), zmax=np.percentile(power, 99.5),
            hovertemplate="t = %{x:.1f} s<br>f = %{y:.1f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )  # fmt: skip
    for a, b in annotated:
        fig.add_vrect(x0=a, x1=b, line=dict(color=ANNOTATED, width=1.5), fillcolor="rgba(0,0,0,0)")
    if view:
        fig.add_vrect(x0=view[0], x1=view[1], line=dict(color=THEME["text"], width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")  # fmt: skip
    fig.add_hrect(
        y0=2.5,
        y1=4.5,
        line=dict(color=THEME["text2"], width=1, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Temps (s)", showgrid=False)
    fig.update_yaxes(title_text="Fréquence (Hz)", showgrid=False)
    return _layout(fig, 330, hovermode="closest")


def topomaps(names, maps, titles):
    """Cartes du scalp (MNE) sur fond sombre, échelle de couleur commune."""
    import mne

    info = mne.create_info(names, 256.0, ch_types="eeg")
    montage_name = (
        "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"
    )
    info.set_montage(mne.channels.make_standard_montage(montage_name), on_missing="ignore")
    cmap = LinearSegmentedColormap.from_list("nuit", [c for _, c in SEQUENTIAL])
    high = float(max(np.max(m) for m in maps))
    fig, axes = plt.subplots(1, len(maps), figsize=(3.0 * len(maps) + 0.8, 3.0), dpi=120)
    fig.patch.set_facecolor(THEME["surface"])
    axes = np.atleast_1d(axes)
    for ax, values, title in zip(axes, maps, titles, strict=True):
        ax.set_facecolor(THEME["surface"])
        image, _ = mne.viz.plot_topomap(
            values, info, axes=ax, show=False, vlim=(0, high), cmap=cmap, names=names, contours=4
        )
        for line in ax.lines:
            line.set_color(THEME["text2"])
        for text in ax.texts:
            text.set_color(THEME["text"])
            text.set_fontsize(7)
        ax.set_title(title, fontsize=10, color=THEME["text"])
    cbar = fig.colorbar(image, ax=list(axes), shrink=0.75, pad=0.02)
    cbar.set_label("Puissance relative 2,5–4,5 Hz", color=THEME["text2"])
    cbar.ax.tick_params(colors=THEME["text2"])
    cbar.outline.set_edgecolor(THEME["hairline"])
    return fig


def score_timeline(result, annotated, detected, threshold=None, band=None):
    """Score lissé de chaque fenêtre, seuil(s), crises annotées et détectées."""
    scores = result["scores"]
    fig = go.Figure()
    _bands(fig, annotated, "annotated", "Crise annotée")
    for a, b in detected:
        fig.add_shape(
            type="rect", x0=a, x1=b, y0=0, y1=0.1, yref="paper",
            fillcolor=DETECTED_FILL, line=dict(color=DETECTED, width=1.2, dash="dot"),
        )  # fmt: skip
    if detected:
        _legend_square(fig, "Crise détectée", DETECTED, DETECTED_FILL)
    if scores.get("std"):
        # Bande ±2 écarts-types du Monte-Carlo dropout.
        value, spread = np.asarray(scores["value"]), np.asarray(scores["std"])
        fig.add_trace(
            go.Scatter(
                x=np.r_[scores["time"], scores["time"][::-1]],
                y=np.r_[value + 2 * spread, (value - 2 * spread)[::-1]],
                fill="toself", fillcolor=ACCENT_FILL, line=dict(width=0),
                name="Incertitude (±2σ)", hoverinfo="skip",
            )
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=scores["time"], y=scores["value"], name="Score lissé",
            line=dict(color=ACCENT, width=1.4), hovertemplate="%{y:.3f}",
        )
    )  # fmt: skip
    for value, name in ([(threshold, "Seuil")] if threshold is not None else []) + (
        [(band[0], "Seuil haut"), (band[1], "Seuil bas")] if band else []
    ):
        fig.add_hline(
            y=value, line=dict(color=THRESHOLD, dash="dash", width=1),
            annotation_text=f"{name} {value:.2f}", annotation_position="top right",
            annotation_font=dict(family=MONO, color=THEME["text2"], size=11),
        )  # fmt: skip
    fig.update_xaxes(title_text="Temps (s)", hoverformat=".0f")
    fig.update_yaxes(title_text="Probabilité de crise", range=scores["range"])
    # Légende sous le graphique : elle ne chevauche jamais la barre d'outils, même sur écran étroit.
    return _layout(
        fig,
        350,
        legend=dict(orientation="h", x=0, y=-0.22, yanchor="top"),
        margin=dict(l=60, r=20, t=20, b=70),
    )


def duration_histogram(detected_durations, annotated_durations):
    fig = go.Figure()
    if len(annotated_durations):
        fig.add_trace(
            go.Histogram(x=annotated_durations, name="Annotées", xbins=dict(size=2),
                         marker=dict(color=ANNOTATED_FILL, line=dict(color=ANNOTATED, width=1.2)))
        )  # fmt: skip
    if len(detected_durations):
        fig.add_trace(
            go.Histogram(x=detected_durations, name="Détectées", xbins=dict(size=2),
                         marker=dict(color=DETECTED_FILL, line=dict(color=DETECTED, width=1.2),
                                     pattern=dict(shape="/", fgcolor=DETECTED)))
        )  # fmt: skip
    fig.update_layout(barmode="group", bargap=0.15)
    fig.update_xaxes(title_text="Durée (s)")
    fig.update_yaxes(title_text="Nombre de crises", dtick=1)
    return _layout(fig, 290, hovermode="closest")


def timeline_png(result, annotated, detected, record):
    """Frise statique pour l'export (fond sombre, mêmes codes que l'interface)."""
    scores = result["scores"]
    with plt.rc_context({"axes.edgecolor": THEME["hairline"], "xtick.color": THEME["text2"], "ytick.color": THEME["text2"],
                         "axes.labelcolor": THEME["text2"], "text.color": THEME["text"]}):  # fmt: skip
        fig, ax = plt.subplots(figsize=(12, 2.8), dpi=150)
        fig.patch.set_facecolor(THEME["bg"])
        ax.set_facecolor(THEME["bg"])
        for a, b in annotated:
            ax.axvspan(a, b, color=ANNOTATED, alpha=0.18, lw=0)
        for a, b in detected:
            ax.axvspan(
                a, b, ymin=0, ymax=0.1, facecolor="none", edgecolor=DETECTED, hatch="///", lw=1
            )
        ax.plot(scores["time"], scores["value"], color=ACCENT, lw=0.9, label="Score lissé")
        ax.axhline(result["params"]["threshold"], color=THRESHOLD, ls="--", lw=0.8, label="Seuil")
        ax.grid(color=THEME["hairline"], lw=0.5)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Probabilité de crise")
        ax.set_title(f"{record} — vert : crises annotées, hachures orange : crises détectées")
        ax.set_xlim(0, scores["time"][-1])
        legend = ax.legend(
            loc="upper right", fontsize=8, facecolor=THEME["surface"], edgecolor=THEME["hairline"]
        )
        for text in legend.get_texts():
            text.set_color(THEME["text"])
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
    return buffer.getvalue()


def metric_lollipop(table, metric, label, retained):
    """Sucettes horizontales triées ; la meilleure valeur en bleu, le modèle retenu marqué ★."""
    ordered = table.sort_values(metric)
    best = ordered[metric].max()
    colors = [ACCENT if v == best else THEME["neutral"] for v in ordered[metric]]
    fig = go.Figure()
    for y, x, c in zip(ordered["Configuration"], ordered[metric], colors, strict=True):
        fig.add_shape(
            type="line", x0=0, x1=x, y0=y, y1=y, line=dict(color=c, width=2 if c == ACCENT else 1.2)
        )
    fig.add_trace(
        go.Scatter(
            x=ordered[metric], y=ordered["Configuration"], mode="markers", showlegend=False,
            marker=dict(size=[13 if c == retained else 9 for c in ordered["Configuration"]], color=colors,
                        symbol=["star" if c == retained else "circle" for c in ordered["Configuration"]],
                        line=dict(color=THEME["bg"], width=1)),
            hovertemplate="%{y}<br>" + label + " = %{x:.3f}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_yaxes(categoryorder="array", categoryarray=ordered["Configuration"].tolist(),
                     tickfont=dict(family="Manrope, sans-serif", size=11, color=THEME["text2"]), showgrid=False)  # fmt: skip
    fig.update_xaxes(title_text=label, range=[0, 1.02])
    return _layout(
        fig, 110 + 21 * len(table), hovermode="closest", margin=dict(l=20, r=20, t=20, b=40)
    )


def confusion_figure(tn, fp, fn, tp):
    """Matrice de confusion : couleur = part de la ligne, grands nombres lisibles."""
    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    rates = counts / counts.sum(axis=1, keepdims=True)
    text = [[f"{int(counts[i, j]):,}".replace(",", " ") + f"<br><span style='font-size:12px'>{rates[i, j]:.0%}</span>"
             for j in range(2)] for i in range(2)]  # fmt: skip
    fig = go.Figure(
        go.Heatmap(
            z=rates, x=["Prédit : hors crise", "Prédit : crise"], y=["Réel : hors crise", "Réel : crise"],
            colorscale=[[0, THEME["surface"]], [1, tint(THEME["accent"], 0.45)]], zmin=0, zmax=1, showscale=False,
            text=text, texttemplate="%{text}", textfont=dict(family=MONO, size=26, color=THEME["text"]),
            hovertemplate="%{y} · %{x}<extra></extra>", xgap=4, ygap=4,
        )
    )  # fmt: skip
    fig.update_yaxes(
        autorange="reversed", showgrid=False, tickfont=dict(family="Manrope, sans-serif", size=12)
    )
    fig.update_xaxes(
        side="top", showgrid=False, tickfont=dict(family="Manrope, sans-serif", size=12)
    )
    return _layout(fig, 300, hovermode="closest", margin=dict(l=20, r=20, t=40, b=10))


def curves(entries, kind, prevalence):
    """Courbes ROC ou précision-rappel ; `entries` = (nom, x, y, aire)."""
    fig = go.Figure()
    for k, (name, x, y, area) in enumerate(entries):
        fig.add_trace(
            go.Scatter(x=x, y=y, name=f"{name} ({area:.3f})", mode="lines",
                       line=dict(width=1.8, color=SERIES[k % 6], dash=DASHES[k % 6]))
        )  # fmt: skip
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Hasard", mode="lines", line=dict(color=THEME["neutral"], dash="dot")))  # fmt: skip
        fig.update_xaxes(title_text="Taux de faux positifs", range=[0, 1])
        fig.update_yaxes(title_text="Taux de vrais positifs (rappel)", range=[0, 1.02])
    else:
        fig.add_hline(
            y=prevalence, line=dict(color=THEME["neutral"], dash="dot"),
            annotation_text=f"Hasard ({prevalence:.3f})", annotation_position="bottom right",
            annotation_font=dict(color=THEME["text2"]),
        )  # fmt: skip
        fig.update_xaxes(title_text="Rappel", range=[0, 1])
        fig.update_yaxes(title_text="Précision", range=[0, 1.02])
    return _layout(fig, 560, hovermode="closest", margin=dict(l=60, r=20, t=20, b=40),
                   legend=dict(orientation="v", x=0, y=-0.18, yanchor="top", font=dict(size=11)))  # fmt: skip


def projection_scatter(frame, color, colors, explained):
    fig = go.Figure()
    for value, hue in colors.items():
        part = frame[frame[color] == value]
        if part.empty:
            continue
        background = value in ("Hors crise", "Vrai négatif")
        fig.add_trace(
            go.Scattergl(
                x=part.x, y=part.y, mode="markers", name=f"{value} ({len(part)})",
                marker=dict(size=4 if background else 7, symbol=OUTCOME_SYMBOLS.get(value, "circle"), color=hue,
                            opacity=0.45 if background else 0.9),
                customdata=np.c_[part.record_id, part.start_s],
                hovertemplate="%{customdata[0]} · t = %{customdata[1]:.0f} s<extra></extra>",
            )
        )  # fmt: skip
    if explained:
        fig.update_xaxes(title_text=f"Composante 1 ({explained[0]:.0%} de variance)")
        fig.update_yaxes(title_text=f"Composante 2 ({explained[1]:.0%} de variance)")
    else:
        fig.update_xaxes(title_text="t-SNE 1")
        fig.update_yaxes(title_text="t-SNE 2")
    return _layout(fig, 520, hovermode="closest")


def loro_bars(frame):
    fig = go.Figure(
        go.Bar(
            x=frame.record_id, y=frame.f1,
            marker=dict(color=np.where(frame.is_test, ACCENT, THEME["neutral"]),
                        pattern=dict(shape=np.where(frame.is_test, "/", ""), fgcolor=THEME["bg"])),
            customdata=np.c_[frame.event_TP, frame.event_FN, frame.event_FP],
            hovertemplate="%{x}<br>F1 fenêtres = %{y:.2f}<br>crises trouvées %{customdata[0]}, "
            "manquées %{customdata[1]}, fausses alarmes %{customdata[2]}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_yaxes(title_text="F1 des fenêtres", range=[0, 1])
    fig.update_xaxes(
        title_text="Enregistrement exclu de l'apprentissage",
        tickangle=-40,
        tickfont=dict(family=MONO, size=10),
    )
    return _layout(fig, 360, hovermode="closest")


def saliency_figure(times, signals, saliency, channels, annotated):
    """Voies empilées sur fond de saillance : plus c'est lumineux, plus la zone a pesé dans la décision."""
    n = len(channels)
    fig = go.Figure(
        go.Heatmap(
            x=times, y=np.arange(n), z=saliency,
            colorscale=[[0, NONE], [0.35, tint(THEME["accent"], 0.25)], [1, tint(THEME["accent"], 0.85)]],
            zmin=0, zmax=1, showscale=True, colorbar=dict(title="Saillance", thickness=10, outlinewidth=0),
            hovertemplate="%{x:.2f} s · saillance %{z:.2f}<extra></extra>",
        )
    )  # fmt: skip
    scale = 0.42 / (np.percentile(np.abs(signals), 99) or 1)
    for k in range(n):
        fig.add_trace(
            go.Scatter(x=times, y=k + signals[k] * scale, mode="lines", showlegend=False,
                       line=dict(color=THEME["trace"], width=0.8), hoverinfo="skip")
        )  # fmt: skip
    for a, b in annotated:
        if b >= times[0] and a <= times[-1]:
            fig.add_vrect(x0=max(a, times[0]), x1=min(b, times[-1]), line=dict(color=ANNOTATED, width=1.5), fillcolor="rgba(0,0,0,0)")  # fmt: skip
    fig.update_yaxes(tickvals=list(range(n)), ticktext=[c.removeprefix("EEG") for c in channels], autorange="reversed",
                     showgrid=False, zeroline=False, tickfont=dict(family=MONO, size=11))  # fmt: skip
    fig.update_xaxes(title_text="Temps (s)", showgrid=False)
    return _layout(fig, 640, hovermode="closest")


def reliability_figure(entries):
    """`entries` = (nom, probabilité moyenne, fréquence observée, effectifs, Brier)."""
    fig = go.Figure(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite", line=dict(color=THEME["neutral"], dash="dot"))
    )  # fmt: skip
    for k, (name, predicted, observed, counts, brier) in enumerate(entries):
        fig.add_trace(
            go.Scatter(
                x=predicted, y=observed, mode="lines+markers", name=f"{name} (Brier {brier:.3f})",
                line=dict(color=SERIES[k % 6], dash=DASHES[k % 6]),
                marker=dict(size=np.clip(np.sqrt(counts) / 2, 5, 18), symbol=SYMBOLS[k % 6]), customdata=counts,
                hovertemplate="prédit %{x:.2f} · observé %{y:.2f} · %{customdata} fenêtres<extra></extra>",
            )
        )  # fmt: skip
    fig.update_xaxes(title_text="Probabilité prédite (moyenne de la classe)", range=[0, 1])
    fig.update_yaxes(title_text="Fréquence réelle des crises", range=[0, 1.02])
    return _layout(fig, 540, hovermode="closest", margin=dict(l=60, r=20, t=20, b=40),
                   legend=dict(orientation="v", x=0, y=-0.16, yanchor="top"))  # fmt: skip


def training_curves(history, folds):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Perte d'apprentissage (modèle final)", "AP par pli de validation"),
    )
    fig.add_trace(go.Scatter(x=[h["epoch"] for h in history], y=[h["loss"] for h in history], mode="lines+markers",
                             name="Perte", line=dict(color=ACCENT)), row=1, col=1)  # fmt: skip
    fig.add_trace(go.Bar(x=[f"Pli {f['fold']}" for f in folds], y=[f["average_precision"] for f in folds],
                         marker=dict(color=tint(THEME["accent"], 0.25), line=dict(color=ACCENT, width=1.2)), name="AP",
                         text=[f"{f['average_precision']:.2f}" for f in folds], textposition="outside",
                         textfont=dict(family=MONO, color=THEME["text"])), row=1, col=2)  # fmt: skip
    fig.update_annotations(
        font=dict(family="Space Grotesk, sans-serif", color=THEME["text"], size=13)
    )
    fig.update_xaxes(title_text="Époque", row=1, col=1)
    fig.update_yaxes(range=[0, 1.1], row=1, col=2)
    return _layout(
        fig, 340, hovermode="closest", showlegend=False, margin=dict(l=60, r=20, t=70, b=40)
    )


# ---------- Pages pédagogiques ----------

BAND_SHADES = [("Delta", 0.5, 4), ("Thêta", 4, 8), ("Alpha", 8, 13), ("Bêta", 13, 30)]


def excerpt_scale(data):
    """Échelle commune : la même pour deux extraits que l'on veut comparer honnêtement."""
    return 0.45 / (np.percentile(np.abs(data), 99.5) or 1)


def stacked_excerpt(
    time_values, data, names, scale, annotated=(), detected=(), height=300, title=None
):
    """Voies sur un seul axe, décalées, à une échelle imposée (µV par couloir identique d'une figure à l'autre)."""
    fig = go.Figure()
    for a, b in annotated:
        fig.add_vrect(x0=a, x1=b, fillcolor=ANNOTATED_FILL, line_width=0, layer="below")
    for a, b in detected:
        fig.add_vrect(
            x0=a,
            x1=b,
            fillcolor=DETECTED_FILL,
            line=dict(color=DETECTED, width=1.2, dash="dot"),
            layer="below",
        )
    for k, (name, values) in enumerate(zip(names, data, strict=True)):
        label = name.removeprefix("EEG")
        fig.add_trace(
            go.Scatter(x=time_values, y=k + values * scale, mode="lines", name=label, showlegend=False,
                         line=dict(color=THEME["trace"], width=0.8), customdata=values,
                         hovertemplate=f"<b>{label}</b> %{{customdata:.0f}}<extra></extra>")
        )  # fmt: skip
    fig.update_yaxes(tickvals=list(range(len(names))), ticktext=[n.removeprefix("EEG") for n in names],
                     autorange="reversed", showgrid=False, zeroline=False, tickfont=dict(family=MONO, size=10))  # fmt: skip
    span = float(time_values[-1] - time_values[0]) if len(time_values) else 0
    fig.update_xaxes(
        showgrid=True,
        gridcolor=THEME["hairline"],
        dtick=1 if span <= 12 else 2 if span <= 24 else 5,
        range=[float(time_values[0]), float(time_values[-1])] if len(time_values) else None,
        hoverformat=".2f",
        title_text="Temps (s)",
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=13)))
    return _layout(fig, height, margin=dict(l=44, r=10, t=36 if title else 10, b=36))


def spectra_figure(spectra):
    """Spectre moyen des fenêtres en crise et hors crise, bandes de l'énoncé en fond."""
    f = spectra["frequencies"]
    fig = go.Figure()
    for k, (name, low, high) in enumerate(BAND_SHADES):
        fig.add_vrect(x0=low, x1=high, fillcolor=tint(THEME["neutral"], 0.14) if k % 2 == 0 else NONE, line_width=0,
                      layer="below", annotation_text=name, annotation_position="top left",
                      annotation_font=dict(color=THEME["text2"], size=11))  # fmt: skip
    fig.add_vrect(x0=2.5, x1=4.5, fillcolor=tint(THEME["green"], 0.08), line=dict(color=ANNOTATED, width=1, dash="dot"),
                  layer="below")  # fmt: skip
    fig.add_trace(go.Scatter(x=f, y=spectra["normal"], name=f"Hors crise ({spectra['n_normal']} fenêtres)",
                             line=dict(color=THEME["text2"], width=1.6, dash="dash")))  # fmt: skip
    fig.add_trace(go.Scatter(x=f, y=spectra["crisis"], name=f"En crise ({spectra['n_crisis']} fenêtres)",
                             line=dict(color=ANNOTATED, width=2.2)))  # fmt: skip
    fig.update_yaxes(type="log", title_text="Puissance moyenne (échelle log)")
    fig.update_xaxes(title_text="Fréquence (Hz)", range=[0, 30], dtick=2)
    return _layout(fig, 360, hovermode="x unified")


def windows_figure(time_values, signal, starts, coverage, labels, seizure):
    """La règle des 50 % : chaque fenêtre de 2 s (pas de 1 s) et la part de crise qu'elle contient."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06
    )
    fig.add_vrect(
        x0=seizure[0],
        x1=seizure[1],
        fillcolor=ANNOTATED_FILL,
        line_width=0,
        layer="below",
        row="all",
        col=1,
    )
    fig.add_trace(go.Scatter(x=time_values, y=signal, line=dict(color=THEME["trace"], width=0.8), showlegend=False,
                               hovertemplate="%{y:.0f}<extra>F7</extra>"), row=1, col=1)  # fmt: skip
    for k, (t, c, lab) in enumerate(zip(starts, coverage, labels, strict=True)):
        y = k % 2
        color = ANNOTATED if lab else THEME["neutral"]
        fig.add_shape(type="rect", x0=t + 0.04, x1=t + 1.96, y0=y + 0.12, y1=y + 0.88, row=2, col=1,
                      line=dict(color=color, width=1.4), fillcolor=ANNOTATED_FILL if lab else "rgba(0,0,0,0)")  # fmt: skip
        fig.add_annotation(x=t + 1, y=y + 0.5, text=f"{c:.0%}", showarrow=False, row=2, col=1,
                           font=dict(family=MONO, size=11, color=THEME["text"] if lab else THEME["text2"]))  # fmt: skip
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(
        range=[-0.1, 2.1], showticklabels=False, showgrid=False, zeroline=False, row=2, col=1
    )
    fig.update_xaxes(dtick=1, showgrid=True, gridcolor=THEME["hairline"], row=1, col=1)
    fig.update_xaxes(
        dtick=1, showgrid=True, gridcolor=THEME["hairline"], title_text="Temps (s)", row=2, col=1
    )
    return _layout(
        fig, 360, hovermode="closest", showlegend=False, margin=dict(l=20, r=10, t=10, b=40)
    )


def decision_figure(times, scores, above, kept, truth, detected, threshold):
    """Du score aux épisodes : seuil, puis durée minimale, puis épisodes comparés aux annotations."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.46, 0.18, 0.18, 0.18], vertical_spacing=0.035,
                        subplot_titles=("1 · Score lissé et seuil", "2 · Fenêtres au-dessus du seuil",
                                        "3 · Après la durée minimale", "4 · Épisodes détectés et crises annotées"))  # fmt: skip
    fig.add_trace(go.Scatter(x=times, y=scores, line=dict(color=ACCENT, width=1.3), showlegend=False,
                             hovertemplate="%{y:.3f}<extra>score</extra>"), row=1, col=1)  # fmt: skip
    fig.add_hline(y=threshold, line=dict(color=THRESHOLD, dash="dash", width=1), row=1, col=1)
    for row, mask, color in ((2, above, THEME["neutral"]), (3, kept, DETECTED)):
        fig.add_trace(go.Bar(x=times, y=mask.astype(int), marker=dict(color=color, line_width=0), width=1.0,
                             showlegend=False, hoverinfo="skip"), row=row, col=1)  # fmt: skip
    for a, b in truth:
        fig.add_shape(
            type="rect", x0=a, x1=b, y0=0.55, y1=1, fillcolor=ANNOTATED, line_width=0, row=4, col=1
        )
    for a, b in detected:
        fig.add_shape(type="rect", x0=a, x1=b, y0=0, y1=0.45, fillcolor=DETECTED_FILL, line=dict(color=DETECTED, width=1.2),
                      row=4, col=1)  # fmt: skip
    for row in (2, 3, 4):
        fig.update_yaxes(
            range=[0, 1.05], showticklabels=False, showgrid=False, zeroline=False, row=row, col=1
        )
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Temps (s)", row=4, col=1)
    fig.update_annotations(
        font=dict(family="Space Grotesk, sans-serif", size=12, color=THEME["text"]),
        x=0,
        xanchor="left",
    )
    return _layout(
        fig, 520, hovermode="x", showlegend=False, bargap=0, margin=dict(l=56, r=10, t=30, b=40)
    )


def durations_figure(durations):
    median = float(np.median(durations))
    fig = go.Figure(go.Histogram(x=durations, xbins=dict(size=2),
                                 marker=dict(color=ANNOTATED_FILL, line=dict(color=ANNOTATED, width=1.2)),
                                 hovertemplate="%{x} s : %{y} crises<extra></extra>"))  # fmt: skip
    fig.add_vline(x=median, line=dict(color=THEME["text"], dash="dash", width=1),
                  annotation_text=f"médiane {median:.0f} s", annotation_position="top right",
                  annotation_font=dict(family=MONO, color=THEME["text"]))  # fmt: skip
    fig.update_xaxes(title_text="Durée de la crise annotée (s)")
    fig.update_yaxes(title_text="Nombre de crises")
    return _layout(fig, 300, hovermode="closest", showlegend=False)


# Positions approximatives du système 10-20 (nez en haut), en coordonnées du cercle unité.
HEAD_POSITIONS = {
    "Fp1": (-0.3, 0.86), "Fp2": (0.3, 0.86), "F7": (-0.72, 0.5), "F3": (-0.38, 0.45), "Fz": (0, 0.42),
    "F4": (0.38, 0.45), "F8": (0.72, 0.5), "T3": (-0.9, 0), "C3": (-0.45, 0), "Cz": (0, 0), "C4": (0.45, 0),
    "T4": (0.9, 0), "T5": (-0.72, -0.5), "P3": (-0.38, -0.45), "Pz": (0, -0.42), "P4": (0.38, -0.45),
    "T6": (0.72, -0.5), "O1": (-0.3, -0.86), "O2": (0.3, -0.86),
}  # fmt: skip


def head_svg(values, title, low, high):
    """Schéma de la tête : chaque électrode est un disque dont la luminosité suit sa valeur."""
    size, r = 260, 104
    cx = cy = size / 2
    parts = [
        f'<svg viewBox="0 0 {size} {size + 24}" class="head" role="img" aria-label="{title}">',
        f'<polygon points="{cx - 14},{cy - r + 2} {cx},{cy - r - 16} {cx + 14},{cy - r + 2}" class="nose"/>',
        f'<ellipse cx="{cx - r - 6}" cy="{cy}" rx="8" ry="20" class="ear"/><ellipse cx="{cx + r + 6}" cy="{cy}" rx="8" ry="20" class="ear"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" class="skull"/>',
    ]
    for name, (x, y) in HEAD_POSITIONS.items():
        if name not in values:
            continue
        level = float(np.clip((values[name] - low) / ((high - low) or 1), 0, 1))
        px, py = cx + x * r * 0.92, cy - y * r * 0.92
        parts.append(
            f'<g class="el"><title>{name} : {values[name]:.2f}</title><circle cx="{px:.1f}" cy="{py:.1f}" r="15" '
            f'style="fill:{tint(THEME["accent"], round(0.06 + 0.86 * level, 2))}"/>'
            f'<text x="{px:.1f}" y="{py + 4:.1f}" class="{"dark" if level > 0.6 else ""}">{name}</text></g>'
        )
    parts.append(f'<text x="{cx}" y="{size + 18}" class="cap">{title}</text></svg>')
    return "".join(parts)


def viewer_colors():
    """Couleurs du lecteur EEG, dérivées du même thème que les graphiques."""
    return {
        "text": THEME["text"], "text2": THEME["text2"], "muted": THEME["muted"], "trace": THEME["trace"],
        "accent": ACCENT, "grid": tint(THEME["neutral"], 0.45), "gridMinor": tint(THEME["neutral"], 0.16),
        "hair": tint(THEME["neutral"], 0.22), "ann": ANNOTATED, "annFill": ANNOTATED_FILL, "det": DETECTED,
        "detFill": tint(THEME["amber"], 0.09), "detHatch": tint(THEME["amber"], 0.35), "viewFill": tint(THEME["accent"], 0.08),
    }  # fmt: skip
