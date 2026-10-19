"""Interface Streamlit : visualiser, détecter et comparer les modèles (classiques et CNN).

Lancement : `app/lancer.sh` (ou `streamlit run app/streamlit_app.py` depuis la racine).

Présentation seulement : tous les calculs viennent de app/service.py et app/inference.py,
qui appellent les fonctions du notebook. Thème sombre dans .streamlit/config.toml et
app/static/theme.css ; composants HTML construits par les fonctions utilitaires ci-dessous.
"""

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from app import figures, service  # noqa: E402
from app.inference import InputError  # noqa: E402
from src import eeg_project as eeg  # noqa: E402

st.set_page_config(
    page_title="Détection des crises d'absence", page_icon="app/static/logo-icon.svg", layout="wide"
)

STATIC = Path(__file__).resolve().parent / "static"
MODELS = {"rf": "Forêt aléatoire · voie F7", "cnn": "CNN 1D · 19 voies"}
MODELS_SHORT = {"rf": "Forêt · F7", "cnn": "CNN · 19 voies"}
RULES = {"retenu": "Réglage hors pli", "hysteresis": "Hystérésis (exploratoire)"}
RULES_SHORT = {"retenu": "Hors pli", "hysteresis": "Hystérésis"}
STATUS_PILL = {"test": "ok", "apprentissage": "warn", "sans annotation": "", "exclu": "err"}

# Icônes au trait (Lucide, licence ISC), tracées en SVG pour rester hors ligne.
ICONS = {
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
    "percent": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "x": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "inbox": '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
}
STATE_STYLE = {
    "empty": ("inbox", ""),
    "error": ("x", "red"),
    "partial": ("alert", "amber"),
    "info": ("info", ""),
}


# ======================================================================
# Composants réutilisables
# ======================================================================


@st.cache_data
def stylesheet():
    return (STATIC / "theme.css").read_text(encoding="utf-8")


def fmt(value, digits=1, suffix=""):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",") + suffix


def percent(value, digits=0):
    return fmt(100 * value, digits, " %")


def esc(value):
    return html.escape(str(value))


def show(markup, slot=None):
    (slot or st).markdown(markup, unsafe_allow_html=True)


def explain(text):
    st.caption(text)


def icon(name, tone=""):
    return f'<span class="ico {tone}" aria-hidden="true"><svg viewBox="0 0 24 24">{ICONS[name]}</svg></span>'


NUMBER = re.compile(r"\d+(?:,\d+)?")


def counting(text):
    """Nombres qui défilent depuis 0 (CSS) ; la valeur exacte reste lue par les lecteurs d'écran."""
    parts, last = [], 0
    for match in NUMBER.finditer(text):
        parts.append(esc(text[last : match.start()]))
        whole, _, frac = match.group(0).partition(",")
        if len(whole) > 1 and whole.startswith("0"):
            parts.append(esc(match.group(0)))
        else:
            parts.append(f'<span class="cu" style="--to:{int(whole)}"></span>')
            if frac:
                padded = len(frac) == 2 and frac.startswith("0")
                if len(frac) > 2 or (frac.startswith("0") and not padded):
                    parts.append(esc("," + frac))
                else:
                    parts.append(
                        f',<span class="cu{" z2" if padded else ""}" style="--to:{int(frac)}"></span>'
                    )
        last = match.end()
    parts.append(esc(text[last:]))
    return f'<span role="text" aria-label="{esc(text)}"><span aria-hidden="true">{"".join(parts)}</span></span>'


def sparkline(values, color="var(--accent)"):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) < 2:
        return ""
    low, high = values.min(), values.max()
    y = 30 - 28 * (values - low) / ((high - low) or 1)
    x = np.linspace(1, 95, len(values))
    line = " ".join(
        f"{'M' if k == 0 else 'L'}{a:.1f},{b:.1f}"
        for k, (a, b) in enumerate(zip(x, y, strict=True))
    )
    area = f"{line} L{x[-1]:.1f},32 L{x[0]:.1f},32 Z"
    return (
        f'<svg class="spark" viewBox="0 0 96 32" aria-hidden="true"><path class="area" d="{area}"/>'
        f'<path d="{line}" style="stroke:{color}"/></svg>'
    )


def ring(fraction, color="var(--accent)"):
    p = float(np.clip(fraction, 0, 1)) * 100
    return f'<div class="ring" style="--p:{p:.1f};--ring-color:{color}" aria-hidden="true"></div>'


def kpi(label, value, note="", ico="activity", tone="", viz="", alert=False):
    # « 14 / 17 » : le dénominateur est plus petit, la valeur principale reste lisible d'un coup d'œil.
    main, sep, rest = str(value).partition(" / ")
    shown = counting(main) + (f"<small>/ {counting(rest)}</small>" if sep else "")
    # Couleur = sens : vert (crises annotées, résultats sur les crises), orange (détections),
    # rose (erreur à signaler). Toute autre carte reste blanche.
    family = {"green": "annotated", "amber": "detected"}.get(tone, "error" if alert else "plain")
    return (
        f'<div class="kpi tone-{family}{" alert" if alert else ""}{" has-viz" if viz else ""}"><div class="top"><span class="label">{esc(label)}</span>'
        f'{icon(ico, tone)}</div><div class="value">{shown}</div>'
        + (f'<div class="note">{esc(note)}</div>' if note else "")
        + (f'<div class="viz">{viz}</div>' if viz else "")
        + "</div>"
    )


def kpis(cards, slot=None):
    show(
        '<div class="kpis">'
        + "".join(
            card.replace('<div class="kpi', f'<div style="--i:{k}" class="anim kpi', 1)
            for k, card in enumerate(cards)
        )
        + "</div>",
        slot,
    )


def section(title, eyebrow=""):
    show(
        '<div class="section anim">'
        + (f'<span class="eyebrow">{esc(eyebrow)}</span>' if eyebrow else "")
        + f"<h3>{esc(title)}</h3></div>"
    )


def pill(text, kind=""):
    return f'<span class="pill {kind}">{esc(text)}</span>'


def table(frame, numeric=None, pills=None, mono=(), best=None, max_height=None, markup=None):
    """Tableau sombre : en-tête collant, nombres alignés à droite en chasse fixe, pastilles de statut.

    `markup` : colonnes dont la fonction renvoie du HTML déjà sûr (mini-frises SVG).
    """
    numeric, pills, markup = numeric or {}, pills or {}, markup or {}
    head = "".join(
        f'<th class="{"num" if c in numeric else ""}">{esc(c)}</th>' for c in frame.columns
    )
    rows = []
    for k, row in enumerate(frame.itertuples(index=False)):
        cells = []
        for column, value in zip(frame.columns, row, strict=True):
            if column in markup:
                cells.append(f'<td class="tl-cell">{markup[column](value)}</td>')
            elif column in pills:
                cells.append(f"<td>{pill(*pills[column](value))}</td>")
            elif column in numeric:
                cells.append(f'<td class="num">{esc(numeric[column](value))}</td>')
            else:
                cells.append(f'<td class="{"id" if column in mono else ""}">{esc(value)}</td>')
        rows.append(
            f'<tr class="{"best" if best is not None and k == best else ""}">{"".join(cells)}</tr>'
        )
    style = f' style="max-height:{max_height}px"' if max_height else ""
    show(
        f'<div class="tbl-wrap"{style}><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


def state_box(kind, title, detail="", slot=None):
    """État vide, erreur, partiel ou informatif, en français simple, avec une petite icône."""
    name, tone = STATE_STYLE[kind]
    show(
        f'<div class="state {kind}" role="status">{icon(name, tone)}<div><b>{esc(title)}</b>'
        + (f'<span class="d">{detail}</span>' if detail else "")
        + "</div></div>",
        slot,
    )


def skeleton(slot, height, label):
    show(
        f'<div class="skeleton" style="height:{height}px" aria-busy="true">{esc(label)}</div>', slot
    )


def legend_key():
    show(
        '<div class="legend-key"><span><span class="sw ann"></span>crise annotée · bande verte</span>'
        '<span><span class="sw det"></span>crise détectée · orange, bord pointillé</span>'
        '<span><span class="sw both"></span>annotée et détectée</span></div>'
    )


def mini_timeline(duration, annotated, detected=()):
    """Mini-frise d'un enregistrement : crises annotées en vert, détectées en orange."""
    width, scale = 168, 168 / max(duration, 1)
    bars = "".join(
        f'<rect class="t-ann" x="{a * scale:.1f}" y="2" width="{max(1.5, (b - a) * scale):.1f}" height="7"/>'
        for a, b in annotated
    ) + "".join(
        f'<rect class="t-det" x="{a * scale:.1f}" y="9" width="{max(1.5, (b - a) * scale):.1f}" height="5"/>'
        for a, b in detected
    )
    label = f"{len(annotated)} crise(s) annotée(s)" + (
        f", {len(detected)} détectée(s)" if detected else ""
    )
    return (
        f'<svg class="mini-tl" width="{width}" height="16" viewBox="0 0 {width} 16" role="img" '
        f'aria-label="{esc(label)}"><line class="t-base" x1="0" y1="8" x2="{width}" y2="8"/>{bars}</svg>'
    )


def plural(n, one, many):
    return one if n <= 1 else many


def signed(value):
    return ("+" if value > 0 else "−" if value < 0 else "±") + fmt(abs(value), 0, " s")


def footer():
    show(
        '<div class="foot"><span><b>Détection automatique des crises d\'absence</b> · Fedi Ben Brahim, '
        "Yassine Boujelbenne · ENIT, Université de Tunis El Manar · 2026–2027</span>"
        "<span>Outil d'aide à l'analyse, ne remplace pas l'avis d'un neurologue.</span></div>"
    )


# ======================================================================
# Contexte partagé : enregistrement, modèle, règle de décision
# ======================================================================


def default_record():
    options = service.record_options(require_signal=False)
    return next((r for r in options if service.catalogue()[r]["status"] == "test"), options[0])


def context():
    """État de la session, initialisé une fois depuis l'URL (liens partageables)."""
    if "ctx" not in st.session_state:
        options = service.record_options(require_signal=False)
        params = st.query_params
        st.session_state.ctx = {
            "record": params.get("id") if params.get("id") in options else default_record(),
            "model": "cnn" if params.get("modele") == "cnn" and service.cnn_available() else "rf",
            "rule": params.get("regle") if params.get("regle") in RULES else "retenu",
        }
        st.session_state.recent = []
    return st.session_state.ctx


def set_context(**changes):
    state = context()
    state.update(changes)
    if "record" in changes:
        recent = [r for r in st.session_state.recent if r != changes["record"]]
        st.session_state.recent = [changes["record"], *recent][:5]


def status_pill(record):
    row = service.catalogue()[record]
    return pill(service.STATUS_LABELS[row["status"]], STATUS_PILL[row["status"]])


def topbar(title, eyebrow, record=False, model=False, rule=False):
    """Barre supérieure collante : titre de page, enregistrement courant, sélecteurs de portée."""
    state = context()
    options = service.record_options(require_signal=False)
    with st.container(key="topbar"):
        right = (
            f'<span class="rec-pill">Enregistrement <b>{esc(state["record"])}</b>{status_pill(state["record"])}</span>'
            if record
            else ""
        )
        show(
            f'<div class="topbar-head"><div><div class="eyebrow">{esc(eyebrow)}</div>'
            f'<div class="topbar-title" role="heading" aria-level="1">{esc(title)}</div></div>{right}</div>'
        )
        if not (record or model or rule):
            return
        widths = (
            ([3.4, 0.42, 0.42] if record else [])
            + ([2.6] if model else [])
            + ([2.3] if rule else [])
        )
        if [r for r in st.session_state.get("recent", []) if r != state["record"]]:
            widths.append(1.3)
        widths.append(max(1.2, 10 - sum(widths)))
        cols = iter(st.columns(widths, vertical_alignment="bottom"))
        if record:
            position = options.index(state["record"])
            choice = next(cols).selectbox(
                "Enregistrement", options, index=position, format_func=service.record_label,
                help="Tapez pour rechercher un identifiant.",
            )  # fmt: skip
            if choice != state["record"]:
                set_context(record=choice)
            if next(cols).button(
                "",
                icon=":material/chevron_left:",
                help="Enregistrement précédent",
                disabled=position == 0,
                width="stretch",
            ):
                set_context(record=options[position - 1])
                st.rerun()
            if next(cols).button(
                "",
                icon=":material/chevron_right:",
                help="Enregistrement suivant",
                disabled=position == len(options) - 1,
                width="stretch",
            ):
                set_context(record=options[position + 1])
                st.rerun()
        if model:
            available = ["rf", "cnn"] if service.cnn_available() else ["rf"]
            picked = next(cols).segmented_control(
                "Modèle", available, default=state["model"], format_func=MODELS_SHORT.get
            )
            if picked and picked != state["model"]:
                set_context(model=picked)
        if rule:
            picked = next(cols).segmented_control("Règle de décision", list(RULES), default=state["rule"], format_func=RULES_SHORT.get)  # fmt: skip
            if picked and picked != state["rule"]:
                set_context(rule=picked)
        recent = [r for r in st.session_state.get("recent", []) if r != state["record"]]
        if recent:
            with next(cols).popover("Récents", icon=":material/history:"):
                for rid in recent:
                    if st.button(service.record_label(rid), key=f"recent_{rid}", type="tertiary"):
                        set_context(record=rid)
                        st.rerun()
        changed = state != {"record": default_record(), "model": "rf", "rule": "retenu"}
        if next(cols).button(
            "Réinitialiser",
            help="Revenir à la portée par défaut",
            disabled=not changed,
            type="tertiary",
        ):
            set_context(record=default_record(), model="rf", rule="retenu")
            st.rerun()
    keys = {"id": state["record"]} if record else {}
    if model:
        keys["modele"] = state["model"]
    if rule:
        keys["regle"] = state["rule"]
    st.query_params.from_dict(keys)


# ======================================================================
# Accueil
# ======================================================================


PAGES = {}  # rempli par main() : objets de page pour les liens internes
PATHS = [
    ("understand", "Comprendre une absence →", "Un fond normal et une crise à la même échelle, les électrodes, la durée des crises."),
    ("visualisation", "Explorer un enregistrement →", "Faire défiler un EEG réel, voie par voie, avec son spectrogramme et sa carte du scalp."),
    ("detection", "Lancer la détection →", "Comparer les épisodes détectés aux crises annotées, avec la forêt ou le CNN."),
    ("errors", "Relire les erreurs →", "Chaque crise manquée et chaque fausse alarme du test, avec son extrait d'EEG."),
]  # fmt: skip


def hero_excerpt():
    """Premier épisode annoté d'un enregistrement du test réservé, sur quatre voies."""
    meta = service.metadata()
    for record in ["230718A-A", *meta["test_records"]]:
        events = service.annotated_events(record) if record in service.catalogue() else []
        if events:
            break
    else:
        return None
    a, b = events[0]
    try:
        time_values, _, names, _ = service.multichannel(record)
    except service.DataError:
        return None
    wanted = [n for n in ("EEGFp1", "EEGF7", "EEGC3", "EEGO1") if n in names]
    filtered = service.display_filter(record, True)
    lo, hi = max(0.0, a - 6), b + 6
    i0, i1 = np.searchsorted(time_values, [lo, hi])
    fig = figures.multichannel_trace(
        time_values[i0:i1], filtered[[names.index(n) for n in wanted], i0:i1], wanted, [(a, b)],
        slider=False, height=330,
    )  # fmt: skip
    return record, (a, b), fig


def page_home():
    meta = service.metadata()
    cohort = service.cohort_summary()
    series = service.cohort_series()
    test, loro = meta["metrics"]["test"], meta["metrics"]["loro"]

    # En-tête éditorial : ce qu'est une absence, à côté d'un extrait réel du jeu de données.
    left, right = st.columns([5, 6], gap="large")
    with left:
        show(
            '<div class="hero anim"><div class="eyebrow">Mini-projet data mining · ENIT</div>'
            "<h1 class=\"display\">Détecter les crises d'absence sur l'EEG</h1>"
            "<p class=\"lead\">Une absence est une crise d'épilepsie de quelques secondes : l'enfant "
            "s'interrompt, le regard fixe, puis reprend sans s'en souvenir. Sur le tracé, elle prend la "
            "forme de pointes-ondes régulières, vers 3 Hz, sur toutes les voies à la fois.</p>"
            '<p class="body">Le signal est découpé en fenêtres de 2 s, chaque fenêtre est classée '
            "(forêt aléatoire sur la voie F7, ou réseau convolutif sur les 19 voies), puis les fenêtres "
            "positives consécutives sont regroupées en épisodes et comparées aux annotations du neurologue.</p></div>"
        )
    with right:
        excerpt = hero_excerpt()
        if excerpt:
            record, (a, b), fig = excerpt
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            show(
                f'<p class="caption">Extrait réel : <span class="mono">{esc(record)}</span>, voies Fp1, F7, C3 et O1, '
                f"filtrées comme pour la détection. Bande verte : crise annotée de {fmt(a, 0)} à {fmt(b, 0)} s.</p>"
            )

    with st.container(key="paths"):
        cols = []
        for row in ("pathrow1", "pathrow2"):
            with st.container(key=row):
                cols += st.columns(2, gap="large")
        for n, (col, (key, title, text)) in enumerate(zip(cols, PATHS, strict=True)):
            # La tuile prend la couleur de la section de la page vers laquelle elle mène.
            with col, st.container(key=f"tile-{SECTION[key]}-{n}"):
                show(f'<span class="tile-section">{SECTION_LABEL[SECTION[key]]}</span>')
                st.page_link(PAGES[key], label=title)
                show(f'<p class="path-text">{text}</p>')

    section("Résultats sur le test réservé", MODELS["rf"])
    found, total = test["event_TP"], test["event_TP"] + test["event_FN"]
    kpis([
        kpi("Crises retrouvées", f"{found} / {total}", "IoU ≥ 0,3 avec l'annotation", "target", "green", ring(found / total, "var(--green)")),
        kpi("Fausses alarmes", fmt(test["false_alarms_per_hour"], 1, " / h"), "par heure de signal", "bell", "red",
            sparkline(series["loro_false_alarms"], "var(--red)")),
        kpi("F1 des fenêtres", fmt(test["f1"], 2), "après post-traitement", "activity", "", ring(test["f1"])),
        kpi("Crises retrouvées (LORO)", percent(loro["event_recall"]), "chaque tracé exclu à son tour", "trend", "",
            sparkline(series["loro_f1"])),
    ])  # fmt: skip
    if service.cnn_available():
        post = service.cnn_metadata()["test"]["postprocessed"]
        found, total = post["event_TP"], post["event_TP"] + post["event_FN"]
        weak = found / total < 0.6
        section("Résultats sur le test réservé", MODELS["cnn"])
        kpis([
            kpi("Crises retrouvées", f"{found} / {total}", "seuil hors pli mal transposé" if weak else "IoU ≥ 0,3",
                "target", "red" if weak else "green", ring(found / total, "var(--red)" if weak else "var(--green)"), alert=weak),
            kpi("Fausses alarmes", fmt(post["false_alarms_per_hour"], 1, " / h"), "par heure de signal", "bell", "red"),
            kpi("F1 des fenêtres", fmt(post["f1"], 2), "après post-traitement", "cpu", "", ring(post["f1"])),
        ])  # fmt: skip
    explain(
        f"Test réservé : {len(meta['test_records'])} enregistrements jamais vus pendant l'entraînement "
        f"({fmt(test['hours'], 1)} h). Même séparation pour les deux modèles ; post-traitement réglé sur des "
        "scores hors pli, jamais sur le test. Mini-courbes : une valeur par enregistrement (validation LORO)."
    )

    section("Données", "Cohorte")
    kpis([
        kpi("Enregistrements", cohort["records"], f"{fmt(cohort['hours'], 1, ' h')} à {fmt(cohort['fs'], 0)} Hz", "database", "", sparkline(series["duration_min"])),
        kpi("Crises annotées", cohort["events"], "par le neurologue", "zap", "green", sparkline(series["events"], "var(--green)")),
        kpi("Fenêtres de 2 s", fmt(cohort["windows"], 0), "pas de 1 s", "layers", "", sparkline(series["windows"])),
        kpi("Fenêtres en crise", percent(cohort["positive_windows"] / cohort["windows"]), "classe minoritaire", "percent",
            "green", ring(cohort["positive_windows"] / cohort["windows"], "var(--green)")),
    ])  # fmt: skip
    explain(
        "Avec 4 % de fenêtres en crise, un modèle qui répond toujours « hors crise » obtient déjà 96 % "
        "d'accuracy : on juge donc les modèles sur le F1 et sur les crises retrouvées."
    )

    section("Chaîne de traitement", "Méthode")
    steps = [
        ("Lecture", "Timetables MATLAB décodées et vérifiées valeur par valeur."),
        ("Fenêtres", "2 s, pas de 1 s ; « crise » si ≥ 50 % des échantillons sont annotés."),
        ("Représentation", "16 caractéristiques sur F7, ou 19 voies brutes à 128 Hz pour le CNN."),
        ("Classification", "Forêt choisie par AP en validation groupée ; CNN à perte pondérée."),
        ("Épisodes", "Lissage, seuil et durée minimale réglés hors pli, puis appariement IoU."),
    ]
    show(
        '<div class="steps">'
        + "".join(
            f'<div class="step anim" style="--i:{k}"><span class="n">0{k + 1}</span><b>{t}</b><span>{d}</span></div>'
            for k, (t, d) in enumerate(steps)
        )
        + "</div>"
    )

    section("Enregistrements", "Catalogue")
    rows = pd.DataFrame(service.catalogue().values())
    found = st.session_state.get("detections", {})
    frame = pd.DataFrame({
        "Identifiant": rows.id,
        "Statut": rows.status,
        "Crises annotées": rows.annotated_seizures,
        "Frise": rows.id,
        "Durée (min)": rows.duration_s / 60,
        "Âge": rows.age.fillna("—"),
        "Sexe": rows.sex.fillna("—"),
        "Remarque": rows.reason,
    })  # fmt: skip
    durations = dict(zip(rows.id, rows.duration_s, strict=True))
    table(
        frame,
        numeric={
            "Crises annotées": lambda v: "—" if pd.isna(v) else f"{int(v)}",
            "Durée (min)": lambda v: fmt(v, 1),
        },
        pills={"Statut": lambda s: (service.STATUS_LABELS[s], STATUS_PILL[s])},
        markup={
            "Frise": lambda rid: mini_timeline(
                durations[rid], service.annotated_events(rid), found.get((rid, "rf"), ())
            )
        },
        mono=("Identifiant",),
        max_height=560,
    )
    explain(
        "Frise : position des crises annotées (vert) sur toute la durée ; les détections de la forêt "
        "(orange) s'ajoutent dès qu'un enregistrement a été analysé dans la page Détection."
    )


# ======================================================================
# Visualisation
# ======================================================================


@st.cache_data
def viewer_template():
    return (STATIC / "eeg_viewer.html").read_text(encoding="utf-8")


def viewer(record, channels, lowpass, center, annotated, detected=None, model="rf"):
    """Lecteur EEG dans le navigateur : déplacement, clavier et survol sans recharger la page."""
    import json

    import streamlit.components.v1 as components

    signal = service.viewer_signal(record, tuple(channels), lowpass)
    score = None
    if detected is not None:
        try:
            result = (
                service.cnn_analyse(record) if model == "cnn" else service.analyse_record(record)
            )
            score = {
                "t": result["scores"]["time"],
                "v": result["scores"]["value"],
                "thr": result["params"]["threshold"],
            }
        except (InputError, service.DataError):
            score = None
    window = 20.0
    payload = {
        **signal, "window": window, "start": max(0.0, float(center) - window / 2),
        "annotated": [list(map(float, e)) for e in annotated], "detected": [list(map(float, e)) for e in (detected or [])],
        "score": score,
    }  # fmt: skip
    colors = figures.viewer_colors()
    css = (
        " ".join(f"--{k}: {v};" for k, v in colors.items())
        + f" --surface: {figures.THEME['surface']}; --raised: {figures.THEME['raised']}; --hairline: {figures.THEME['hairline']}; --border: #758195;"
    )
    page = (
        viewer_template()
        .replace("__CSSVARS__", css)
        .replace("__COLORS__", json.dumps(colors))
        .replace("__PAYLOAD__", json.dumps(payload))
    )
    height = 44 + 6 + 42 * len(signal["channels"]) + 22 + (86 if score else 56) + 2 + 40
    components.html(page, height=height)


def page_visualisation():
    topbar("Visualiser un enregistrement", "Exploration du signal", record=True)
    state = context()
    record = state["record"]
    try:
        time_values, _, names, fs = service.multichannel(record)
    except service.DataError as exc:
        state_box("error", "Lecture impossible", esc(exc))
        return
    annotated = service.annotated_events(record)
    detected = st.session_state.get("detections", {}).get((record, state["model"]))
    duration = float(time_values[-1])

    tab_signal, tab_spectro, tab_topo = st.tabs(["Signal EEG", "Spectrogramme", "Carte du scalp"])
    with tab_signal:
        targets = {"Début de l'enregistrement": 0.0}
        focus = st.session_state.get("focus")
        focused = bool(focus and focus[0] == record)
        if focused:
            targets = {focus[2]: focus[1], **targets}
        for k, (a, b) in enumerate(annotated, 1):
            targets[f"Crise annotée {k} ({fmt(a, 0)}–{fmt(b, 0)} s)"] = (a + b) / 2
        c = st.columns([2, 5], vertical_alignment="bottom")
        jump = c[0].selectbox(
            "Ouvrir à", list(targets), index=0 if focused else (1 if annotated else 0)
        )
        default = [n for n in ("EEGFp1", "EEGF7", "EEGF3", "EEGC3", "EEGP3", "EEGO1", "EEGFp2", "EEGF8", "EEGC4", "EEGO2") if n in names]  # fmt: skip
        chosen = c[1].multiselect(
            "Voies", names, default=default, format_func=lambda n: n.removeprefix("EEG")
        )
        t = st.columns([1, 1, 2])
        lowpass = t[0].toggle("Passe-bas 40 Hz", value=True)
        overlay = t[1].toggle(
            f"Détections ({'CNN' if state['model'] == 'cnn' else 'forêt'})",
            value=detected is not None, disabled=detected is None,
            help="Lancez la détection sur cet enregistrement pour superposer ses épisodes et son score.",
        )  # fmt: skip
        if not chosen:
            state_box("empty", "Aucune voie sélectionnée", "Ajoutez au moins une voie ci-dessus.")
        else:
            viewer(
                record,
                chosen,
                lowpass,
                targets[jump],
                annotated,
                detected if overlay else None,
                state["model"],
            )
            explain(
                "Signal filtré comme dans le pipeline (passe-haut 0,5 Hz). Faites glisser le tracé ou le cadre de la vue "
                "d'ensemble, utilisez les flèches du clavier après un clic sur le tracé, et survolez une voie pour lire "
                "le temps et l'amplitude. Une absence : pointes-ondes amples et rythmiques vers 3 Hz, sur toutes les voies."
            )  # fmt: skip
            if not annotated:
                state_box(
                    "info",
                    "Pas d'annotation pour cet enregistrement",
                    "Aucune crise de référence à afficher.",
                )

    with tab_spectro:
        channel = st.selectbox(
            "Voie",
            names,
            index=names.index("EEGF7") if "EEGF7" in names else 0,
            format_func=lambda n: n.removeprefix("EEG"),
        )
        slot = st.empty()
        skeleton(slot, 330, "Calcul du spectrogramme…")
        f, t, power = service.channel_spectrogram(record, channel)
        slot.plotly_chart(figures.spectrogram_figure(f, t, power, annotated), width="stretch")
        explain(
            "Puissance (dB) de chaque fréquence au cours du temps, sur tout l'enregistrement. Crises annotées "
            "encadrées en vert, bande 2,5–4,5 Hz en tirets. Une absence apparaît comme une bande claire autour "
            "de 3 Hz et de ses harmoniques."
        )

    with tab_topo:
        if len(names) < 4:
            state_box(
                "partial",
                "Montage incomplet",
                "Trop peu d'électrodes pour dessiner une carte du scalp.",
            )
            return
        first = (annotated[0][0] + annotated[0][1]) / 2 if annotated else duration / 2
        c = st.columns(2)
        center = c[0].slider("Instant étudié (s)", 1.0, duration - 1, float(np.clip(first, 1, duration - 1)), step=1.0, key=f"topo_{record}")  # fmt: skip
        baseline = c[1].slider(
            "Instant de référence (s)", 1.0, duration - 1,
            float(np.clip(center - 60 if center > 61 else center + 60, 1, duration - 1)), step=1.0, key=f"base_{record}",
        )  # fmt: skip
        labels, now = service.spike_wave_map(record, center)
        _, ref = service.spike_wave_map(record, baseline)
        during = any(a <= center <= b for a, b in annotated)
        fig = figures.topomaps(
            labels, [now, ref],
            [f"t = {fmt(center, 0)} s" + (" (crise annotée)" if during else ""), f"Référence t = {fmt(baseline, 0)} s"],
        )  # fmt: skip
        st.columns([3, 2])[0].pyplot(fig, width="stretch")
        explain(
            "Part de la puissance 0,5–30 Hz contenue dans la bande 2,5–4,5 Hz, par électrode, sur une fenêtre "
            "de 2 s. Pendant une absence, cette part augmente sur toute la tête : l'activité est généralisée."
        )


# ======================================================================
# Détection
# ======================================================================


def page_detection():
    topbar("Lancer la détection", "Détection", record=True, model=True, rule=True)
    state = context()
    use_cnn = state["model"] == "cnn"
    params = service.cnn_metadata()["params"] if use_cnn else service.metadata()["params"]

    upload = None
    if not use_cnn:
        with st.expander("Analyser un fichier externe au lieu de l'enregistrement sélectionné"):
            upload = st.file_uploader(
                "Fichier `_d.mat` original, `.npz` exporté par le projet, ou `.csv` avec Time (s) et EEGF7",
                type=["mat", "npz", "csv"],
            )
    if state["rule"] == "hysteresis":
        c = st.columns(3)
        high = c[0].slider("Début si le score dépasse", 0.05, 0.95, float(round(min(0.95, params["threshold"] * 1.4), 2)), 0.01)  # fmt: skip
        low = c[1].slider("Fin si le score passe sous", 0.01, high, float(round(min(high, params["threshold"] * 0.6), 2)), 0.01)  # fmt: skip
        min_windows = c[2].slider("Durée minimale (fenêtres)", 1, 10, params["min_windows"])
        st.caption(
            "Exploratoire : ces seuils ne sont pas réglés hors pli. Les ajuster en regardant un enregistrement "
            "annoté revient à les apprendre sur lui."
        )

    head_slot, cards_slot, chart_slot = st.empty(), st.empty(), st.empty()
    skeleton(cards_slot, 128, "Analyse en cours…")
    skeleton(chart_slot, 320, "Calcul du score de chaque fenêtre…")
    record = state["record"]
    try:
        if upload is not None:
            result = service.analyse_upload(upload.getvalue(), upload.name)
            record = result["record_id"] or Path(upload.name).stem
        elif use_cnn:
            result = service.cnn_analyse(record)
        else:
            result = service.analyse_record(record)
    except (InputError, service.DataError) as exc:
        cards_slot.empty()
        row = service.catalogue().get(record, {})
        if upload is None and row and not row.get("available", True):
            state_box(
                "partial", "Analyse impossible : données incomplètes",
                f"{esc(row['reason'])}. La forêt a besoin de la voie F7 et le CNN des 19 voies. "
                "La page Visualisation reste disponible pour les voies présentes.",
                slot=chart_slot,
            )  # fmt: skip
        else:
            state_box("error", "Analyse impossible", f"{esc(exc)} Vérifiez le format du fichier ou choisissez un autre enregistrement.", slot=chart_slot)  # fmt: skip
        return

    annotated = [(e["start_s"], e["end_s"]) for e in (result["reference"] or {}).get("events", [])]
    scores = np.asarray(result["scores"]["value"])
    if state["rule"] == "hysteresis":
        labels = service.hysteresis(scores, high, low, min_windows)
        detected = service.detections_from_labels(result, labels)
        band, threshold = (high, low), None
    else:
        detected = [(e["start_s"], e["end_s"]) for e in result["events"]]
        band, threshold = None, params["threshold"]
    st.session_state.setdefault("detections", {})[(record, state["model"])] = detected
    toast_key = (record, state["model"], state["rule"], tuple(detected))
    if st.session_state.get("last_toast") != toast_key:
        st.session_state.last_toast = toast_key
        n = len(detected)
        st.toast(f"Détection terminée : {n} crise{'s' if n > 1 else ''} trouvée{'s' if n > 1 else ''}", icon=":material/check_circle:")  # fmt: skip
    if upload is not None:
        state_box("info", f"Fichier externe : {upload.name}", ("Reconnu comme " + esc(record)) if result["record_id"] else "Enregistrement inconnu : pas de comparaison possible.")  # fmt: skip

    matches, tp, fp, fn = (
        service.compare_to_reference(annotated, detected) if annotated else ([], None, None, None)
    )
    duration = result["duration_s"]
    detected_time = sum(b - a for a, b in detected)
    step = max(1, len(scores) // 60)
    cards = [
        kpi("Crises détectées", str(len(detected)), MODELS[state["model"]], "zap", "amber", sparkline(scores[::step], "var(--amber)")),
        kpi("Temps en crise détecté", fmt(detected_time, 0, " s"), f"{percent(detected_time / duration, 1)} du tracé", "clock", "amber",
            ring(detected_time / duration, "var(--amber)")),
    ]  # fmt: skip
    if annotated:
        cards += [
            kpi("Crises annotées retrouvées", f"{tp} / {len(annotated)}", f"{fn} manquée{'s' if fn > 1 else ''}" if fn else "aucune manquée",
                "target", "red" if fn else "green", ring(tp / len(annotated), "var(--red)" if fn else "var(--green)"), alert=bool(fn)),
            kpi("Fausses alarmes", str(fp), "détections sans crise annotée", "bell", "red" if fp else "green", alert=bool(fp)),
        ]  # fmt: skip
    else:
        cards += [
            kpi(
                "Durée analysée",
                fmt(result["analyzed_duration_s"] / 60, 1, " min"),
                "sans annotation de référence",
                "clock",
            ),
            kpi("Fenêtres", fmt(result["n_windows"], 0), "de 2 s, pas de 1 s", "layers"),
        ]
    if annotated:
        headline = (
            f"<b>{tp}</b> {plural(tp, 'crise annotée', 'crises annotées')} sur {len(annotated)} "
            f"{'retrouvée' if tp <= 1 else 'retrouvées'}, "
            + (
                f"<b>{fp}</b> {plural(fp, 'fausse alarme', 'fausses alarmes')}"
                if fp
                else "aucune fausse alarme"
            )
        )
    else:
        headline = f"<b>{len(detected)}</b> {plural(len(detected), 'épisode détecté', 'épisodes détectés')}"
    rule_text = (
        f"seuil {fmt(params['threshold'], 2)}"
        if threshold is not None
        else f"seuils {fmt(high, 2)} / {fmt(low, 2)}"
    )
    bits = [
        MODELS[state["model"]],
        RULES[state["rule"]].lower(),
        rule_text,
        f"{fmt(duration / 60, 1)} min analysées",
    ]
    show(
        '<div class="result anim"><div class="eyebrow">Résultat</div>'
        f'<p class="headline">{headline}.</p><p class="subline">{" · ".join(esc(b) for b in bits)}</p></div>',
        head_slot,
    )
    kpis(cards, slot=cards_slot)
    chart_slot.plotly_chart(
        figures.score_timeline(result, annotated, detected, threshold, band), width="stretch"
    )
    legend_key()
    explain(
        ("Courbe bleue : probabilité moyenne du CNN sur 30 passages avec dropout actif (Monte-Carlo), après lissage ; "
         "la bande bleue translucide indique ±2 écarts-types, c'est-à-dire l'incertitude du réseau."
         if use_cnn else "Courbe bleue : probabilité de crise donnée par la forêt à chaque fenêtre, après lissage.")
        + " Une carte rose marquée ▲ signale une crise manquée ou une fausse alarme."
    )  # fmt: skip
    if not annotated:
        state_box(
            "info",
            "Pas d'annotation de référence",
            "Les épisodes détectés ne peuvent pas être comparés à une vérité terrain.",
        )

    tab_names = (
        ["Crises", "Tableau de bord"]
        + (["Ce que regarde le réseau"] if use_cnn else [])
        + ["Export"]
    )
    tabs = dict(zip(tab_names, st.tabs(tab_names), strict=True))
    by_detection = {j: (i, iou) for i, j, iou in matches}
    by_truth = {i: (j, iou) for i, j, iou in matches}
    times = np.asarray(result["scores"]["time"])
    spread = np.asarray(result["scores"].get("std") or np.zeros_like(scores))
    with tabs["Crises"]:
        if annotated:
            section("Confrontation aux annotations", "Par crise")
            rows = []
            for i, (a, b) in enumerate(annotated):
                match = by_truth.get(i)
                c, d = detected[match[0]] if match else (None, None)
                rows.append({
                    "Crise": i + 1, "Début réel (s)": a, "Fin réelle (s)": b, "Début détecté (s)": c,
                    "Fin détectée (s)": d, "Retard début": None if c is None else c - a,
                    "Écart fin": None if d is None else d - b, "IoU": match[1] if match else None, "Résultat": match,
                })  # fmt: skip
            dash = lambda f: lambda v: "—" if v is None or pd.isna(v) else f(v)  # noqa: E731
            table(
                pd.DataFrame(rows),
                numeric={"Crise": str, "Début réel (s)": dash(lambda v: fmt(v, 0)), "Fin réelle (s)": dash(lambda v: fmt(v, 0)),
                         "Début détecté (s)": dash(lambda v: fmt(v, 0)), "Fin détectée (s)": dash(lambda v: fmt(v, 0)),
                         "Retard début": dash(signed), "Écart fin": dash(signed), "IoU": dash(lambda v: fmt(v, 2))},
                pills={"Résultat": lambda m: ("Détectée", "ok") if m else ("Manquée", "err")},
            )  # fmt: skip
            explain("Retard positif : la détection commence (ou finit) plus tard que l'annotation.")
        section("Crises détectées", MODELS[state["model"]])
        rows = []
        for k, (a, b) in enumerate(detected):
            inside = (times >= a) & (times <= b)
            row = {"N°": k + 1, "Début (s)": a, "Fin (s)": b, "Durée (s)": b - a,
                   "Score max": float(scores[inside].max()) if inside.any() else np.nan}  # fmt: skip
            if use_cnn:
                lower = float((scores[inside] - 2 * spread[inside]).max()) if inside.any() else -1
                row["Certitude"] = "probable" if lower >= params["threshold"] else "incertaine"
            match = by_detection.get(k)
            row["Statut"] = ("ok", match) if match else ("fa" if annotated else "na", None)
            rows.append(row)
        if rows:
            table(
                pd.DataFrame(rows),
                numeric={"N°": str, "Début (s)": lambda v: fmt(v, 0), "Fin (s)": lambda v: fmt(v, 0),
                         "Durée (s)": lambda v: fmt(v, 0), "Score max": lambda v: fmt(v, 2)},
                pills={
                    "Statut": lambda s: (f"Détectée · crise annotée {s[1][0] + 1}", "ok") if s[0] == "ok"
                    else (("Fausse alarme", "err") if s[0] == "fa" else ("Sans référence", "")),
                    **({"Certitude": lambda c: ("Probable", "info") if c == "probable" else ("Incertaine", "warn")} if use_cnn else {}),
                },
            )  # fmt: skip
        else:
            state_box(
                "empty", "Aucune crise détectée",
                "Le score ne franchit jamais la règle de décision. Essayez la règle « Hystérésis » dans la barre "
                "supérieure pour explorer un seuil plus bas.",
            )  # fmt: skip
        explain(
            "Une détection correspond à une crise annotée si leur IoU (intersection sur union) atteint 0,3."
            + (" « Incertaine » : la probabilité moins deux écarts-types Monte-Carlo ne dépasse jamais le seuil." if use_cnn else "")
        )  # fmt: skip

    with tabs["Tableau de bord"]:
        stats_det = service.event_statistics(detected, duration)
        stats_ref = service.event_statistics(annotated, duration) if annotated else None
        indicators = [
            ("Nombre de crises", "count", 0), ("Crises par heure", "per_hour", 1),
            ("Temps total en crise (s)", "total_s", 0), ("Part du tracé en crise (%)", "fraction", None),
            ("Durée moyenne (s)", "mean_duration_s", 1), ("Intervalle moyen entre crises (s)", "mean_gap_s", 0),
        ]  # fmt: skip

        def cell(stats, key, digits):
            if stats is None:
                return "—"
            return fmt(100 * stats[key], 1) if digits is None else fmt(float(stats[key]), digits)

        left, right = st.columns([2, 3])
        with left:
            table(
                pd.DataFrame({
                    "Indicateur": [r[0] for r in indicators],
                    "Détecté": [cell(stats_det, *r[1:]) for r in indicators],
                    "Annoté": [cell(stats_ref, *r[1:]) for r in indicators],
                }),
                numeric={"Détecté": str, "Annoté": str},
            )  # fmt: skip
        with right:
            if len(detected) or annotated:
                st.plotly_chart(figures.duration_histogram(stats_det["durations"], stats_ref["durations"] if stats_ref else []), width="stretch")  # fmt: skip
            else:
                state_box("empty", "Pas de durée à afficher", "Aucune crise détectée ni annotée.")
        explain(
            f"Calculé sur {fmt(duration / 60, 1)} min de signal. Histogramme par classes de 2 s ; hachures = détectées."
        )

    if use_cnn:
        with tabs["Ce que regarde le réseau"]:
            targets = {}
            for i, (a, b) in enumerate(annotated):
                targets[f"Crise annotée {i + 1} ({fmt(a, 0)}–{fmt(b, 0)} s)"] = (a + b) / 2
            for k, (a, b) in enumerate(detected):
                if k not in by_detection:
                    targets[f"Fausse alarme {k + 1} ({fmt(a, 0)}–{fmt(b, 0)} s)"] = (a + b) / 2
            if not targets:
                targets = {f"Détection {k + 1}": (a + b) / 2 for k, (a, b) in enumerate(detected)}
            targets["Activité de fond (t = 30 s)"] = 30.0
            c = st.columns([3, 2], vertical_alignment="bottom")
            target = c[0].selectbox("Zone à expliquer", list(targets))
            slot, info = st.empty(), c[1].empty()
            skeleton(slot, 640, "Calcul de la carte de saillance…")
            sal_times, signals, sal, prob = service.cnn_saliency(record, targets[target])
            show(
                f'<span class="rec-pill">Probabilité maximale du réseau sur la zone <b>{fmt(prob, 2)}</b></span>',
                info,
            )
            slot.plotly_chart(
                figures.saliency_figure(
                    sal_times, signals, sal, service.cnn_metadata()["channels"], annotated
                ),
                width="stretch",
            )
            explain(
                "Carte de saillance (gradient × entrée) : pour chaque voie et chaque instant, l'influence du signal "
                "sur la sortie du réseau, moyennée sur des fenêtres de 2 s décalées de 0,25 s (6 s affichées). Si les "
                "zones lumineuses tombent sur les complexes pointe-onde, le réseau a appris le bon motif ; si elles "
                "tombent sur des artefacts, il s'appuie sur un raccourci."
            )

    with tabs["Export"]:
        frame = pd.DataFrame(detected, columns=["debut_s", "fin_s"])
        frame.insert(0, "crise", range(1, len(frame) + 1))
        frame["duree_s"] = frame.fin_s - frame.debut_s
        stem = f"{record}_{state['model']}"
        c = st.columns(3)
        c[0].download_button("Crises détectées (CSV)", frame.to_csv(index=False), f"{stem}_crises.csv", "text/csv", width="stretch", icon=":material/table:")  # fmt: skip
        c[1].download_button("Frise (PNG)", figures.timeline_png(result, annotated, detected, record), f"{stem}_frise.png", "image/png", width="stretch", icon=":material/image:")  # fmt: skip
        c[2].download_button("Analyse complète (JSON)", service.analysis_json(result), f"{stem}_analyse.json", "application/json", width="stretch", icon=":material/data_object:")  # fmt: skip
        explain(
            f"Source : {result['source']}. Empreinte SHA-256 du fichier : {result['input_sha256'][:16]}…"
        )


# ======================================================================
# Modèles
# ======================================================================


def reset_model_filters(sets, models):
    st.session_state.f_sets = sets
    st.session_state.f_models = models


def page_models():
    topbar("Comparaison des modèles", "Évaluation")
    meta = service.metadata()
    table_all = service.model_table()
    has_cnn = service.cnn_available()
    if has_cnn:
        table_all = pd.concat(
            [table_all, pd.DataFrame([service.cnn_table_row()])], ignore_index=True
        )
        table_all = table_all.sort_values("average_precision_mean", ascending=False).reset_index(
            drop=True
        )
    retained = eeg.config_label(meta["configuration"])
    chosen_row = table_all[table_all.configuration == meta["configuration"]].iloc[0]
    base = table_all.iloc[0]
    baseline = (base.TN + base.FP) / (base.TN + base.FP + base.FN + base.TP)
    show(
        '<p class="lead anim">38 configurations classiques (4 modèles × stratégies de déséquilibre × 2 jeux de '
        "caractéristiques)"
        + (" et un CNN 1D" if has_cnn else "")
        + ", choisies sur l'AP de validation interne "
        "(plis groupés par enregistrement) ; le test réservé n'est lu qu'une fois.</p>"
    )
    kpis([
        kpi("Configurations comparées", str(len(table_all)), "38 classiques" + (" + CNN" if has_cnn else ""), "layers"),
        kpi("AP validation (retenue)", fmt(chosen_row.average_precision_mean, 3), "critère de sélection", "trend", "",
            ring(chosen_row.average_precision_mean)),
        kpi("AP test (retenue)", fmt(chosen_row.average_precision, 3), "lu une seule fois", "target", "",
            ring(chosen_row.average_precision, "var(--green)")),
        kpi("Toujours « hors crise »", percent(baseline), "accuracy de référence", "percent", "", ring(baseline)),
    ])  # fmt: skip

    names = ["Tableau", "Courbes ROC et PR", "Calibration"] + (["Deep learning"] if has_cnn else [])
    names += ["Espace des fenêtres", "Validation par enregistrement"]
    tabs = dict(zip(names, st.tabs(names), strict=True))

    with tabs["Tableau"]:
        all_sets = sorted(table_all["Caractéristiques"].unique())
        all_models = list(eeg.MODELS.values()) + (["CNN 1D"] if has_cnn else [])
        st.session_state.setdefault("f_sets", all_sets)
        st.session_state.setdefault("f_models", all_models)
        c = st.columns([1, 1, 1], vertical_alignment="bottom")
        sets = c[0].multiselect("Caractéristiques", all_sets, key="f_sets")
        models = c[1].multiselect("Modèles", all_models, key="f_models")
        c[2].button(
            "Réinitialiser les filtres", type="tertiary", on_click=reset_model_filters, args=(all_sets, all_models),
            disabled=set(sets) == set(all_sets) and set(models) == set(all_models),
        )  # fmt: skip
        shown = table_all[
            table_all["Caractéristiques"].isin(sets) & table_all["Modèle"].isin(models)
        ].reset_index(drop=True)
        explain(
            f"{len(shown)} configuration(s) sur {len(table_all)} · ★ : forêt retenue en validation interne"
        )
        if shown.empty:
            state_box(
                "empty", "Aucune configuration ne correspond aux filtres",
                f"Caractéristiques : {esc(', '.join(sets) or 'aucune')} · Modèles : {esc(', '.join(models) or 'aucun')}. "
                "Élargissez un filtre ou réinitialisez-les.",
            )  # fmt: skip
        else:
            view = pd.DataFrame({
                "Configuration": [("★ " if c == retained else "") + c for c in shown.Configuration],
                "AP validation": shown.average_precision_mean, "AP test": shown.average_precision,
                "Accuracy": shown.accuracy, "Précision": shown.precision, "Rappel": shown.recall, "F1-score": shown.f1,
                "VP": shown.TP, "FP": shown.FP, "FN": shown.FN, "VN": shown.TN,
            })  # fmt: skip
            three = lambda v: fmt(v, 3)  # noqa: E731
            whole = lambda v: fmt(v, 0)  # noqa: E731
            table(
                view,
                numeric={"AP validation": three, "AP test": three, "Accuracy": three, "Précision": three, "Rappel": three,
                         "F1-score": three, "VP": whole, "FP": whole, "FN": whole, "VN": whole},
                best=next((k for k, c in enumerate(view.Configuration) if c.startswith("★")), None),
                max_height=440,
            )  # fmt: skip
            explain(
                "Métriques des fenêtres sur le test réservé, au seuil par défaut (sans post-traitement), triées par AP "
                "de validation. L'accuracy est trompeuse car les crises sont rares : on compare surtout le F1 et l'AP."
            )
            left, right = st.columns([5, 7])
            with left:
                section("Matrice de confusion", "Test réservé")
                pick = st.selectbox("Configuration", list(shown.Configuration), index=list(shown.Configuration).index(retained)
                                    if retained in set(shown.Configuration) else 0)  # fmt: skip
                row = shown[shown.Configuration == pick].iloc[0]
                st.plotly_chart(
                    figures.confusion_figure(row.TN, row.FP, row.FN, row.TP),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                explain(
                    "Nombre de fenêtres et part de chaque ligne (réel). Couleur plus claire = part plus grande."
                )
            with right:
                section("Comparaison", "Fenêtres du test")
                metric = st.segmented_control("Métrique", ["F1-score", "AP test", "Rappel", "Précision"], default="F1-score") or "F1-score"  # fmt: skip
                plot = view.assign(Configuration=shown.Configuration)
                st.plotly_chart(
                    figures.metric_lollipop(plot, metric, metric, retained), width="stretch"
                )
                explain(
                    "Meilleure valeur en bleu ; ★ = configuration retenue en validation interne."
                )

    scores = service.test_scores()
    if has_cnn:
        scores = scores.merge(
            service.cnn_test_scores()[["record_id", "start_s", "cnn"]], on=["record_id", "start_s"]
        )

    def label(name):
        return "CNN 1D · 19 voies brutes" if name == "cnn" else eeg.config_label(name)

    y = scores.Class.to_numpy()
    with tabs["Courbes ROC et PR"]:
        options = ["cnn" if o == "deep__original__cnn" else o for o in table_all.configuration]
        defaults = [meta["configuration"]] + (["cnn"] if has_cnn else [])
        defaults += [n for n in ("improved__original__svm", "base__original__rf") if n in options]
        chosen = st.multiselect("Configurations", options, default=defaults, format_func=label)
        if not chosen:
            state_box(
                "empty",
                "Aucune configuration choisie",
                "Ajoutez au moins une configuration ci-dessus.",
            )
        else:
            roc_entries, pr_entries = [], []
            for name in chosen:
                s = scores[name].to_numpy()
                fpr, tpr, _ = roc_curve(y, s)
                precision, recall, _ = precision_recall_curve(y, s)
                roc_entries.append((label(name), fpr, tpr, auc(fpr, tpr)))
                pr_entries.append((label(name), recall, precision, average_precision_score(y, s)))
            left, right = st.columns(2)
            with left:
                section("Courbe ROC", "Aire sous la courbe")
                st.plotly_chart(figures.curves(roc_entries, "roc", y.mean()), width="stretch")
            with right:
                section("Courbe précision-rappel", "Précision moyenne")
                st.plotly_chart(figures.curves(pr_entries, "pr", y.mean()), width="stretch")
        explain(
            f"Scores continus des {len(y)} fenêtres du test réservé ({int(y.sum())} en crise). Avec {percent(y.mean())} de "
            "fenêtres positives, la courbe ROC paraît excellente pour presque tous les modèles ; la courbe précision-rappel "
            "est plus parlante. Chaque courbe a sa couleur et son type de trait. Le SVM donne une marge, pas une probabilité."
        )

    with tabs["Calibration"]:
        probabilistic = [
            "cnn" if n == "deep__original__cnn" else n
            for n in table_all.configuration
            if not n.endswith("__svm")
        ]
        defaults = [meta["configuration"]] + (["cnn"] if has_cnn else []) + ["base__original__rf"]
        chosen = st.multiselect(
            "Modèles", probabilistic, default=defaults, format_func=label, key="calib"
        )
        if chosen:
            entries = [(label(n), *service.calibration(y, scores[n].to_numpy())) for n in chosen]
            st.plotly_chart(figures.reliability_figure(entries), width="stretch")
        else:
            state_box("empty", "Aucun modèle choisi", "Ajoutez au moins un modèle ci-dessus.")
        explain(
            "Courbe de fiabilité sur le test réservé : fenêtres regroupées par probabilité prédite (classes de 0,1), "
            "comparées à la fréquence réelle des crises. Sous la diagonale, le modèle est trop sûr de lui ; au-dessus, "
            "trop prudent. Taille des points = nombre de fenêtres ; Brier plus bas = mieux."
        )

    if has_cnn:
        with tabs["Deep learning"]:
            cnn = service.cnn_metadata()
            post, raw = cnn["test"]["postprocessed"], cnn["test"]["raw"]
            rf = meta["metrics"]
            three = lambda v: fmt(v, 3)  # noqa: E731
            table(
                pd.DataFrame([
                    {"Modèle": MODELS["rf"], "Entrée": "16 caractéristiques", "AP validation": rf["cv"]["average_precision"],
                     "AP test": chosen_row.average_precision, "F1 fenêtres (post-traité)": rf["test"]["f1"],
                     "Crises retrouvées": f"{rf['test']['event_TP']} / {rf['test']['event_TP'] + rf['test']['event_FN']}",
                     "Fausses alarmes / h": rf["test"]["false_alarms_per_hour"]},
                    {"Modèle": MODELS["cnn"], "Entrée": "19 voies × 256 échantillons", "AP validation": cnn["cv"]["average_precision_mean"],
                     "AP test": cnn["test"]["average_precision"], "F1 fenêtres (post-traité)": post["f1"],
                     "Crises retrouvées": f"{post['event_TP']} / {post['event_TP'] + post['event_FN']}",
                     "Fausses alarmes / h": post["false_alarms_per_hour"]},
                ]),
                numeric={"AP validation": three, "AP test": three, "F1 fenêtres (post-traité)": three,
                         "Crises retrouvées": str, "Fausses alarmes / h": lambda v: fmt(v, 1)},
            )  # fmt: skip
            weak = post["event_TP"] / (post["event_TP"] + post["event_FN"]) < 0.6
            kpis([
                kpi("Époques (modèle final)", str(cnn["epochs"]), "médiane des plis", "layers"),
                kpi("Seuil réglé hors pli", fmt(cnn["params"]["threshold"], 2), "sur les scores des plis", "target", "",
                    ring(cnn["params"]["threshold"], "var(--amber)")),
                kpi("F1 brut au seuil 0,5", fmt(raw["f1"], 2), "fenêtres, sans post-traitement", "activity", "", ring(raw["f1"])),
                kpi("Entraînement", fmt(cnn["training_minutes"], 0, " min"), "sur CPU", "cpu"),
            ])  # fmt: skip
            if weak:
                state_box(
                    "partial", "Le seuil réglé hors pli se transpose mal au modèle final",
                    "Le classement des fenêtres est aussi bon que celui de la forêt (AP proche), mais le modèle final, "
                    "réentraîné sur les 12 tracés, a des probabilités décalées : le seuil choisi sur les modèles des plis "
                    "retient trop peu d'épisodes. Résultat signalé tel quel, sans correction après lecture du test.",
                )  # fmt: skip
            st.plotly_chart(
                figures.training_curves(cnn["history"], cnn["cv"]["folds"]), width="stretch"
            )
            with st.expander("Architecture et protocole"):
                st.markdown(
                    "- **Entrée** : fenêtres de 2 s sur les 19 voies, passe-haut 0,5 Hz, rééchantillonnées à 128 Hz "
                    "(anti-repliement), normalisées voie par voie par la médiane et l'IQR de l'enregistrement (sans étiquette).\n"
                    "- **Réseau** : 3 blocs convolution 1D (noyau 7) + normalisation + ReLU, deux max-pooling, moyenne "
                    "globale, dropout 0,3, sortie logit — environ 48 000 paramètres.\n"
                    "- **Déséquilibre** : `BCEWithLogitsLoss` avec `pos_weight` = négatifs / positifs.\n"
                    "- **Augmentation** : décalage de ±0,125 s, amplitude ±10 %, bruit gaussien léger.\n"
                    "- **Validation** : 3 plis groupés par enregistrement ; un tracé d'apprentissage sert à l'arrêt précoce "
                    "dans chaque pli. Les scores hors pli règlent lissage, seuil et durée minimale. Modèle final "
                    "réentraîné sur les 12 tracés ; test lu une seule fois.\n"
                    "- **Incertitude** : 30 passages avec dropout actif (Monte-Carlo dropout).\n"
                    "- **Limite** : 12 tracés d'apprentissage, très peu pour un réseau ; pas de LORO pour le CNN."
                )

    with tabs["Espace des fenêtres"]:
        c = st.columns([1, 2])
        method = c[0].segmented_control("Projection", ["ACP", "t-SNE"], default="ACP", help="Le t-SNE prend une trentaine de secondes au premier calcul.") or "ACP"  # fmt: skip
        coloring = c[1].segmented_control("Couleur et symbole", ["Classe", "Erreurs de la forêt (test)"], default="Classe") or "Classe"  # fmt: skip
        slot = st.empty()
        skeleton(
            slot,
            520,
            "Projection des fenêtres…" + (" (t-SNE : environ 30 s)" if method == "t-SNE" else ""),
        )
        frame, explained = service.window_projection(method)
        if coloring == "Classe":
            frame["couleur"] = np.where(frame.Class == 1, "Crise", "Hors crise")
            palette = {
                "Hors crise": figures.OUTCOME_COLORS["Vrai négatif"],
                "Crise": figures.ANNOTATED,
            }
        else:
            frame = frame.merge(
                service.test_window_outcomes(), on=["record_id", "start_s"], how="inner"
            )
            frame["couleur"] = frame.outcome
            palette = figures.OUTCOME_COLORS
        slot.plotly_chart(
            figures.projection_scatter(frame, "couleur", palette, explained), width="stretch"
        )
        explain(
            "Chaque point est une fenêtre de 2 s décrite par les 16 caractéristiques enrichies (toutes les fenêtres en "
            "crise, et un échantillon des autres) ; chaque catégorie a sa couleur et son symbole. "
            + ("L'ACP est ajustée sur les enregistrements d'apprentissage seulement. " if method == "ACP" else "")
            + "Les erreurs se trouvent dans la zone où crises et activité de fond se touchent."
        )  # fmt: skip

    with tabs["Validation par enregistrement"]:
        loro = service.loro_table()
        if loro is None:
            state_box(
                "empty", "Résultats LORO absents", "Exécutez le notebook final pour les produire."
            )
            return
        configs = sorted(loro.configuration.unique(), key=lambda n: n != meta["configuration"])
        name = st.selectbox("Configuration", configs, format_func=eeg.config_label)
        part = loro[
            (loro.configuration == name) & (loro.postprocessing == "Post-traitement")
        ].copy()
        part["is_test"] = part.record_id.isin(meta["test_records"])
        part = part.sort_values("f1")
        tp, fn, fp, hours = (
            part.event_TP.sum(),
            part.event_FN.sum(),
            part.event_FP.sum(),
            part.hours.sum(),
        )
        worst = part.iloc[0]
        kpis([
            kpi("Crises retrouvées", f"{tp} / {tp + fn}", "16 tracés exclus tour à tour", "target", "green", ring(tp / (tp + fn), "var(--green)")),
            kpi("Fausses alarmes", fmt(fp / hours, 1, " / h"), "par heure de signal", "bell", "red", sparkline(part.false_alarms_per_hour, "var(--red)")),
            kpi("F1 médian des fenêtres", fmt(part.f1.median(), 2), "un point par tracé", "activity", "", sparkline(part.f1)),
            kpi("Pire enregistrement", worst.record_id, f"F1 {fmt(worst.f1, 2)}", "alert", "red", alert=worst.f1 < 0.3),
        ])  # fmt: skip
        st.plotly_chart(figures.loro_bars(part), width="stretch")
        explain(
            "Validation « un enregistrement exclu » : le modèle est réentraîné sans l'enregistrement, puis évalué sur "
            "lui, avec post-traitement réglé hors pli. C'est l'estimation la plus honnête pour un nouveau patient. "
            "Barres bleues hachurées : enregistrements du test réservé."
        )


# ======================================================================
# Comprendre une absence
# ======================================================================

FEATURE_MEANINGS = {
    "log_std": "Écart-type du signal (log) : son amplitude typique.",
    "log_amplitude": "Écart entre le minimum et le maximum (log).",
    "log_line_length": "Longueur de ligne : somme des variations d'un échantillon à l'autre (log). Sensible aux pointes.",
    "skewness": "Asymétrie de la distribution des valeurs.",
    "kurtosis": "Aplatissement : grand quand le signal contient des pointes isolées.",
    "log_power_delta": "Puissance 0,5–4 Hz (log).",
    "log_power_theta": "Puissance 4–8 Hz (log).",
    "log_power_alpha": "Puissance 8–13 Hz (log).",
    "log_power_beta": "Puissance 13–30 Hz (log).",
    "rel_delta": "Part de la puissance totale dans la bande delta.",
    "rel_theta": "Part de la puissance totale dans la bande thêta.",
    "rel_alpha": "Part de la puissance totale dans la bande alpha.",
    "rel_beta": "Part de la puissance totale dans la bande bêta.",
    "rel_spike_wave": "Part de la puissance dans la bande 2,5–4,5 Hz, celle des pointes-ondes.",
    "log_slow_fast_ratio": "Rapport entre ondes lentes et rapides (log).",
    "peak_frequency": "Fréquence où le spectre atteint son maximum, entre 1 et 30 Hz.",
}
DISPLAY_CHANNELS = ["EEGFp1", "EEGFp2", "EEGF7", "EEGF3", "EEGF4", "EEGF8", "EEGC3", "EEGC4", "EEGP3", "EEGP4", "EEGO1", "EEGO2"]  # fmt: skip


def demo_record():
    """Enregistrement de démonstration : du test réservé, annoté, avec 19 voies."""
    meta = service.metadata()
    for record in ["230718A-A", *meta["test_records"]]:
        if record in service.catalogue() and service.annotated_events(record):
            return record
    return None


def numbered(n, title, text=""):
    show(
        f'<div class="mstep anim"><span class="mstep-n">{n:02d}</span><div><h3>{esc(title)}</h3>'
        + (f"<p>{text}</p>" if text else "")
        + "</div></div>"
    )


def page_understand():
    topbar("Qu'est-ce qu'une absence ?", "Comprendre")
    record = demo_record()
    if record is None:
        state_box(
            "empty",
            "Aucun enregistrement annoté disponible",
            "Les extraits de cette page viennent du jeu de données.",
        )
        return
    events = service.annotated_events(record)
    a, b = max(events, key=lambda e: e[1] - e[0])
    length = min(10.0, b - a)
    start = a + (b - a - length) / 2
    calm = service.calm_excerpt(record, length)
    show(
        '<p class="lead anim">Les extraits de cette page sont réels. Ils viennent tous de '
        f'<span class="mono">{esc(record)}</span>, un enregistrement du test réservé, filtré comme pour '
        "la détection (0,5 à 40 Hz).</p>"
    )

    section("Un fond normal, puis une crise", "Même échelle")
    ts, zs, names = service.excerpt(record, start, start + length, DISPLAY_CHANNELS)
    scale = figures.excerpt_scale(zs)
    seizure = service.excerpt_profile(record, start, start + length)
    left, right = st.columns(2, gap="medium")
    if calm is not None:
        tn, zn, _ = service.excerpt(record, calm, calm + length, DISPLAY_CHANNELS)
        normal = service.excerpt_profile(record, calm, calm + length)
        with left:
            st.plotly_chart(figures.stacked_excerpt(tn, zn, names, scale, height=360, title="Fond normal"),
                            width="stretch", config={"displayModeBar": False})  # fmt: skip
            show(f'<p class="caption"><span class="mono">{fmt(calm, 0)}–{fmt(calm + length, 0)} s</span> · fréquence dominante '
                 f'<b>{fmt(normal["dominant_hz"], 1)} Hz</b> · amplitude efficace <b>{fmt(normal["rms"], 0)}</b></p>')  # fmt: skip
    with right:
        st.plotly_chart(figures.stacked_excerpt(ts, zs, names, scale, annotated=[(a, b)], height=360, title="Au cœur d'une crise annotée"),
                        width="stretch", config={"displayModeBar": False})  # fmt: skip
        show(f'<p class="caption"><span class="mono">{fmt(start, 0)}–{fmt(start + length, 0)} s</span> (crise annotée de {fmt(a, 0)} à {fmt(b, 0)} s) · '
             f'fréquence dominante <b>{fmt(seizure["dominant_hz"], 1)} Hz</b> · amplitude efficace <b>{fmt(seizure["rms"], 0)}</b></p>')  # fmt: skip
    if calm is not None:
        show(
            '<p class="body">Les deux extraits durent '
            f"{fmt(length, 0)} s et sont dessinés à la même échelle. À gauche, l'activité est irrégulière. À droite, "
            f"le tracé devient régulier : une pointe brève suivie d'une onde lente, environ {fmt(seizure['dominant_hz'], 1)} fois "
            f"par seconde, sur toutes les voies à la fois, avec une amplitude {fmt(seizure['rms'] / normal['rms'], 1)} fois plus grande.</p>"
            f'<p class="note">Fréquence dominante : maximum du spectre moyen des {seizure["channels"]} voies EEG '
            f"(résolution {fmt(seizure['resolution_hz'], 2)} Hz). Le fond normal est un passage d'amplitude médiane, "
            "choisi à plus de 30 s de toute crise.</p>"
        )

    section("Une décharge généralisée", "19 électrodes")
    center = (a + b) / 2
    reference = calm + length / 2 if calm is not None else max(1.0, a - 60)
    labels, during = service.spike_wave_map(record, center)
    _, before = service.spike_wave_map(record, reference)
    high = float(max(np.max(during), np.max(before)))
    left, right = st.columns([3, 2], gap="large")
    with left:
        show(
            '<div class="heads">'
            + figures.head_svg(
                dict(zip(labels, during, strict=True)),
                f"Pendant la crise · t = {fmt(center, 0)} s",
                0,
                high,
            )
            + figures.head_svg(
                dict(zip(labels, before, strict=True)),
                f"Fond normal · t = {fmt(reference, 0)} s",
                0,
                high,
            )
            + "</div>"
        )
    with right:
        show(
            '<p class="body" style="margin-top:0">Les électrodes suivent le système international 10-20 : la lettre donne la '
            "région (Fp frontopolaire, F frontale, C centrale, T temporale, P pariétale, O occipitale), un chiffre impair "
            "le côté gauche, un chiffre pair le côté droit, « z » la ligne médiane. Le nez est en haut du schéma.</p>"
            '<p class="body">La luminosité de chaque disque donne la part de la puissance comprise dans la bande '
            "2,5–4,5 Hz sur une fenêtre de 2 s. Pendant la crise, toutes les électrodes s'allument en même temps : "
            "c'est ce qui distingue une absence d'un artefact local, qui ne touche qu'une ou deux voies.</p>"
        )

    section("Des crises brèves et rares", "Cohorte")
    durations = service.seizure_durations()
    cohort = service.cohort_summary()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(
            figures.durations_figure(durations), width="stretch", config={"displayModeBar": False}
        )
    with right:
        kpis([
            kpi("Durée médiane", fmt(float(np.median(durations)), 0, " s"), f"de {fmt(durations.min(), 0)} à {fmt(durations.max(), 0)} s", "clock", "green"),
            kpi("Temps en crise", percent(cohort["positive_windows"] / cohort["windows"]), "des fenêtres de 2 s", "percent", "green",
                ring(cohort["positive_windows"] / cohort["windows"], "var(--green)")),
        ])  # fmt: skip
        show(
            '<p class="body">Une fenêtre de 2 s ne contient souvent qu\'une partie de crise : les plus courtes ne couvrent '
            "que trois ou quatre fenêtres. Et comme les crises occupent une petite part du tracé, un détecteur qui ne "
            "signale jamais rien aurait déjà raison presque tout le temps.</p>"
        )

    section("Ce qui rend la détection difficile", "En bref")
    loro = service.loro_table()
    meta = service.metadata()
    spread = ""
    if loro is not None:
        part = loro[
            (loro.configuration == meta["configuration"])
            & (loro.postprocessing == "Post-traitement")
        ]
        spread = f" En validation par patient, le F1 va de {fmt(part.f1.min(), 2)} à {fmt(part.f1.max(), 2)} selon l'enregistrement."
    items = [
        ("Rares", f"{percent(cohort['positive_windows'] / cohort['windows'])} des fenêtres seulement sont en crise : l'accuracy ne veut presque rien dire."),
        ("Brèves", f"Médiane de {fmt(float(np.median(durations)), 0)} s : quelques fenêtres mal classées suffisent à manquer une crise."),
        ("Variables", "Amplitude, rythme et voies dominantes changent d'un enfant à l'autre." + spread),
        ("Imitées", "Saturations, mouvements et clignements peuvent produire des fenêtres qui ressemblent à une décharge. La page Erreurs les montre."),
    ]  # fmt: skip
    show(
        '<div class="steps">'
        + "".join(
            f'<div class="step anim" style="--i:{k}"><b>{t}</b><span>{d}</span></div>'
            for k, (t, d) in enumerate(items)
        )
        + "</div>"
    )


# ======================================================================
# Méthode
# ======================================================================


def page_method():
    topbar("Du signal aux épisodes", "Méthode")
    record = demo_record()
    if record is None:
        state_box(
            "empty",
            "Aucun enregistrement annoté disponible",
            "La démonstration s'appuie sur le jeu de données.",
        )
        return
    meta = service.metadata()
    a, b = service.annotated_events(record)[0]
    show(
        '<p class="lead anim">La chaîne complète, appliquée pas à pas à '
        f'<span class="mono">{esc(record)}</span> (test réservé) avec les fonctions mêmes du notebook. '
        "Chaque figure est recalculée à l'ouverture de la page.</p>"
    )

    numbered(1, "Filtrer", "Un passe-haut de 0,5 Hz retire la dérive lente et la composante continue, propres à chaque "
             "enregistrement. Le modèle voit ainsi la forme du signal, pas son décalage.")  # fmt: skip
    time_values, raw, names, fs = service.multichannel(record)
    if eeg.CHANNEL in names:
        i0, i1 = np.searchsorted(time_values, [a - 8, b + 4])
        k = names.index(eeg.CHANNEL)
        brut = raw[k, i0:i1].astype(float)
        filtre = service.display_filter(record, False)[k, i0:i1].astype(float)
        stack = np.stack([brut - np.median(brut), filtre])
        st.plotly_chart(
            figures.stacked_excerpt(time_values[i0:i1], stack, ["Brut", "Filtré"], figures.excerpt_scale(stack), annotated=[(a, b)], height=230),
            width="stretch", config={"displayModeBar": False},
        )  # fmt: skip
        explain(
            f"Voie F7, {fmt(a - 8, 0)}–{fmt(b + 4, 0)} s ; le signal brut est recentré pour l'affichage."
        )

    numbered(2, "Découper en fenêtres", "Fenêtres de 2 s qui avancent d'une seconde (50 % de recouvrement). Une fenêtre "
             "est étiquetée « crise » quand au moins la moitié de ses échantillons tombe dans une crise annotée.")  # fmt: skip
    starts, coverage, labels = service.window_labels(record, a - 5, a + 7)
    t2, z2, _ = service.excerpt(record, a - 5, a + 7, [eeg.CHANNEL])
    if len(starts):
        st.plotly_chart(figures.windows_figure(t2, z2[0], starts, coverage, labels, (a, b)), width="stretch",
                        config={"displayModeBar": False})  # fmt: skip
        explain(
            "Chaque rectangle est une fenêtre ; le pourcentage est la part annotée. Cadre vert : étiquette « crise »."
        )

    numbered(3, "Décrire chaque fenêtre", "Seize nombres par fenêtre, calculés sur la voie F7 filtrée puis normalisés par la "
             "médiane et l'écart interquartile de l'enregistrement : chaque patient sert de référence à lui-même.")  # fmt: skip
    spectra = service.class_spectra(record)
    left, right = st.columns([3, 2], gap="large")
    with left:
        if spectra:
            st.plotly_chart(
                figures.spectra_figure(spectra), width="stretch", config={"displayModeBar": False}
            )
            explain(
                "Spectre moyen des fenêtres de cet enregistrement. Pendant les crises, la puissance monte autour de 3 Hz (bande encadrée)."
            )
    with right:
        table(
            pd.DataFrame({"Caractéristique": list(FEATURE_MEANINGS), "Ce qu'elle mesure": list(FEATURE_MEANINGS.values())}),
            mono=("Caractéristique",), max_height=360,
        )  # fmt: skip

    numbered(4, "Classer", f"{len(meta['train_records'])} enregistrements servent à l'apprentissage, {len(meta['test_records'])} "
             "sont mis de côté pour le test, sans jamais séparer les fenêtres d'un même enregistrement. Parmi 38 "
             f"configurations, la validation interne retient : <b>{esc(meta['label'])}</b>.")  # fmt: skip
    cohort = service.cohort_summary()
    kpis([
        kpi("Apprentissage", f"{len(meta['train_records'])} tracés", "plis groupés par enregistrement", "layers"),
        kpi("Test réservé", f"{len(meta['test_records'])} tracés", "lus une seule fois", "target"),
        kpi("AP en validation", fmt(meta["metrics"]["cv"]["average_precision"], 3), "critère de choix", "trend", "", ring(meta["metrics"]["cv"]["average_precision"])),
        kpi("Fenêtres en crise", percent(cohort["positive_windows"] / cohort["windows"]), "déséquilibre des classes", "percent", "green",
            ring(cohort["positive_windows"] / cohort["windows"], "var(--green)")),
    ])  # fmt: skip

    params = meta["params"]
    numbered(5, "Regrouper en épisodes", f"Le score est moyenné sur {params['smoothing']} fenêtres, comparé au seuil "
             f"{fmt(params['threshold'], 2)}, puis les suites de moins de {params['min_windows']} fenêtres sont effacées. "
             "Ces trois réglages ont été choisis sur des prédictions hors pli, jamais sur le test.")  # fmt: skip
    try:
        result = service.analyse_record(record)
    except (InputError, service.DataError) as exc:
        state_box("error", "Analyse impossible", esc(exc))
        return
    times, scores, above, kept = service.decision_steps(result)
    truth = service.annotated_events(record)
    detected = [(e["start_s"], e["end_s"]) for e in result["events"]]
    st.plotly_chart(figures.decision_figure(times, scores, above, kept, truth, detected, params["threshold"]),
                    width="stretch", config={"displayModeBar": False})  # fmt: skip
    explain(f"{int(above.sum())} fenêtres franchissent le seuil, {int(kept.sum())} restent après la durée minimale, "
            f"regroupées en {len(detected)} épisodes.")  # fmt: skip

    numbered(6, "Comparer aux annotations", "Un épisode détecté compte comme une crise retrouvée si son intersection avec "
             "la crise annotée représente au moins 30 % de leur union (IoU ≥ 0,3). Chaque crise ne peut être retrouvée "
             "qu'une fois ; un épisode sans crise correspondante est une fausse alarme.")  # fmt: skip
    matches = eeg.match_intervals(truth, detected)[0]
    if matches:
        i, j, iou = matches[0]
        (ta, tb), (da, db) = truth[i], detected[j]
        inter, union = max(0.0, min(tb, db) - max(ta, da)), max(tb, db) - min(ta, da)
        show(
            f'<p class="body">Exemple : la crise annotée de <b>{fmt(ta, 0)}</b> à <b>{fmt(tb, 0)}</b> s et l\'épisode détecté '
            f"de <b>{fmt(da, 0)}</b> à <b>{fmt(db, 0)}</b> s se recouvrent sur {fmt(inter, 0)} s pour une union de "
            f"{fmt(union, 0)} s : IoU = {fmt(iou, 2)}, la crise est retrouvée.</p>"
        )


# ======================================================================
# Erreurs
# ======================================================================


def open_in_viewer(record, center, label):
    set_context(record=record)
    st.session_state.focus = (record, float(center), label)
    st.session_state.goto = "visualisation"  # le changement de page se fait hors du rappel


def page_errors():
    if st.session_state.pop("goto", None):
        st.switch_page(PAGES["visualisation"])
    topbar("Ce que le modèle manque, ce qu'il voit de trop", "Erreurs")
    state = context()
    c = st.columns([2.2, 3.6, 2.4, 1.8], vertical_alignment="bottom")
    models = ["rf", "cnn"] if service.cnn_available() else ["rf"]
    model = (
        c[0].segmented_control(
            "Modèle", models, default=state["model"], format_func=MODELS_SHORT.get
        )
        or state["model"]
    )
    kind = c[1].segmented_control("Type", ["tout", "manquee", "fausse_alarme"], default="tout",
                                  format_func={"tout": "Tout", "manquee": "Manquées", "fausse_alarme": "Fausses alarmes"}.get) or "tout"  # fmt: skip
    include = c[3].toggle(
        "Apprentissage",
        help="Ajoute les enregistrements d'apprentissage (résultats optimistes : le modèle les a vus).",
    )
    frame = service.error_catalogue(model, include)
    records = ["Tous", *sorted(frame.record.unique())] if len(frame) else ["Tous"]
    record = c[2].selectbox("Enregistrement", records)
    shown = frame
    if len(frame):
        if kind != "tout":
            shown = shown[shown.kind == kind]
        if record != "Tous":
            shown = shown[shown.record == record]
    missed = int((frame.kind == "manquee").sum()) if len(frame) else 0
    false = int((frame.kind == "fausse_alarme").sum()) if len(frame) else 0
    kpis([
        kpi("Crises manquées", str(missed), "aucun épisode avec IoU ≥ 0,3", "x", "red", alert=missed > 0),
        kpi("Fausses alarmes", str(false), "épisodes sans crise annotée", "bell", "red", alert=false > 0),
        kpi("Enregistrements concernés", str(frame.record.nunique() if len(frame) else 0), "test réservé" + (" et apprentissage" if include else ""), "database"),
        kpi("Score maximal médian des manquées", fmt(float(frame[frame.kind == "manquee"].peak.median()), 2) if missed else "—",
            "proche du seuil = crise presque vue", "activity"),
    ])  # fmt: skip
    if shown.empty:
        state_box(
            "empty",
            "Aucune erreur pour ces filtres",
            "Changez de type, d'enregistrement ou de modèle.",
        )
        return
    show(
        f'<p class="caption">{len(shown)} erreur(s) affichée(s), triées par enregistrement puis par temps.</p>'
    )
    limit = st.session_state.get("err_limit", 8)
    cols = st.columns(2, gap="medium")
    for k, row in enumerate(shown.sort_values(["record", "start"]).head(limit).itertuples()):
        with cols[k % 2], st.container(key=f"errcard_{k}"):
            missed_row = row.kind == "manquee"
            show(
                '<div class="err-head">'
                + pill("Crise manquée" if missed_row else "Fausse alarme", "err")
                + f'<span class="mono">{esc(row.record)}</span><span class="dim">{fmt(row.start, 0)}–{fmt(row.end, 0)} s</span>'
                + (pill("Apprentissage", "warn") if row.status == "apprentissage" else "")
                + "</div>"
            )
            lo, hi = row.start - 5, row.end + 5
            t, z, names = service.excerpt(row.record, lo, hi, ["EEGFp1", "EEGF7", "EEGC3", "EEGO1"])
            bands = (
                {"annotated": [(row.start, row.end)]}
                if missed_row
                else {"detected": [(row.start, row.end)]}
            )
            st.plotly_chart(figures.stacked_excerpt(t, z, names, figures.excerpt_scale(z), height=200, **bands),
                            width="stretch", config={"displayModeBar": False}, key=f"errfig{k}")  # fmt: skip
            facts = [f"durée {fmt(row.end - row.start, 0)} s", f"score maximal {fmt(row.peak, 2)}"]
            facts.append(
                f"meilleur recouvrement IoU {fmt(row.overlap, 2)}"
                if row.overlap
                else "aucun recouvrement"
            )
            show(f'<p class="err-facts">{" · ".join(facts)}</p>')
            st.button("Ouvrir dans la visualisation", key=f"erropen{k}", icon=":material/open_in_new:",
                      on_click=open_in_viewer, args=(row.record, (row.start + row.end) / 2,
                                                     f"{'Crise manquée' if missed_row else 'Fausse alarme'} {fmt(row.start, 0)}–{fmt(row.end, 0)} s"))  # fmt: skip
    if len(shown) > limit:
        if st.button(f"Afficher {min(8, len(shown) - limit)} de plus", type="tertiary"):
            st.session_state.err_limit = limit + 8
            st.rerun()
    st.download_button(
        "Exporter la liste (CSV)",
        shown.to_csv(index=False),
        f"erreurs_{model}.csv",
        "text/csv",
        icon=":material/download:",
    )


# ======================================================================
# Limites
# ======================================================================


def page_limits():
    topbar("Ce que ces résultats permettent de dire", "Limites")
    meta = service.metadata()
    test = meta["metrics"]["test"]
    found, total = test["event_TP"], test["event_TP"] + test["event_FN"]
    low, high = service.wilson(found, total)
    rows = pd.DataFrame(service.catalogue().values())
    ages = sorted(
        {int(a) for v in rows.age.dropna() for a in str(v).split(" / ") if a.strip().isdigit()}
    )
    loro = service.loro_table()
    part = (
        loro[
            (loro.configuration == meta["configuration"])
            & (loro.postprocessing == "Post-traitement")
        ]
        if loro is not None
        else None
    )
    tp, n = (
        (int(part.event_TP.sum()), int(part.event_TP.sum() + part.event_FN.sum()))
        if part is not None
        else (0, 0)
    )
    llow, lhigh = service.wilson(tp, n)
    show('<p class="lead anim">Les chiffres de ce site sont exacts, mais ils reposent sur peu d\'enregistrements. Voici ce '
         "qu'ils permettent d'affirmer, et ce qu'ils ne permettent pas.</p>")  # fmt: skip

    def block(eyebrow, title, items):
        section(title, eyebrow)
        show(
            '<div class="lims">'
            + "".join(
                f'<div class="lim anim" style="--i:{k}"><b>{t}</b><p>{d}</p></div>'
                for k, (t, d) in enumerate(items)
            )
            + "</div>"
        )

    block("Taille", "Peu de données", [
        (f"{found} crises sur {total}, avec une large marge",
         f"Sur le test réservé, la forêt retrouve {found} crises sur {total}. L'intervalle de confiance à 95 % (Wilson) va de "
         f"<b>{percent(low)}</b> à <b>{percent(high)}</b> : avec {total} crises, une seule de plus ou de moins déplace le taux de "
         f"{fmt(100 / total, 0)} points."),
        ("La validation par patient resserre l'intervalle",
         f"En excluant chaque enregistrement à tour de rôle, {tp} crises sur {n} sont retrouvées, soit un intervalle de "
         f"<b>{percent(llow)}</b> à <b>{percent(lhigh)}</b>. C'est l'estimation la plus fiable du site." if n else "Résultats LORO absents."),
    ])  # fmt: skip
    block("Patients", "Des enregistrements très différents", [
        ("Une forte variabilité",
         f"En validation par patient, le F1 des fenêtres va de <b>{fmt(part.f1.min(), 2)}</b> "
         f"({esc(part.sort_values('f1').record_id.iloc[0])}) à <b>{fmt(part.f1.max(), 2)}</b>, médiane {fmt(part.f1.median(), 2)}. "
         "Un chiffre moyen cache des patients pour lesquels le détecteur échoue." if part is not None else "Résultats LORO absents."),
        ("Pas d'identifiant patient",
         "Rien ne garantit que deux enregistrements ne viennent pas du même enfant ; s'il y en a, la séparation par "
         "enregistrement reste un peu optimiste."),
        ("Une population étroite",
         f"Âges de {ages[0]} à {ages[-1]} ans, un seul service, une seule fréquence d'échantillonnage (256 Hz). Rien "
         "ne dit que le modèle tiendrait sur un autre appareil ou chez l'adulte." if ages else "Âges inconnus."),
    ])  # fmt: skip
    block("Méthode", "Des choix qui pèsent", [
        ("Une seule voie pour la forêt",
         "La forêt ne regarde que F7. Une décharge visible surtout en postérieur peut lui échapper ; le CNN voit les 19 voies "
         "mais son seuil, réglé hors pli, se transpose mal au modèle final."),
        ("Des annotations à la seconde, sans double relecture",
         "Les bornes des crises sont notées à la seconde par une seule personne. Un écart d'une ou deux secondes sur une crise "
         "de six secondes change beaucoup l'IoU ; d'où le seuil de 0,3 plutôt que 0,5."),
        ("Des réglages choisis hors pli",
         "Seuil, lissage et durée minimale ont été choisis sur les prédictions de validation, jamais sur le test. Le test n'a "
         "été lu qu'une fois ; le refaire après coup gonflerait les chiffres."),
    ])  # fmt: skip
    block("Pistes", "Pour aller plus loin", [
        ("Plus d'enregistrements", "Le levier principal : chaque nouveau patient resserre les intervalles et révèle de nouveaux cas difficiles."),
        ("Moyenner les modèles des plis", "Pour le CNN, prédire avec la moyenne des trois modèles de validation garderait le seuil cohérent avec les scores."),
        ("Une base externe", "Tester sur une base publique (CHB-MIT, TUH) en précisant qu'elle contient surtout d'autres types de crises."),
        ("Un usage d'aide, pas de diagnostic", "Le détecteur peut signaler des passages à relire ; la décision reste celle du neurologue."),
    ])  # fmt: skip


# ======================================================================
# Parcours de lecture
# ======================================================================

# Trois sections, trois couleurs : la couleur d'une page dit dans quelle partie du site on se trouve.
SECTION = {
    "home": "comprendre", "understand": "comprendre", "method": "comprendre",
    "visualisation": "explorer", "detection": "explorer", "errors": "explorer",
    "modeles": "evaluer", "limits": "evaluer",
}  # fmt: skip
SECTION_LABEL = {"comprendre": "Comprendre", "explorer": "Explorer", "evaluer": "Évaluer"}


def section_style(key):
    """Variables de la section courante : voile de page, lien actif, cartes de la page."""
    name = SECTION.get(key, "explorer")
    show(
        f"<style>:root {{ --section: var(--s-{name}); --section-ink: var(--s-{name}-ink); "
        f"--section-wash: var(--s-{name}-wash); }}</style>"
    )


READING_ORDER = [
    "home",
    "understand",
    "method",
    "visualisation",
    "detection",
    "errors",
    "modeles",
    "limits",
]


def page_nav(current):
    """Précédent / suivant dans l'ordre de lecture du site."""
    keys = [k for k in READING_ORDER if k in PAGES]
    key = next((k for k in keys if PAGES[k].url_path == current.url_path), None)
    if key is None:
        return
    i = keys.index(key)
    with st.container(key="pagenav"):
        left, right = st.columns(2)
        if i > 0:
            with left, st.container(key=f"pn-{SECTION[keys[i - 1]]}-prev"):
                show('<span class="pn-dir">Précédent</span>')
                st.page_link(
                    PAGES[keys[i - 1]], label=f"← {PAGES[keys[i - 1]].title}", width="stretch"
                )
        if i < len(keys) - 1:
            with right, st.container(key=f"pn-{SECTION[keys[i + 1]]}-next"):
                show('<span class="pn-dir right">Suivant</span>')
                st.page_link(
                    PAGES[keys[i + 1]], label=f"{PAGES[keys[i + 1]].title} →", width="stretch"
                )


# ======================================================================
# Navigation
# ======================================================================


def main():
    show(f"<style>{stylesheet()}</style>")
    problems = service.missing_setup()
    if problems:
        state_box("error", "Préparation nécessaire", "<br>".join(f"Fichier manquant <code>{p}</code> : lancer <code>{c}</code>" for p, c in problems))  # fmt: skip
        st.stop()
    try:
        service.detector()
    except RuntimeError as exc:
        state_box("error", "Modèle indisponible", esc(exc))
        st.stop()

    context()
    st.logo(str(STATIC / "logo.svg"), size="medium", icon_image=str(STATIC / "logo.svg"))
    PAGES.update(
        home=st.Page(page_home, title="Accueil", default=True),
        understand=st.Page(page_understand, title="Comprendre", url_path="comprendre"),
        method=st.Page(page_method, title="Méthode", url_path="methode"),
        visualisation=st.Page(page_visualisation, title="Visualisation", url_path="visualisation"),
        detection=st.Page(page_detection, title="Détection", url_path="detection"),
        errors=st.Page(page_errors, title="Erreurs", url_path="erreurs"),
        modeles=st.Page(page_models, title="Modèles", url_path="modeles"),
        limits=st.Page(page_limits, title="Limites", url_path="limites"),
    )  # fmt: skip
    navigation = st.navigation([PAGES[k] for k in READING_ORDER], position="top")
    section_style(
        next((k for k, page in PAGES.items() if page.url_path == navigation.url_path), "home")
    )
    navigation.run()
    page_nav(navigation)
    footer()


if __name__ == "__main__":
    main()
