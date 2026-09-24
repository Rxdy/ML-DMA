"""
Génère le rapport chronologique du projet au format PDF : rapport/rapport_ML_dechets.pdf

Les figures sont recalculées à partir des données du projet (data/processed) et de
la pipeline (scripts/09_pipeline.py), pour que les graphiques restent cohérents
avec le code.

À lancer depuis le dossier dechets/ :  python rapport/generer_rapport_pdf.py
Dépendances en plus de requirements.txt : reportlab. Polices : Noto Sans / Noto Serif
et JetBrains Mono (installées sur le système).
"""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------ paramètres
ETABLISSEMENT = os.environ.get("ETABLISSEMENT", "")
TITRE = "Machine learning : Déchets"
SOUS_TITRE = "Prédire la production de déchets ménagers par habitant et par département"
MEMBRES = sorted(["ALVES Rudy", "ANTHONY Josselin", "BENASSIE Noé", "TAVERNIER Florian"])  # ordre alphabétique
LOGO = "rapport/images/logo_irup.png"
MODULE = "DataScience & Machine Learning — apprentissage supervisé"
DATE = "24 septembre 2026"
DEPOT = "https://github.com/Rxdy/ML-DMA"
SORTIE = "rapport/rapport_ML_dechets.pdf"

# ------------------------------------------------------------------ polices et couleurs
NOTO = "/usr/share/fonts/truetype/noto/"
MONO = "/usr/share/fonts/truetype/jetbrains-mono-zorin-os/"
for nom, fichier in [("Sans", NOTO + "NotoSans-Regular.ttf"), ("Sans-B", NOTO + "NotoSans-Bold.ttf"),
                     ("Sans-I", NOTO + "NotoSans-Italic.ttf"), ("Sans-SB", NOTO + "NotoSans-SemiBold.ttf"),
                     ("Serif", NOTO + "NotoSerif-Regular.ttf"), ("Serif-B", NOTO + "NotoSerif-Bold.ttf"),
                     ("Serif-SB", NOTO + "NotoSerif-SemiBold.ttf"), ("Mono", MONO + "JetBrainsMono-Regular.ttf"),
                     ("Sym", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
    pdfmetrics.registerFont(TTFont(nom, fichier))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-B")
for f in [NOTO + "NotoSans-Regular.ttf", NOTO + "NotoSans-SemiBold.ttf"]:
    font_manager.fontManager.addfont(f)

VERT = colors.HexColor("#2F5236")        # accent : titres, filets
VERT_CLAIR = colors.HexColor("#E4ECDF")  # fonds d'encadrés
AMBRE = colors.HexColor("#9C6B2A")       # points d'attention
AMBRE_CLAIR = colors.HexColor("#F3E7D2")
ENCRE = colors.HexColor("#1B221C")
GRIS = colors.HexColor("#5B6559")
FILET = colors.HexColor("#C9D2C3")
MPL = {"vert": "#2F5236", "ambre": "#B07A2E", "gris": "#8B958A", "grille": "#DDE3D8", "encre": "#1B221C"}
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 9, "axes.edgecolor": MPL["gris"],
                     "axes.labelcolor": MPL["encre"], "xtick.color": MPL["gris"], "ytick.color": MPL["gris"],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": MPL["grille"], "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.unicode_minus": False})

# ------------------------------------------------------------------ styles de texte
S = {
    "corps": ParagraphStyle("corps", fontName="Sans", fontSize=10, leading=15, textColor=ENCRE, spaceAfter=7),
    "h1": ParagraphStyle("h1", fontName="Serif-B", fontSize=19, leading=24, textColor=VERT, spaceBefore=4, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Serif-SB", fontSize=13, leading=17, textColor=ENCRE, spaceBefore=12, spaceAfter=5),
    "etape": ParagraphStyle("etape", fontName="Mono", fontSize=8.5, leading=11, textColor=AMBRE, spaceAfter=2),
    "puce": ParagraphStyle("puce", fontName="Sans", fontSize=10, leading=14.5, textColor=ENCRE, leftIndent=14,
                           bulletIndent=3, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.8, leading=12, textColor=ENCRE),
    "cellh": ParagraphStyle("cellh", fontName="Sans-SB", fontSize=8.3, leading=11, textColor=VERT),
    "legende": ParagraphStyle("legende", fontName="Sans-I", fontSize=8.5, leading=11.5, textColor=GRIS,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "encadre": ParagraphStyle("encadre", fontName="Sans", fontSize=9.5, leading=14, textColor=ENCRE),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=11.5, textColor=ENCRE),
    "toc1": ParagraphStyle("toc1", fontName="Sans-SB", fontSize=10.5, leading=16, textColor=ENCRE, leftIndent=0),
    "toc2": ParagraphStyle("toc2", fontName="Sans", fontSize=9.5, leading=14, textColor=GRIS, leftIndent=16),
}


SYMBOLES = "−√≈→"  # absents des polices Noto : affichés avec DejaVu Sans


def sym(texte):
    for ch in SYMBOLES:
        texte = texte.replace(ch, f'<font name="Sym">{ch}</font>')
    return texte


def p(texte, style="corps"):
    return Paragraph(sym(texte), S[style])


def puces(items):
    return [Paragraph(sym(t), S["puce"], bulletText="•") for t in items]


def c(texte):
    """Code en ligne."""
    return f'<font name="Mono" size="8.8" color="#2F5236">{texte}</font>'


def tableau(lignes, largeurs, entete=True, surligne=()):
    data = [[Paragraph(sym(str(x)), S["cellh" if (entete and i == 0) else "cell"]) for x in ligne]
            for i, ligne in enumerate(lignes)]
    t = Table(data, colWidths=[w * cm for w in largeurs], repeatRows=1 if entete else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, FILET),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if entete:
        style += [("LINEBELOW", (0, 0), (-1, 0), 1, VERT)]
    for r in surligne:
        style += [("BACKGROUND", (0, r), (-1, r), VERT_CLAIR)]
    t.setStyle(TableStyle(style))
    return t


def encadre(titre, texte, attention=False):
    fond, trait = (AMBRE_CLAIR, AMBRE) if attention else (VERT_CLAIR, VERT)
    contenu = [Paragraph(f'<font name="Sans-B" color="{trait.hexval()}">{titre}</font>', S["encadre"]),
               Spacer(1, 2), Paragraph(sym(texte), S["encadre"])]
    t = Table([[contenu]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fond), ("LINEBEFORE", (0, 0), (0, -1), 2.5, trait),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 10)])


def code(texte):
    t = Table([[Preformatted(texte, S["code"])]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F6F2")),
                           ("BOX", (0, 0), (-1, -1), 0.5, FILET),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return KeepTogether([t, Spacer(1, 10)])


def figure(fig, largeur_cm, legende):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = largeur_cm * cm, largeur_cm * cm * ratio
    return KeepTogether([img, p(legende, "legende")])


def image_fichier(chemin, largeur_cm, legende):
    img = Image(chemin)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = largeur_cm * cm, largeur_cm * cm * ratio
    return KeepTogether([img, p(legende, "legende")])


class Chapitre(Paragraph):
    """Titre de chapitre, repris dans le sommaire."""
    def __init__(self, texte, niveau=0):
        super().__init__(texte, S["h1"] if niveau == 0 else S["h2"])
        self.niveau = niveau


# ------------------------------------------------------------------ données et figures
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> dechets/
ns = {}
exec(open("scripts/09_pipeline.py").read().split("\n# --- Validation croisée")[0].replace("print(", "(lambda *a, **k: None)("), ns)
train, test, X_train, y_train, X_test, y_test = (ns[k] for k in ["train", "test", "X_train", "y_train", "X_test", "y_test"])
lin = ns["pipelines"]["Régression linéaire"].fit(X_train, y_train)
pred = lin.predict(X_test)
reg = pd.read_csv("data/processed/processed_regression_dataset.csv").dropna(subset=["RATIO_DMA_lag1"])


def fr(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


_ex = train[(train["N_DEPT"].isin(["Ain", "Landes", "Paris"])) & (train["ANNEE"] == 2019)]
EXEMPLE_XY = [["Département", "VA_POPANNEE", "RATIO_DMA_lag1", "TONNAGE_DMA_lag1", "cluster", "y = RATIO_DMA"]] + [
    [r.N_DEPT, fr(r.VA_POPANNEE, 0), fr(r.RATIO_DMA_lag1), fr(r.TONNAGE_DMA_lag1, 0), str(int(r.cluster)),
     "<b>" + fr(r.RATIO_DMA) + "</b>"] for r in _ex.itertuples()]
CV = pd.read_csv("data/processed/detail_cv_plis.csv")
CALC = pd.read_csv("data/processed/detail_calcul_test.csv")
PAR_ANNEE = pd.read_csv("data/processed/detail_test_par_annee.csv")
COEFS = pd.read_csv("data/processed/detail_coefficients.csv")



def fig_baseline_par_annee():
    rows = [(a, r2_score(g.RATIO_DMA, g.RATIO_DMA_lag1), mean_absolute_error(g.RATIO_DMA, g.RATIO_DMA_lag1))
            for a, g in reg.groupby("ANNEE")]
    a, r2, mae = zip(*rows)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    couleurs = [MPL["vert"] if x <= 2019 else MPL["ambre"] for x in a]
    ax.bar([str(x) for x in a], r2, color=couleurs, width=0.6)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.015, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8.5, color=MPL["encre"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² de la baseline naïve")
    ax.grid(axis="x", visible=False)
    ax.text(5.5, 0.97, "années de test", ha="center", fontsize=8.5, color=MPL["ambre"])
    return fig


def fig_nuage():
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for annee, coul in [(2021, MPL["vert"]), (2023, MPL["ambre"])]:
        m = (test["ANNEE"] == annee).to_numpy()
        ax.scatter(y_test[m], pred[m], s=16, alpha=0.7, color=coul, label=str(annee), edgecolors="none")
    ax.plot([330, 920], [330, 920], ls="--", lw=1, color=MPL["gris"])
    ax.set_xlim(330, 920); ax.set_ylim(330, 920)
    ax.set_xlabel("ratio réel (kg/hab)"); ax.set_ylabel("ratio prédit (kg/hab)")
    ax.legend(frameon=False, loc="upper left")
    return fig


def fig_courbe_apprentissage():
    lc = pd.read_csv("data/processed/learning_curve.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(lc.n_train, lc.r2, marker="o", ms=4, color=MPL["vert"])
    ax.axhline(0.702, ls="--", lw=1, color=MPL["ambre"])
    ax.text(505, 0.6985, "baseline naïve (0,702)", fontsize=8, color=MPL["ambre"], ha="right", va="top")
    ax.set_ylim(0.65, 0.76)
    ax.set_xlabel("nombre de lignes d'entraînement"); ax.set_ylabel("R² sur le test")
    return fig


def fig_modeles():
    noms = ["Baseline\nmoyenne", "Baseline\nnaïve", "Régression\nlinéaire", "Random\nForest"]
    cv = [-0.013, 0.872, 0.865, 0.845]
    te = [-0.010, 0.702, 0.716, 0.705]
    x = np.arange(len(noms))
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.bar(x - 0.19, cv, 0.36, color=MPL["vert"], label="validation croisée (2013–2019)")
    ax.bar(x + 0.19, te, 0.36, color=MPL["ambre"], label="test (2021–2023)")
    for i in range(len(noms)):
        for dx, v in [(-0.19, cv[i]), (0.19, te[i])]:
            ax.text(i + dx, max(v, 0) + 0.02, f"{v:.3f}".replace(".", ","), ha="center", fontsize=7.5)
    ax.set_xticks(x, noms); ax.set_ylim(-0.05, 1.02); ax.set_ylabel("R²")
    ax.axhline(0, color=MPL["gris"], lw=0.8); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.16), ncol=2)
    return fig


def fig_clusters():
    cl = pd.read_csv("data/processed/processed_clusters_traitement.csv")
    traitements = ["Incinération avec récupération d'énergie", "Incinération sans récupération d'énergie",
                   "Valorisation matière", "Valorisation organique", "Stockage", "Stockage pour inertes", "Non précisé"]
    cols = [t for t in traitements if t in cl.columns]
    prof = cl.groupby("cluster")[cols].mean()
    prof = prof.div(prof.sum(axis=1), axis=0) * 100
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    noms = {0: "0 — Stockage", 1: "1 — Incinération", 2: "2 — Atypique", 3: "3 — Valorisation"}
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    gauche = np.zeros(len(prof))
    for col, coul in zip(cols, palette):
        ax.barh([noms[i] for i in prof.index], prof[col], left=gauche, color=coul, label=col, height=0.6)
        gauche += prof[col].to_numpy()
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("% du tonnage traité")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.28))
    return fig


# ------------------------------------------------------------------ mise en page
class Rapport(BaseDocTemplate):
    def __init__(self, fichier):
        super().__init__(fichier, pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                         topMargin=2.3 * cm, bottomMargin=2.2 * cm, title=TITRE,
                         author=", ".join(MEMBRES), subject=SOUS_TITRE)
        cadre = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="corps")
        self.addPageTemplates([PageTemplate("garde", [cadre], onPage=self.page_garde),
                               PageTemplate("contenu", [cadre], onPage=self.entete_pied)])

    def page_garde(self, cv, doc):
        l, h = A4
        cv.saveState()
        # logo de l'établissement, sur fond blanc
        hauteur_logo = 2.3 * cm
        cv.drawImage(LOGO, 2.1 * cm, h - 1.5 * cm - hauteur_logo, height=hauteur_logo,
                     width=hauteur_logo * 789 / 468, preserveAspectRatio=True, mask="auto")
        # bandeau titre
        haut, bas = h - 5.0 * cm, h - 13.4 * cm
        cv.setFillColor(VERT)
        cv.rect(0, bas, l, haut - bas, stroke=0, fill=1)
        cv.setFillColor(colors.HexColor("#C9DBC4"))
        cv.setFont("Mono", 9)
        y = haut - 1.3 * cm
        if ETABLISSEMENT:
            cv.drawString(2.3 * cm, y, ETABLISSEMENT.upper()); y -= 0.55 * cm
        cv.drawString(2.3 * cm, y, MODULE.upper())
        cv.setFillColor(colors.white)
        cv.setFont("Serif-B", 32)
        cv.drawString(2.3 * cm, haut - 3.4 * cm, "Machine learning :")
        cv.drawString(2.3 * cm, haut - 4.75 * cm, "Déchets")
        cv.setFont("Sans", 12)
        cv.setFillColor(colors.HexColor("#E4ECDF"))
        cv.drawString(2.3 * cm, haut - 6.1 * cm, SOUS_TITRE)

        cv.setFillColor(GRIS); cv.setFont("Mono", 8.5)
        cv.drawString(2.3 * cm, bas - 1.9 * cm, "MEMBRES DU GROUPE")
        for i, m in enumerate(MEMBRES):
            nom, prenom = m.split(" ", 1)
            yy = bas - (2.9 + 0.8 * i) * cm
            cv.setFillColor(ENCRE)
            cv.setFont("Sans-SB", 13); cv.drawString(2.3 * cm, yy, nom)
            cv.setFont("Sans", 13); cv.drawString(2.3 * cm + pdfmetrics.stringWidth(nom + " ", "Sans-SB", 13), yy, prenom)

        cv.setStrokeColor(FILET); cv.setLineWidth(0.6)
        cv.line(2.3 * cm, 5.2 * cm, l - 2.3 * cm, 5.2 * cm)
        cv.setFillColor(GRIS); cv.setFont("Mono", 8.5)
        cv.drawString(2.3 * cm, 4.4 * cm, "DATE"); cv.drawString(8.3 * cm, 4.4 * cm, "DÉPÔT GITHUB")
        cv.setFillColor(ENCRE); cv.setFont("Sans", 11)
        cv.drawString(2.3 * cm, 3.75 * cm, DATE)
        cv.setFillColor(VERT)
        cv.drawString(8.3 * cm, 3.75 * cm, DEPOT.replace("https://", ""))
        cv.linkURL(DEPOT, (8.3 * cm, 3.6 * cm, 15.5 * cm, 4.2 * cm))
        cv.setFillColor(GRIS); cv.setFont("Sans", 8.5)
        cv.drawString(2.3 * cm, 2.6 * cm, "Données : ADEME (SINOE) et INSEE, data.gouv.fr, licence Ouverte 2.0")
        cv.restoreState()

    def entete_pied(self, cv, doc):
        l, h = A4
        cv.saveState()
        cv.setFont("Sans", 8); cv.setFillColor(GRIS)
        cv.drawString(2.3 * cm, h - 1.4 * cm, TITRE)
        cv.drawRightString(l - 2.3 * cm, h - 1.4 * cm, "Groupe : " + ", ".join(m.split(" ", 1)[1] for m in MEMBRES))
        cv.setStrokeColor(FILET); cv.setLineWidth(0.5)
        cv.line(2.3 * cm, h - 1.6 * cm, l - 2.3 * cm, h - 1.6 * cm)
        cv.drawRightString(l - 2.3 * cm, 1.3 * cm, str(doc.page))
        cv.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Chapitre):
            texte = f.getPlainText()
            cle = f"ch{self.seq.nextf('ch')}"
            self.canv.bookmarkPage(cle)
            self.canv.addOutlineEntry(texte, cle, level=f.niveau)
            self.notify("TOCEntry", (f.niveau, texte, self.page, cle))


def etape(numero, date):
    return p(f"ÉTAPE {numero} · {date}", "etape")


# ------------------------------------------------------------------ contenu
H = []  # histoire du document
H += [NextPageTemplate("contenu"), PageBreak()]

# Sommaire
H += [p("Sommaire", "h1"), Spacer(1, 8)]
toc = TableOfContents()
toc.levelStyles = [S["toc1"], S["toc2"]]
toc.dotsMinLevel = 0
H += [toc, PageBreak()]

# Introduction
H += [Chapitre("Introduction"),
      p("Ce rapport retrace, dans l'ordre où nous les avons prises, les décisions du projet : le choix du thème, "
        "la recherche et la consolidation des données, leur nettoyage, le choix de ce que l'on prédit, puis la "
        "construction et l'évaluation des modèles. Pour chaque étape, nous indiquons ce que nous avons fait et "
        "<b>pourquoi</b>."),
      p("<b>La question posée</b> : peut-on prédire combien de kilos de déchets ménagers chaque habitant d'un "
        "département produira lors de la prochaine enquête, à partir de ce que l'on sait des enquêtes précédentes ?"),
      encadre("Le résultat en trois phrases",
              "La production de déchets par habitant d'un département est très stable d'une enquête à l'autre : "
              "la simple règle « comme à l'enquête précédente » explique déjà 70 % à 87 % des écarts entre départements. "
              "Nos modèles (régression linéaire, Random Forest) font aussi bien, mais pas mieux de façon utile. "
              "Avec les variables disponibles, l'historique du département contient presque toute l'information."),
      p("Le code, les données et le journal de bord détaillé sont disponibles sur le dépôt "
        f'<link href="{DEPOT}" color="#2F5236"><u>{DEPOT.replace("https://", "")}</u></link>. '
        "Chaque étape correspond à un script numéroté du dossier " + c("dechets/scripts/") + "."),
      Spacer(1, 6), p("Chronologie en un coup d'œil", "h2"),
      tableau([["Étape", "Ce qui a été fait", "Script"],
               ["1", "Choix du thème : les déchets ménagers et assimilés", "—"],
               ["2", "Source principale ADEME (SINOE) : exploration", c("01_explore.py")],
               ["3", "Consolidation avec d'autres sources (population, 2023, revenu)", c("07_income_prep.py")],
               ["4", "Choix de ce que l'on prédit : le ratio de déchets par habitant", c("03_regression_prep.py")],
               ["5", "Nettoyage : formats, clés, valeurs manquantes, doublons", c("01") + ", " + c("03")],
               ["6", "Suppression des données inutiles", c("03") + ", " + c("05")],
               ["7", "Les entrées X et la sortie y du modèle", c("09_pipeline.py")],
               ["8", "Mise en forme : profil de traitement, historique, découpage train/test", c("02") + " à " + c("05")],
               ["9", "Baseline : le repère à battre", c("06_train_model.py")],
               ["10", "Premiers modèles, courbe d'apprentissage, test du revenu", c("06") + ", " + c("08")],
               ["11", "Preprocessor et pipeline", c("09_pipeline.py")],
               ["12", "Évaluation détaillée et interprétation", c("09") + ", " + c("10")],
               ["13", "Vérification terrain, et pourquoi ne pas estimer 2025", c("09_pipeline.py")],
               ["14", "Comment améliorer le modèle : analyse des erreurs et pistes", c("11_analyse_erreurs.py")],
               ], [1.3, 10.6, 4.5]),
      PageBreak()]

# 1. Thème
H += [etape(1, "16 septembre 2026"), Chapitre("1. Choix du thème : les déchets ménagers"),
      p("Les <b>déchets ménagers et assimilés (DMA)</b> sont les déchets collectés par le service public de gestion "
        "des déchets : ceux des ménages, et ceux des petites activités économiques (commerces, artisans) collectés "
        "avec eux. Dans nos données, ils sont répartis en 7 catégories :"),
      *puces(["ordures ménagères résiduelles (la poubelle « classique ») ;",
              "matériaux recyclables (emballages, papiers, verre) ;",
              "déchets verts et biodéchets ;",
              "encombrants ;",
              "déchets dangereux (y compris les équipements électriques et électroniques) ;",
              "déblais et gravats ;",
              "autres."]),
      p("Chaque tonne est ensuite traitée de l'une de 7 façons : incinération avec ou sans récupération d'énergie, "
        "valorisation matière (recyclage), valorisation organique (compostage, méthanisation), stockage, stockage "
        "pour déchets inertes, ou traitement non précisé."),
      p("Pourquoi ce thème", "h2"),
      *puces(["<b>Un enjeu concret</b> : chaque territoire organise sa collecte et son traitement, et la réduction "
              "des déchets fait l'objet d'objectifs publics. Anticiper les volumes a une utilité réelle.",
              "<b>Des données ouvertes et officielles</b> : l'ADEME publie ces chiffres pour tous les départements "
              "français depuis 2009, sous licence ouverte.",
              "<b>Une question de prédiction naturelle</b> : combien de déchets produira un département ? La "
              "réponse est un nombre, connu pour les années passées, ce qui en fait un problème d'apprentissage supervisé.",
              "<b>Des ordres de grandeur parlants</b> : environ 38 à 41 millions de tonnes par an en France "
              "(gravats compris), soit un peu plus de 500 kg par habitant et par an en moyenne hors gravats."]),
      PageBreak()]

# 2. Source principale
H += [etape(2, "16 septembre 2026"), Chapitre("2. La source principale : SINOE (ADEME)"),
      p("Notre point de départ est le jeu de données <b>« Destination des DMA collectés par type de traitement »</b> "
        "de l'ADEME, issu de sa base SINOE et publié sur data.gouv.fr (licence Ouverte 2.0) : fichier "
        + c("sinoe_dma.csv") + "."),
      tableau([["Caractéristique", "Valeur"],
               ["Taille", "14 660 lignes × 10 colonnes"],
               ["Format", "« long » : une ligne = année × département × type de déchet × type de traitement → tonnage"],
               ["Couverture", "101 départements (métropole et outre-mer), 14 régions"],
               ["Période", "2009 à 2021, années impaires uniquement (7 enquêtes)"],
               ["Qualité", "aucune valeur manquante, aucun doublon, 54 tonnages nuls, aucun négatif"]],
              [4.0, 12.4]),
      Spacer(1, 8), p("Ce qui était intéressant", "h2"),
      *puces(["<b>Le niveau de détail</b> : pour chaque département, on sait quelle part des déchets part en "
              "incinération, en recyclage, en stockage… Cela permet de décrire la <b>politique de traitement</b> de "
              "chaque territoire, et de regrouper les départements qui se ressemblent.",
              "<b>La profondeur temporelle</b> : 7 enquêtes sur 13 ans pour chaque département."]),
      p("Ce qui posait question", "h2"),
      *puces(["<b>Seulement des années impaires, et rien après 2021</b>, alors que la page data.gouv.fr annonçait une "
              "mise à jour récente. Vérification faite sur les quatre jeux SINOE de l'ADEME : tous indiquent "
              "« années impaires à partir de 2009 ». L'enquête est <b>biennale par construction</b> : ce n'est pas "
              "une donnée manquante à combler. La date de mise à jour correspond à une republication du fichier.",
              "<b>Pas de population</b> : impossible de comparer un grand et un petit département, ni de raisonner "
              "par habitant avec ce seul fichier.",
              "<b>Pas de 2023</b> : le fichier s'arrête à l'enquête 2021."]),
      p("Ces deux manques ont motivé l'étape suivante : consolider avec d'autres sources."),
      PageBreak()]

# 3. Consolidation
H += [etape(3, "16 septembre 2026"), Chapitre("3. Consolider les données"),
      p("Nous avons cherché d'autres jeux publiés par l'ADEME et l'INSEE, avec une contrainte : pouvoir les relier "
        "proprement à nos départements par le code département (" + c("C_DEPT") + ")."),
      tableau([["Fichier", "Lignes", "Période", "Apport", "Décision"],
               [c("sinoe_dma.csv"), "14 660", "2009–2021", "Tonnage par type de déchet et de traitement", "Clustering"],
               [c("chiffres_cles_hors_gravats"), "800", "2009–2023",
                "Tonnage, <b>population</b> et <b>ratio kg/habitant</b> déjà calculés. Seule source avec <b>2023</b>",
                "<b>Base de la régression</b>"],
               [c("chiffres_cles_avec_gravats"), "701", "2009–2021", "Mêmes indicateurs, gravats compris",
                "Contrôle, puis écarté"],
               [c("insee_population (âge, sexe)"), "101 dép.", "1975–2023", "Pyramide des âges par département",
                "Non utilisé"],
               [c("insee_revenu_median"), "491", "2013–2021", "Revenu médian (Filosofi), assemblé à partir de 5 formats différents",
                "Testé (étape 10)"]],
              [3.9, 1.4, 1.8, 6.0, 3.3], surligne=(2,)),
      Spacer(1, 8), p("Pourquoi le fichier « hors gravats » comme base", "h2"),
      *puces(["Il apporte la <b>population</b> et le <b>ratio par habitant</b>, ce qui manquait au fichier principal.",
              "C'est le <b>seul qui va jusqu'en 2023</b> : 8 enquêtes au lieu de 7.",
              "Les gravats viennent surtout des chantiers (BTP), pas du comportement des ménages. Ils ajoutent de la "
              "variabilité sans rapport avec la population : l'écart-type du ratio passe de 80,3 kg/hab sans gravats "
              "à 100,6 kg/hab avec."]),
      p("Vérifications de cohérence entre sources", "h2"),
      *puces(["Les deux fichiers « chiffres-clés » donnent <b>exactement la même population</b> pour chaque département "
              "et chaque année (0 écart sur 701 lignes comparées).",
              "Les codes et noms de départements et de régions sont identiques d'un fichier à l'autre.",
              "<b>Piège évité</b> : la colonne " + c("TONNAGE_DMA") + " porte le même nom dans deux fichiers mais pas le "
              "même périmètre. En additionnant le détail de " + c("sinoe_dma.csv") + ", on trouve 10,5 % d'écart en "
              "moyenne avec le fichier hors gravats (jusqu'à 28 % dans les Landes). En retirant les gravats du calcul, "
              "l'écart tombe à 0,03 %. La cible de la régression est donc toujours lue dans le fichier hors gravats, "
              "jamais recalculée."]),
      encadre("Et le revenu des ménages ?",
              "Plus tard dans le projet, pour tenter d'améliorer les prédictions, nous avons cherché une variable "
              "explicative supplémentaire. Kaggle ne proposait que des jeux d'images de déchets ou des données mondiales "
              "sans lien avec les départements français. Nous nous sommes tournés vers l'INSEE (revenu médian Filosofi), "
              "publié dans un format différent chaque année : 5 fichiers ont dû être harmonisés "
              "(script " + c("07_income_prep.py") + ")."),
      PageBreak()]

# 4. Que prédire
H += [etape(4, "16 septembre 2026"), Chapitre("4. Qu'allons-nous prédire ?"),
      p("Avant d'aller plus loin dans le nettoyage, il fallait fixer l'objectif : c'est lui qui détermine quelles "
        "données sont vraiment nécessaires. Trois options ont été envisagées :"),
      tableau([["Option", "Idée", "Décision"],
               ["1", "Régression du ratio de déchets par habitant, par département et par année, "
                     "plus un clustering des départements selon leur profil de traitement", "<b>Retenue</b>"],
               ["2", "Classification du mode de traitement dominant de chaque département", "Écartée"],
               ["3", "Séries temporelles : extrapoler au-delà de 2023", "Écartée : 8 points par département, trop peu"]],
              [1.5, 10.4, 4.5], surligne=(1,)),
      Spacer(1, 8), p("La cible : le ratio par habitant, pas le tonnage", "h2"),
      p("La corrélation entre le tonnage d'un département et sa population vaut <b>0,97</b>. Prédire le tonnage "
        "brut reviendrait à réapprendre une multiplication (ratio × population) : le modèle aurait l'air excellent "
        "sans avoir rien appris d'utile. C'est une <b>fuite de données</b>. En prédisant " + c("RATIO_DMA") +
        " (kg par habitant), on retire l'effet de taille, et le modèle doit expliquer les vraies différences de "
        "comportement entre départements."),
      p("Ce que disent les corrélations avec la cible", "h2"),
      tableau([["Variable", "Corrélation", "Lecture"],
               ["Ratio à l'enquête précédente", "0,91", "Signal très fort : un département change peu en deux ans"],
               ["Population", "−0,17", "Signal faible : les départements peuplés produisent un peu moins par habitant"],
               ["Profil de traitement (cluster)", "0,13", "Signal faible"],
               ["Année", "0,02", "Quasi nul : il n'y a pas de tendance nationale nette"]],
              [5.2, 2.3, 8.9]),
      Spacer(1, 6),
      image_fichier("data/figures/correlation_matrix.png", 9.0,
                    "Figure 1 — Matrice de corrélation des variables du jeu de régression (script 04_correlation.py)."),
      p("Conséquence directe : le temps sera représenté par <b>la valeur de l'enquête précédente</b> du même "
        "département, pas par le numéro de l'année. C'est un problème de <b>régression supervisée</b> : chaque "
        "exemple est un couple département × année dont on connaît la réponse."),
      PageBreak()]

# 5. Nettoyage
H += [etape(5, "16 septembre 2026"), Chapitre("5. Nettoyage des données"),
      p("Chaque opération répond à un constat de l'exploration, aucune n'est appliquée « par défaut »."),
      tableau([["Constat", "Traitement", "Pourquoi"],
               ["Les fichiers « chiffres-clés » utilisent " + c(";") + " et la virgule décimale",
                "Lecture avec " + c("sep=';', decimal=','") + " et encodage UTF-8 avec BOM",
                "Sinon les nombres sont lus comme du texte"],
               ["Codes département de longueur variable (" + c("1") + " / " + c("01") + ")",
                "Codes complétés à 2 caractères", "Sinon la jointure perd des départements sans message d'erreur"],
               ["Aucune valeur manquante dans les sources", "Rien à imputer", "—"],
               ["54 tonnages à 0, aucun négatif", "Conservés",
                "Absence réelle de tonnage pour une combinaison donnée, pas une erreur"],
               ["Années paires absentes", "Non imputées", "L'enquête est biennale : il n'y a rien à combler"],
               ["Doublons possibles", "Contrôle à deux niveaux (voir ci-dessous)", "Un doublon fausserait l'historique"]],
              [5.0, 5.4, 6.0]),
      Spacer(1, 8), p("Les doublons : contrôlés à deux niveaux", "h2"),
      *puces(["<b>Lignes strictement identiques</b> : 0.",
              "<b>Clés répétées</b>, le cas le plus dangereux : une même combinaison déclarée deux fois avec des valeurs "
              "différentes, qu'un simple " + c("duplicated()") + " ne détecte pas. Clé année × département × type de "
              "déchet × type de traitement pour le fichier principal, département × année pour les autres : 0 doublon.",
              "Un même département apparaît bien une fois par enquête. Ce n'est pas un doublon : ce sont des "
              "observations successives du même territoire (données de panel)."]),
      encadre("Pourquoi c'est important ici",
              "Une clé en double casserait la variable d'historique : « l'enquête précédente » serait la ligne "
              "dupliquée elle-même, c'est-à-dire la réponse donnée au modèle. Une jointure sur une clé en double "
              "multiplierait aussi les lignes en silence. Ces contrôles sont inscrits dans le code : unicité vérifiée "
              "par " + c("assert") + ", jointures déclarées avec " + c("validate='many_to_one'") + " et nombre de "
              "lignes comparé avant et après.", attention=True),
      PageBreak()]

# 6. Données inutiles
H += [etape(6, "16 septembre 2026"), Chapitre("6. Se débarrasser des données inutiles"),
      p("Garder une donnée « au cas où » alourdit le modèle et peut l'induire en erreur. Voici ce qui a été retiré, "
        "et pourquoi."),
      tableau([["Élément écarté", "Raison"],
               [c("L_REGION") + ", " + c("C_REGION") + ", " + c("N_DEPT"),
                "Libellés et doublons de codes : ils identifient un département, ils ne l'expliquent pas. "
                + c("C_DEPT") + " est gardé comme clé de jointure uniquement."],
               [c("ANNEE") + " comme variable d'entrée", "Corrélation de 0,02 avec la cible. Remplacée par l'historique du département."],
               [c("TONNAGE_DMA") + " comme cible", "Quasi-copie de la population (corrélation 0,97) : fuite de données."],
               ["Fichier « avec gravats »", "Sert au contrôle croisé de la population, puis écarté : les gravats "
                                            "brouillent la comparaison par habitant."],
               ["Fichier INSEE population par âge et sexe", "Redondant avec la population déjà présente dans les fichiers ADEME."],
               ["Les 101 lignes de 2009", "Première enquête : pas d'enquête précédente, donc pas d'historique possible."]],
              [5.4, 11.0]),
      Spacer(1, 8),
      p("Après ce tri, le jeu de données de régression compte <b>699 lignes</b> (101 départements × 7 enquêtes de 2011 "
        "à 2023, moins quelques trous ponctuels de l'enquête) et <b>4 variables d'entrée</b> :"),
      tableau([["Variable", "Rôle", "Nature"],
               [c("RATIO_DMA_lag1"), "Ratio de l'enquête précédente, même département", "Numérique"],
               [c("TONNAGE_DMA_lag1"), "Tonnage de l'enquête précédente", "Numérique"],
               [c("VA_POPANNEE"), "Population du département", "Numérique"],
               [c("cluster"), "Profil de traitement du département (étape 8)", "Catégorielle"],
               [c("RATIO_DMA"), "<b>Cible</b> : kg de déchets par habitant", "Numérique"]],
              [4.2, 8.6, 3.6], surligne=(5,)),
      PageBreak()]

# 7. X et y
H += [etape(7, "16 septembre 2026"), Chapitre("7. Les entrées X et la sortie y"),
      p("Une fois les données nettoyées et triées, on les sépare en deux, selon la convention de l'apprentissage supervisé :"),
      *puces(["<b>X, les données d'entrée</b> (les « features ») : ce que le modèle connaît au moment de prédire. "
              "C'est un tableau : une ligne par exemple, une colonne par variable.",
              "<b>y, la donnée de sortie</b> (la « cible ») : ce que le modèle doit apprendre à prédire. C'est une "
              "seule colonne, avec une valeur par ligne de X."]),
      p("Apprendre, c'est trouver une fonction <b>f</b> telle que <b>f(X) ≈ y</b> sur les exemples connus, puis "
        "l'appliquer à des X dont on ne connaît pas encore le y."),
      p("Dans notre projet", "h2"),
      tableau([["", "Contenu", "Dimensions"],
               ["<b>X</b>", c("VA_POPANNEE") + ", " + c("RATIO_DMA_lag1") + ", " + c("TONNAGE_DMA_lag1") + ", " + c("cluster"),
                "699 lignes × 4 colonnes"],
               ["<b>y</b>", c("RATIO_DMA") + " : kg de déchets par habitant", "699 valeurs"],
               ["X_train, y_train", "Enquêtes 2011 à 2019 : le modèle apprend dessus", "501 × 4, et 501 valeurs"],
               ["X_test, y_test", "Enquêtes 2021 et 2023 : on compare les prédictions à y_test", "198 × 4, et 198 valeurs"]],
              [3.0, 9.6, 3.8]),
      Spacer(1, 8),
      p("Exemple réel : trois lignes de X_train et leur y (enquête 2019)", "h2"),
      tableau(EXEMPLE_XY, [2.6, 2.6, 2.9, 3.3, 1.6, 3.4]),
      Spacer(1, 6),
      p("Pour l'Ain en 2019, le modèle reçoit la population, le ratio et le tonnage de l'enquête précédente (2017) "
        "et le profil de traitement 1 (incinération). Il doit en déduire y, le ratio 2019. Une ligne = un département "
        "à une date ; " + c("C_DEPT") + " et " + c("ANNEE") + " servent à construire et à découper les données, mais "
        "ne font pas partie de X."),
      encadre("Pourquoi ce découpage compte",
              "Toute information présente dans X doit être connue <b>avant</b> la date prédite. C'est pour cela que X "
              "contient le ratio de l'enquête <b>précédente</b> et pas celui de l'année prédite : sinon, on donnerait la "
              "réponse au modèle. Après le preprocessor (étape 11), X passe de 4 à 7 colonnes, le cluster étant "
              "éclaté en 4 colonnes 0/1."),
      PageBreak()]

# 7. Mise en forme
H += [etape(8, "16 septembre 2026"), Chapitre("8. Mettre les données en forme"),
      p("Profil de traitement et clustering", "h2"),
      p("Pour chaque département, le tonnage est réparti entre les 7 modes de traitement, puis converti en "
        "pourcentages (chaque ligne somme à 100). Sans cette mise en pourcentage, un grand département ressemblerait "
        "à un autre grand département simplement par sa taille. Un k-means regroupe ensuite les départements aux "
        "profils proches. Le nombre de groupes, k = 4, a été choisi par score de silhouette (0,276, le meilleur "
        "entre 2 et 7). Ce groupe devient une variable d'entrée de la régression (détail en annexe A)."),
      p("Variables d'historique (lag)", "h2"),
      p("Pour chaque département, " + c("RATIO_DMA_lag1") + " reprend le ratio de l'enquête précédente "
        "(" + c("groupby('C_DEPT').shift(1)") + " sur les données triées par année). C'est la traduction directe de la "
        "corrélation de 0,91 observée à l'étape 4."),
      p("Découpage entraînement / test : par année, pas au hasard", "h2"),
      tableau([["Jeu", "Années", "Lignes", "Rôle"],
               ["Entraînement", "2011 à 2019", "501", "Le modèle apprend sur ces données"],
               ["Test", "2021 et 2023", "198", "Jamais vues pendant l'entraînement : simulent le futur"]],
              [3.2, 3.2, 2.0, 8.0]),
      Spacer(1, 6),
      p("Un découpage aléatoire mettrait des années récentes dans l'entraînement et des années anciennes dans le "
        "test : le modèle « verrait le futur ». Le découpage par année reproduit la vraie situation de prédiction. "
        "Contrôle inscrit dans le code : aucun couple département × année n'est à la fois dans les deux jeux."),
      p("Mise à l'échelle", "h2"),
      p("Les variables numériques sont ramenées entre 0 et 1 (" + c("MinMaxScaler") + "), sinon la population "
        "(des centaines de milliers) écraserait le ratio (quelques centaines). Le scaler est calibré <b>uniquement "
        "sur l'entraînement</b>. Conséquence attendue : sur le test, certaines valeurs dépassent légèrement 1 "
        "(jusqu'à 1,03), parce que 2021 et 2023 atteignent des niveaux jamais vus entre 2011 et 2019. Calibrer sur "
        "toutes les données aurait masqué ce signal et fait fuiter des informations du test."),
      PageBreak()]

# 8. Baseline
H += [etape(9, "16 septembre 2026"), Chapitre("9. Établir la baseline"),
      p("Avant d'entraîner le moindre modèle, il faut un <b>repère</b> : sans lui, impossible de savoir si un score "
        "est bon. Un R² de 0,70 peut être excellent… ou ne rien valoir si une règle triviale fait aussi bien."),
      tableau([["Baseline", "Règle", "Rôle"],
               ["<b>Naïve</b>", "Prédire le ratio de l'enquête précédente, tel quel",
                "Le <b>vrai repère à battre</b> : la corrélation de 0,91 le rend déjà très fort"],
               ["Moyenne", "Prédire toujours la moyenne de l'entraînement (" + c("DummyRegressor") + ")",
                "Plancher absolu : un modèle qui fait moins bien n'a rien appris. Ajoutée à l'étape 11"]],
              [2.6, 6.6, 7.2], surligne=(1,)),
      Spacer(1, 8),
      p("Sur le test (2021 et 2023), la baseline naïve obtient un <b>R² de 0,702</b> et se trompe en moyenne de "
        "<b>36,5 kg par habitant</b> (MAE). La baseline moyenne obtient un R² de −0,010 : elle n'explique rien."),
      p("Les métriques utilisées", "h2"),
      p("Pour chaque ligne, l'erreur est l'écart entre la valeur réelle <b>y</b> et la valeur prédite <b>ŷ</b>. "
        "On résume les n erreurs de trois façons :"),
      tableau([["Métrique", "Formule", "Ce qu'elle dit"],
               ["<b>MAE</b>", "MAE = (1/n) × Σ |y − ŷ|",
                "De combien de kg/hab la prédiction se trompe en moyenne. La plus lisible."],
               ["<b>RMSE</b>", "RMSE = √[ (1/n) × Σ (y − ŷ)² ]",
                "Même unité, mais les grosses erreurs pèsent plus lourd (elles sont mises au carré)."],
               ["<b>R²</b>", "R² = 1 − SS<sub>res</sub> / SS<sub>tot</sub><br/>SS<sub>res</sub> = Σ (y − ŷ)²<br/>"
                             "SS<sub>tot</sub> = Σ (y − ȳ)²",
                "Part des écarts entre départements expliquée. 1 = parfait ; 0 = pas mieux que prédire la moyenne ȳ ; "
                "négatif = pire que la moyenne."]],
              [2.0, 5.6, 8.8]),
      Spacer(1, 6),
      KeepTogether([p("Légende des symboles et abréviations", "h2"),
      tableau([["Symbole", "Se lit", "Signification"],
               ["n", "« n »", "Nombre de lignes évaluées (198 sur le test)"],
               ["y", "« y »", "Valeur réelle (ce que l'ADEME a mesuré)"],
               ["ŷ", "« y chapeau »", "Valeur prédite par le modèle"],
               ["ȳ", "« y barre »", "Moyenne des valeurs réelles"],
               ["y − ŷ", "", "Erreur (ou résidu) sur une ligne : positive si le modèle sous-estime"],
               ["Σ", "« somme »", "Somme sur toutes les lignes"],
               ["| … |", "« valeur absolue »", "La valeur sans son signe : |−40| = 40"],
               ["√", "« racine carrée »", "Ramène une valeur au carré dans l'unité d'origine (kg/hab)"],
               ["SS", "<i>Sum of Squares</i>", "Somme des carrés"],
               ["SS<sub>res</sub>", "« somme des carrés des résidus »",
                "Σ (y − ŷ)² : ce que le modèle n'explique pas"],
               ["SS<sub>tot</sub>", "« somme des carrés totale »",
                "Σ (y − ȳ)² : la variabilité totale des valeurs réelles autour de leur moyenne"],
               ["MAE", "<i>Mean Absolute Error</i>", "Erreur absolue moyenne"],
               ["RMSE", "<i>Root Mean Squared Error</i>", "Racine de l'erreur quadratique moyenne"],
               ["R²", "« R deux »", "Coefficient de détermination"]],
              [2.4, 5.0, 9.0])]),
      Spacer(1, 6),
      p("Le calcul complet, pas à pas, pour chaque modèle, pli et année, est donné en annexe B. Les autres termes "
        "techniques sont définis dans le glossaire."),
      PageBreak()]

# 9. Premiers modèles
H += [etape(10, "16 septembre 2026"), Chapitre("10. Premiers modèles"),
      p("Deux modèles, du plus simple au plus élaboré, entraînés sur les mêmes 4 variables (script "
        + c("06_train_model.py") + ") :"),
      *puces(["<b>Régression linéaire</b> : une somme pondérée des variables. Simple et interprétable.",
              "<b>Random Forest</b> : 300 arbres de décision (profondeur maximale 6). Capte les relations non "
              "linéaires et mesure l'importance de chaque variable."]),
      tableau([["Modèle (test 2021–2023)", "MAE", "RMSE", "R²"],
               ["Baseline naïve", "36,49", "44,58", "0,702"],
               ["Régression linéaire", "35,89", "43,64", "0,715"],
               ["Random Forest", "36,05", "44,21", "0,707"]],
              [7.4, 3.0, 3.0, 3.0]),
      Spacer(1, 6),
      p("<b>Premier constat : les modèles battent à peine la baseline.</b> La Random Forest attribue <b>94,9 %</b> "
        "de l'importance au ratio de l'enquête précédente. La population, le profil de traitement et le tonnage "
        "précédent se partagent le reste."),
      p("Plus de données aiderait-il ?", "h2"),
      p("Nous avons réentraîné la régression sur des portions croissantes de l'entraînement (de 50 à 501 lignes), "
        "en mesurant à chaque fois le score sur le même test."),
      figure(fig_courbe_apprentissage(), 13.5,
             "Figure 2 — Courbe d'apprentissage de la régression linéaire : le score ne progresse pas avec le nombre de lignes."),
      p("La courbe est <b>plate dès 50 lignes</b>. Si le manque de données était le problème, le score monterait "
        "avec le nombre d'exemples. Ici, la limite vient du <b>contenu des variables</b>, pas de leur quantité."),
      KeepTogether([p("Une variable supplémentaire : le revenu", "h2"),
      p("Hypothèse : le pouvoir d'achat influence la quantité de déchets. Le revenu médian INSEE a été ajouté, "
        "avec son historique (script " + c("08_income_experiment.py") + ", période 2015–2021 commune aux deux sources) :"),
      tableau([["Période 2015–2021", "MAE", "R²"],
               ["Baseline naïve", "21,72", "0,871"],
               ["Régression sans revenu", "19,86", "0,887"],
               ["Régression avec revenu", "20,17", "0,885"]],
              [8.4, 4.0, 4.0])]),
      Spacer(1, 6),
      p("Le revenu <b>n'améliore rien</b> (il pèse 2,7 % dans la Random Forest). Hypothèse testée et écartée. "
        "Ces scores ne sont pas comparables aux précédents : la période est différente."),
      PageBreak()]

# 10. Preprocessor et pipeline
H += [etape(11, "24 septembre 2026"), Chapitre("11. Preprocessor et pipeline"),
      p("Les scripts 05 et 06 faisaient chaque transformation à la main. Le script " + c("09_pipeline.py") +
        " regroupe la même chaîne dans les deux objets de scikit-learn prévus pour ça."),
      p("Le preprocessor : l'étape de transformation", "h2"),
      p("Un " + c("ColumnTransformer") + " applique à chaque type de colonne la transformation adaptée :"),
      *puces(["variables numériques → " + c("MinMaxScaler") + " (entre 0 et 1) ;",
              "profil de traitement → " + c("OneHotEncoder") + " : le cluster (0 à 3) devient 4 colonnes 0/1."]),
      encadre("Un défaut corrigé au passage",
              "Dans la première version, le cluster était mis à l'échelle comme un nombre. Or « cluster 3 » n'est pas "
              "« trois fois cluster 1 » : c'est une catégorie. Le traiter comme un nombre lui donnait un faux ordre. "
              "L'encodage one-hot corrige ce point. L'effet sur le score est quasi nul (R² de 0,715 à 0,716), ce qui "
              "est cohérent avec le poids très faible du cluster.", attention=True),
      p("La pipeline : la chaîne complète", "h2"),
      code('preprocessor = ColumnTransformer([\n'
           '    ("num", MinMaxScaler(),  ["VA_POPANNEE", "RATIO_DMA_lag1", "TONNAGE_DMA_lag1"]),\n'
           '    ("cat", OneHotEncoder(handle_unknown="ignore"), ["cluster"]),\n'
           '])\n'
           'pipeline = Pipeline([("preprocessor", preprocessor), ("modele", LinearRegression())])\n\n'
           'pipeline.fit(X_train, y_train)   # calibre le preprocessor PUIS entraîne, sur le train seul\n'
           'pipeline.predict(X_test)         # mêmes transformations PUIS prédiction'),
      p("Ce que la pipeline apporte :"),
      *puces(["<b>Aucune fuite de données, par construction</b> : le preprocessor est calibré à l'intérieur de "
              + c("fit") + ", donc sur l'entraînement seul, y compris dans chaque pli de la validation croisée. "
              "Ce n'est plus une règle à respecter à la main.",
              "<b>Le même traitement partout</b> : test et nouvelles données passent exactement par les mêmes transformations.",
              "<b>Une comparaison équitable</b> : tous les modèles reçoivent le même preprocessor, seul le modèle change.",
              "<b>Un seul objet</b> prend les données brutes et renvoie une prédiction."]),
      PageBreak()]

# 11. Évaluation
H += [etape(12, "24 septembre 2026"), Chapitre("12. Évaluation et interprétation"),
      p("Deux évaluations complémentaires :"),
      *puces(["une <b>validation croisée temporelle</b> sur l'entraînement : 4 plis où l'on entraîne sur les "
              "enquêtes passées et l'on valide sur la suivante (2013, 2015, 2017, 2019) ;",
              "le <b>test final</b> sur 2021 et 2023, jamais vus."]),
      tableau([["Modèle", "Validation croisée : MAE", "R²", "Test : MAE", "RMSE", "R²"],
               ["Baseline moyenne", "62,23", "−0,013", "61,99", "82,14", "−0,010"],
               ["<b>Baseline naïve</b>", "<b>18,68</b>", "<b>0,872</b>", "36,49", "44,58", "0,702"],
               ["Régression linéaire", "19,55", "0,865", "<b>35,75</b>", "<b>43,53</b>", "<b>0,716</b>"],
               ["Random Forest", "20,88", "0,845", "36,08", "44,36", "0,705"]],
              [4.4, 3.4, 1.8, 2.3, 2.2, 2.3]),
      Spacer(1, 6),
      figure(fig_modeles(), 14.0, "Figure 3 — R² des quatre approches, en validation croisée et sur le test."),
      KeepTogether([p("D'où viennent ces chiffres : le calcul sur le test", "h2"),
      p("Les 198 erreurs du test (y − ŷ) sont additionnées en valeur absolue pour la MAE, au carré pour la RMSE et le "
        "R². SS<sub>tot</sub> est la même pour tous les modèles : c'est la variabilité des vraies valeurs autour de leur "
        "moyenne (" + fr(CALC.moyenne_y[0]) + " kg/hab)."),
      tableau([["Modèle", "Σ |y − ŷ|", "MAE = Σ/198", "Σ (y − ŷ)²", "RMSE = √(Σ/198)", "R² = 1 − Σ/SS<sub>tot</sub>"]] +
              [[r.modele, fr(r.somme_erreurs_abs), fr(r.MAE, 2), fr(r.somme_erreurs_carrees, 0), fr(r.RMSE, 2),
                "1 − " + fr(r.somme_erreurs_carrees, 0) + " / " + fr(r.SS_tot, 0) + " = <b>" + fr(r.R2, 3) + "</b>"]
               for r in CALC.itertuples()],
              [3.3, 2.1, 2.1, 2.3, 2.5, 4.1])]),
      Spacer(1, 6),
      p("Le gain sur la baseline naïve est-il réel ?", "h2"),
      p("Sur le test, la régression linéaire se trompe de 0,74 kg/hab de moins que la baseline naïve. Un bootstrap "
        "(5 000 rééchantillonnages des 198 départements-années) donne un intervalle de confiance à 95 % de "
        "<b>0,14 à 1,34 kg/hab</b> : réel au sens statistique, mais <b>négligeable</b> (2 % de l'erreur, sur un ratio "
        "moyen d'environ 520 kg/hab). La régression ne fait mieux que la baseline que sur 54 % des départements-années. "
        "Elle a d'ailleurs réappris la persistance : sur la seule variable d'historique, elle donne "
        "<b>ratio ≈ 0,97 × ratio précédent + 17</b>."),
      encadre("Modèle retenu",
              "Aucun modèle ne bat la baseline naïve de façon utile : elle est la meilleure en validation croisée, et "
              "la régression linéaire ne la devance que de très peu sur le test. S'il faut retenir un modèle "
              "entraîné, c'est la <b>régression linéaire</b> : meilleure que la Random Forest sur les deux "
              "évaluations, plus simple et interprétable. Mais la conclusion honnête est que la règle « comme à "
              "l'enquête précédente » fait aussi bien."),
      p("Pourquoi le score baisse-t-il sur le test ?", "h2"),
      p("Le R² passe de 0,865 en validation croisée à 0,716 sur le test. Ce n'est <b>pas du surapprentissage</b> : "
        "la baseline naïve, qui n'apprend rien, chute de la même façon."),
      figure(fig_baseline_par_annee(), 14.0,
             "Figure 4 — Qualité de la règle « comme à l'enquête précédente », année par année."),
      p("Les années de test marquent une <b>rupture</b> : le ratio moyen monte à 563 kg/hab en 2021 (contre 525 à "
        "540 auparavant), puis retombe à 519 en 2023. « Comme la dernière fois » devient une moins bonne hypothèse, "
        "et le modèle, qui repose surtout dessus, baisse avec elle. L'origine de cette rupture n'est pas "
        "identifiable avec nos données. La période Covid et l'extension des consignes de tri sont des hypothèses, "
        "non vérifiées."),
      figure(fig_nuage(), 8.2,
             "Figure 5 — Ratio prédit par la pipeline (régression linéaire) contre ratio réel, jeu de test. "
             "En 2023, la plupart des points passent sous la diagonale : le modèle prévoit plus de déchets qu'il n'y en a eu."),
      PageBreak()]

# 13. Vérification terrain
H += [etape(13, "24 septembre 2026"), Chapitre("13. Tester le modèle sur le terrain"),
      p("Une vérification « terrain » a déjà eu lieu", "h2"),
      p("Nous ne pouvions pas aller mesurer nous-mêmes les déchets des départements. Mais le découpage par année "
        "joue exactement ce rôle : le modèle a été entraîné jusqu'en 2019, puis confronté aux <b>vraies</b> valeurs de "
        "2021 et 2023, publiées par l'ADEME et qu'il n'avait jamais vues. C'est un test en conditions réelles, a "
        "posteriori (on parle de <i>backtest</i>). Les scores du chapitre 12 sont donc mesurés sur le terrain."),
      encadre("Et des données simulées (mocks) ?",
              "Générer de fausses données reste utile pour <b>tester le code</b> : vérifier que la pipeline accepte un "
              "département inconnu, une valeur manquante ou un nouveau format sans planter. En revanche, elles ne "
              "peuvent pas servir à <b>évaluer le modèle</b> : c'est nous qui fixerions les règles qui les produisent, "
              "et le modèle serait jugé sur sa capacité à retrouver nos propres hypothèses. Le score serait circulaire. "
              "Nous n'avons donc utilisé que des données réelles pour mesurer la performance.", attention=True),
      p("Pourquoi nous ne donnons pas d'estimation pour 2025", "h2"),
      p("La prochaine enquête (2025) n'est pas encore publiée. Il serait tentant de la « prédire », mais nous avons "
        "choisi de ne pas le faire :"),
      *puces(["<b>Une entrée manque</b> : le modèle utilise la population de l'année prédite, et la cible elle-même est "
              "un tonnage divisé par cette population. Sans la population 2025, il faudrait la remplacer par une "
              "approximation dont on ne peut pas mesurer l'effet.",
              "<b>Le résultat serait invérifiable</b> : il n'apporterait rien à l'évaluation du modèle, qui repose "
              "entièrement sur des années dont on connaît la vraie valeur.",
              "<b>Le risque principal n'est pas couvert</b> : l'erreur vient surtout d'un mouvement national d'une "
              "enquête à l'autre (chapitre 14), que le modèle ne sait pas anticiper. Une estimation 2025 donnerait "
              "une fausse impression de précision."]),
      p("Une estimation pour 2025 ne serait réaliste qu'une fois la population 2025 connue et, surtout, avec une "
        "information sur l'évolution nationale."),
      PageBreak()]

# 14. Améliorer le modèle
_err = pd.read_csv("data/processed/erreurs_test.csv", dtype={"C_DEPT": str})
_err["evolution"] = _err.RATIO_DMA - _err.RATIO_DMA_lag1
_par_an = _err.groupby("ANNEE").agg(err=("erreur", "mean"), sd=("erreur", "std"), evo=("evolution", "mean"),
                                    baisse=("evolution", lambda e: (e < 0).mean()))
_nat = _err.groupby("ANNEE")["evolution"].transform("mean")
_oracle = [("Baseline naïve", _err.RATIO_DMA_lag1), ("Régression linéaire (modèle retenu)", _err.prediction),
           ("Baseline naïve + évolution nationale connue", _err.RATIO_DMA_lag1 + _nat)]
_top = _err.assign(a=_err.erreur.abs()).nlargest(6, "a")
H += [etape(14, "24 septembre 2026"), Chapitre("14. Comment améliorer le modèle"),
      p("Pour savoir quelles informations ajouter, nous avons d'abord regardé <b>où</b> le modèle se trompe "
        "(script " + c("11_analyse_erreurs.py") + ")."),
      p("L'erreur est surtout commune à tous les départements", "h2"),
      tableau([["Année de test", "Erreur moyenne du modèle", "Écart-type", "Évolution réelle moyenne", "Départements en baisse"]] +
              [[str(a), fr(r.err, 1) + " kg/hab", fr(r.sd, 1), fr(r.evo, 1) + " kg/hab", f"{r.baisse:.0%}"]
               for a, r in _par_an.iterrows()],
              [2.8, 3.8, 2.3, 4.0, 3.5]),
      Spacer(1, 6),
      p("En 2021, presque tous les départements ont augmenté ; en 2023, " + f"{_par_an.baisse.iloc[1]:.0%}" + " ont "
        "baissé. Le modèle, qui prolonge la valeur précédente, sous-estime d'environ 22 kg/hab en 2021 et surestime "
        "d'environ 43 kg/hab en 2023, pour <b>tous</b> les départements à la fois. Une grande partie de l'erreur ne "
        "vient donc pas d'un département en particulier, mais d'un mouvement national."),
      p("Pour mesurer ce que vaudrait cette information, une expérience de pensée : ajouter à la valeur précédente "
        "de chaque département l'évolution nationale moyenne de l'année (connue ici après coup)."),
      tableau([["Approche (test 2021–2023)", "MAE", "R²"]] +
              [[n, fr(mean_absolute_error(_err.RATIO_DMA, pr), 2), fr(r2_score(_err.RATIO_DMA, pr), 3)] for n, pr in _oracle],
              [10.4, 3.0, 3.0], surligne=(3,)),
      Spacer(1, 6),
      encadre("Ce que ça montre",
              "Connaître la tendance nationale ferait passer l'erreur moyenne de 36,5 à 19,9 kg/hab, et le R² de 0,70 "
              "à 0,88. Aucune variable locale testée (population, profil de traitement, revenu) n'en approche. "
              "La piste prioritaire est donc une information qui <b>change dans le temps</b>, d'abord au niveau "
              "national, puis au niveau du département."),
      KeepTogether([p("Les plus grosses erreurs", "h2"),
      tableau([["Département", "Année", "Précédent", "Réel", "Prédit", "Erreur"]] +
              [[r.N_DEPT, str(r.ANNEE), fr(r.RATIO_DMA_lag1, 0), fr(r.RATIO_DMA, 0), fr(r.prediction, 0), fr(r.erreur, 0)]
               for r in _top.itertuples()],
              [5.0, 1.8, 2.3, 2.3, 2.3, 2.7])]),
      Spacer(1, 6),
      p("Deux motifs se dégagent. Certains départements font un <b>aller-retour</b> brutal : l'Eure-et-Loir passe de "
        "513 à 609 puis 454 kg/hab, le Territoire-de-Belfort de 502 à 603 puis 479. Un tel yo-yo évoque plutôt un "
        "changement de périmètre ou de déclaration qu'un changement de comportement : c'est à vérifier auprès de la "
        "source. Par ailleurs, la <b>Corse</b> a l'erreur moyenne la plus forte (+52 kg/hab). Or la population "
        "utilisée pour le ratio est la population résidente : les touristes produisent des déchets sans être comptés, "
        "ce qui pourrait expliquer les ratios élevés des départements touristiques (Corse, Landes, Var)."),
      PageBreak(),
      p("Les pistes, par ordre de priorité", "h2"),
      p("Oui, le modèle pourrait être consolidé avec davantage d'informations. Voici celles qui répondent aux "
        "erreurs observées. Les sources sont des pistes à confirmer au moment de la collecte."),
      tableau([["Priorité", "Information à collecter", "Pourquoi (hypothèse)", "Source envisagée"],
               ["1 · national, dans le temps",
                "Consommation des ménages ; événements réglementaires (extension des consignes de tri à tous les "
                "emballages, généralisée début 2023 ; tri à la source des biodéchets, obligatoire depuis 2024) ; "
                "crise sanitaire 2020–2021",
                "L'essentiel de l'erreur est un mouvement commun à tous les départements",
                "INSEE (comptes nationaux) ; textes réglementaires, codés en variables 0/1 par année"],
               ["2 · département, dans le temps",
                "Tarification incitative (part de la population couverte) ; nombre de déchèteries",
                "Des politiques qui changent la production d'une enquête à l'autre, et à des dates différentes selon "
                "les territoires",
                "ADEME / SINOE (annuaire des déchèteries repéré au début du projet)"],
               ["3 · mode de vie",
                "Tourisme (capacité d'hébergement, résidences secondaires) ; type d'habitat (maisons avec jardin) ; "
                "structure par âge, taille des ménages",
                "Population réelle différente de la population résidente ; les jardins produisent des déchets verts",
                "INSEE (recensement, tourisme) ; le fichier par âge déjà téléchargé"],
               ["4 · plus fin géographiquement",
                "Descendre au niveau des intercommunalités (EPCI), qui organisent la collecte",
                "Des situations plus variées (une EPCI adopte la tarification incitative, sa voisine non)",
                "ADEME, si les indicateurs existent à ce niveau. Le niveau commune n'est pas réaliste : le "
                "tonnage n'y est pas mesuré"],
               ["Qualité", "Vérifier les allers-retours suspects (Eure-et-Loir, Territoire-de-Belfort)",
                "Une erreur de déclaration n'est pas un comportement à apprendre", "ADEME (métadonnées SINOE)"]],
              [2.6, 5.0, 4.6, 4.2]),
      Spacer(1, 8),
      encadre("Plus de lignes, ou plus d'information ?",
              "Descendre à l'échelle des villes ne suffirait pas en soi : la courbe d'apprentissage (chapitre 10) montre "
              "qu'avec les mêmes variables, multiplier les lignes n'améliore rien. Un niveau plus fin n'est utile que "
              "s'il apporte des <b>situations différentes</b>, par exemple des politiques de collecte qui varient d'un "
              "territoire à l'autre.", attention=True),
      p("Comment intégrer une nouvelle variable", "h2"),
      p("La démarche est la même pour chaque piste, et la pipeline la rend simple :"),
      *puces(["<b>Collecter et documenter</b> la source : producteur, années, niveau géographique, licence.",
              "<b>Harmoniser</b> sur la clé département × année, avec les mêmes contrôles (unicité, " + c("validate") +
              ", nombre de lignes avant et après).",
              "<b>Ne garder que ce qui est connu avant la date prédite</b> : la valeur de l'enquête précédente, "
              "sinon c'est une fuite.",
              "<b>L'ajouter au preprocessor</b> : une ligne dans la liste des colonnes numériques ou catégorielles. "
              "Le reste de la pipeline ne change pas.",
              "<b>Évaluer avec le même protocole</b> : validation croisée temporelle, test 2021–2023, comparaison à la "
              "baseline naïve.",
              "<b>La garder seulement si le gain est réel et utile</b> : intervalle de confiance du bootstrap au-dessus "
              "de 0, et gain significatif en kg/hab.",
              "Autre piste de modélisation : prédire directement l'<b>évolution</b> (y − valeur précédente) plutôt que "
              "le niveau, pour que le modèle se concentre sur ce qui change."]),
      p("Précaution : avec seulement 8 enquêtes, une variable nationale ne prend que 8 valeurs. Le risque de "
        "sur-interpréter une coïncidence est réel ; il faudra plusieurs enquêtes supplémentaires pour confirmer "
        "un effet."),
      PageBreak()]

# 15. Conclusion
H += [Chapitre("15. Problèmes rencontrés et conclusion"),
      tableau([["Problème", "Ce que nous avons fait"],
               ["Même nom de colonne, périmètre différent (gravats)", "Cible toujours lue dans le fichier hors gravats"],
               ["Une enquête tous les deux ans, pas de 2022", "Assumé, non imputé ; le temps est porté par l'historique du département"],
               ["Le score stagne", "Courbe d'apprentissage : la limite vient des variables, pas du volume"],
               ["Pas de mesure « terrain » possible", "Test sur des années réelles jamais vues ; pas de données simulées pour évaluer"]],
              [7.2, 9.2]),
      Spacer(1, 12), p("Conclusion", "h2"),
      p("Peut-on prédire la production de déchets par habitant d'un département ? <b>Oui, à environ 36 kg près</b>, "
        "soit 7 % d'une valeur moyenne de 520 kg, sur des années que le modèle n'avait jamais vues. Mais cette "
        "réussite n'est pas celle du machine learning : la simple règle « comme à l'enquête précédente » obtient le "
        "même résultat. Nos modèles ne font que la retrouver."),
      p("Ce constat dit quelque chose du sujet lui-même. Les écarts entre départements sont considérables, de "
        "350 kg par habitant dans les Hauts-de-Seine à près de 890 dans les Landes, mais ils sont <b>durables</b> : ils "
        "tiennent à des habitudes, à une organisation de la collecte et à un territoire qui changent lentement. Ce "
        "qui bouge d'une enquête à l'autre est d'abord un mouvement d'ensemble : une hausse générale en 2021, une baisse "
        "presque partout en 2023. Aucune caractéristique stable d'un département (population, politique de traitement, "
        "revenu) ne peut anticiper un tel mouvement."),
      p("Sur le plan de la méthode, le projet montre qu'un score ne veut rien dire sans le bon repère. Face à la "
        "moyenne, un R² de 0,72 paraît excellent ; face à la baseline naïve, il n'apporte rien. De même, tester sur "
        "des années futures réelles plutôt que sur un tirage au hasard a fait apparaître la rupture de 2023, qu'un "
        "découpage aléatoire aurait masquée. Enfin, un résultat négatif, quand il est démontré (courbe d'apprentissage, "
        "test du revenu, bootstrap), reste un résultat : il évite de complexifier un modèle pour rien et indique où "
        "investir."),
      p("C'est la suite logique du travail : collecter des informations qui <b>évoluent dans le temps</b>, d'abord au "
        "niveau national, puis propres à chaque territoire, et les tester avec la même pipeline et le même protocole. "
        "Le clustering des profils de traitement (annexe A) ouvre par ailleurs la partie non supervisée du module."),
      PageBreak()]

# Glossaire
GLOSSAIRE = [
    ("ADEME", "Agence de la transition écologique. Publie les données SINOE sur les déchets."),
    ("Apprentissage supervisé", "Apprendre à prédire une valeur (y) à partir d'exemples dont on connaît déjà la réponse."),
    ("Backtest", "Tester un modèle sur des années passées qu'il n'a pas vues pendant l'entraînement, comme s'il les prédisait."),
    ("Baseline", "Règle simple servant de repère : un modèle n'est utile que s'il fait mieux qu'elle."),
    ("Baseline naïve", "Ici : prédire que le ratio sera le même qu'à l'enquête précédente."),
    ("Bootstrap", "Rééchantillonner les données au hasard, avec remise, des milliers de fois pour mesurer la stabilité d'un résultat."),
    ("Cible (y)", "La valeur à prédire : ici, le ratio de déchets par habitant (" + c("RATIO_DMA") + ")."),
    ("Clustering", "Regrouper des éléments qui se ressemblent, sans réponse connue à l'avance (apprentissage non supervisé)."),
    ("Corrélation", "Mesure, de −1 à 1, de la force du lien linéaire entre deux variables."),
    ("Données de panel", "Mêmes individus (ici, les départements) observés à plusieurs dates."),
    ("DMA", "Déchets ménagers et assimilés : déchets collectés par le service public, ceux des ménages et des petites activités."),
    ("EPCI", "Établissement public de coopération intercommunale : l'intercommunalité, qui organise souvent la collecte."),
    ("Feature (X)", "Variable d'entrée du modèle : ce qu'il connaît au moment de prédire."),
    ("Fuite de données", "Situation où le modèle reçoit, directement ou non, la réponse qu'il doit prédire. Rend les scores trompeurs."),
    ("INSEE", "Institut national de la statistique et des études économiques. Source de la population et du revenu."),
    ("Intervalle de confiance", "Plage de valeurs dans laquelle se trouve un résultat avec une probabilité donnée (ici 95 %)."),
    ("Jeu d'entraînement (train)", "Données sur lesquelles le modèle apprend : enquêtes 2011 à 2019."),
    ("Jeu de test", "Données gardées de côté pour évaluer le modèle : enquêtes 2021 et 2023."),
    ("k-means", "Algorithme de clustering qui forme k groupes autour de k centres."),
    ("Lag", "Variable d'historique : la valeur de l'enquête précédente pour le même département."),
    ("MAE", "<i>Mean Absolute Error</i> : erreur absolue moyenne, en kg/habitant."),
    ("MinMaxScaler", "Transformation qui ramène une variable entre 0 et 1."),
    ("Mock", "Donnée fictive, générée pour tester un programme."),
    ("OneHotEncoder", "Transformation d'une variable catégorielle en plusieurs colonnes 0/1, une par catégorie."),
    ("Pipeline", "Enchaînement preprocessor → modèle en un seul objet, qui garantit le même traitement partout."),
    ("Pli (fold)", "Une des découpes entraînement / validation de la validation croisée."),
    ("Preprocessor", "L'étape de transformation des données avant le modèle (mise à l'échelle, encodage)."),
    ("R²", "Coefficient de détermination : part des écarts expliquée par le modèle (1 = parfait, 0 = pas mieux que la moyenne)."),
    ("Random Forest", "Modèle combinant de nombreux arbres de décision ; capte les relations non linéaires."),
    ("Régression linéaire", "Modèle qui prédit une valeur comme une somme pondérée des variables d'entrée."),
    ("RMSE", "<i>Root Mean Squared Error</i> : racine de l'erreur quadratique moyenne ; pénalise les grosses erreurs."),
    ("Score de silhouette", "Mesure, de −1 à 1, de la qualité d'un clustering : chaque élément est-il plus proche de son groupe que des autres ?"),
    ("SINOE", "Base de données de l'ADEME sur les déchets, source de nos données principales."),
    ("SS", "<i>Sum of Squares</i> : somme des carrés (SS<sub>res</sub> pour les erreurs, SS<sub>tot</sub> pour la variabilité totale)."),
    ("Surapprentissage", "Quand un modèle apprend les particularités de l'entraînement au point de mal prédire de nouvelles données."),
    ("Validation croisée", "Répéter entraînement et validation sur plusieurs découpes des données pour une évaluation plus fiable."),
]
H += [Chapitre("Glossaire"),
      tableau([["Terme", "Définition"]] + [[f"<b>{t}</b>", d] for t, d in sorted(GLOSSAIRE, key=lambda x: x[0].lower())],
              [4.4, 12.0]),
      PageBreak()]

# Annexe A
H += [Chapitre("Annexe A. Clustering des profils de traitement"),
      figure(fig_clusters(), 15.0, "Figure 6 — Profil de traitement moyen de chaque groupe de départements (k-means, k = 4)."),
      tableau([["Groupe", "Profil dominant", "Départements"],
               ["0 — Stockage", "44 % de stockage, 27 % de valorisation matière", "42, dont Aisne, Allier, Ardèche"],
               ["1 — Incinération", "40 % d'incinération avec récupération d'énergie, environ 6 fois plus que le groupe 0",
                "52, dont Ain, Calvados, Finistère"],
               ["2 — Atypique", "33 % « non précisé » et 46 % de stockage : probablement un défaut de déclaration",
                "2 : Lozère, Mayotte"],
               ["3 — Valorisation", "54 % de valorisation matière et organique, le plus haut des 4 groupes",
                "5 : Charente, Charente-Maritime, Landes, Morbihan, Savoie"]],
              [3.4, 8.0, 5.0]),
      Spacer(1, 8),
      KeepTogether([p("Choix du nombre de groupes", "h2"),
      p("Le score de silhouette mesure si chaque département est plus proche de son groupe que du groupe voisin "
        "(de −1 à 1). Le meilleur score est obtenu pour k = 4 :"),
      tableau([["k", "2", "3", "4", "5", "6", "7"],
               ["Silhouette", "0,244", "0,228", "<b>0,276</b>", "0,249", "0,241", "0,248"]],
              [2.6, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3])]),
      PageBreak()]

# Annexe B : détail des calculs
_cv = CV.copy()
_ex3 = test.assign(pred=pred)
_ex3 = _ex3[(_ex3.N_DEPT.isin(["Ain", "Landes", "Paris"])) & (_ex3.ANNEE == 2023)]
_pop_min, _pop_max = train.VA_POPANNEE.min(), train.VA_POPANNEE.max()
_ain = train[(train.N_DEPT == "Ain") & (train.ANNEE == 2019)].iloc[0]
H += [Chapitre("Annexe B. Détail des calculs"),
      p("Tous ces chiffres sont produits par le script " + c("10_detail_metriques.py") + " (" + c("make details") + ")."),
      p("B.1 Trois erreurs, une par une (test 2023, régression linéaire)", "h2"),
      tableau([["Département", "Ratio 2021 (entrée)", "Réel 2023 (y)", "Prédit (ŷ)", "Erreur y − ŷ", "|y − ŷ|", "(y − ŷ)²"]] +
              [[r.N_DEPT, fr(r.RATIO_DMA_lag1), fr(r.RATIO_DMA), fr(r.pred), fr(r.RATIO_DMA - r.pred),
                fr(abs(r.RATIO_DMA - r.pred)), fr((r.RATIO_DMA - r.pred) ** 2, 0)] for r in _ex3.itertuples()],
              [2.6, 2.6, 2.3, 2.1, 2.4, 2.0, 2.4]),
      Spacer(1, 4),
      p("La MAE est la moyenne de la colonne |y − ŷ| sur les 198 lignes du test ; la RMSE est la racine de la moyenne "
        "de la colonne (y − ŷ)². Paris est prédit presque parfaitement (sa valeur bouge peu), les Landes "
        "beaucoup moins bien (forte baisse en 2023)."),
      p("B.2 Validation croisée, pli par pli", "h2"),
      tableau([["Validé sur", "Entraîné sur", "Modèle", "MAE", "RMSE", "R²"]] +
              [[str(r.annee_validee), f"{r.n_train} lignes", r.modele, fr(r.MAE, 2), fr(r.RMSE, 2), fr(r.R2, 3)]
               for r in _cv.itertuples()] +
              [["<b>Moyenne</b>", "", "<b>" + m + "</b>", fr(g.MAE.mean(), 2), fr(g.RMSE.mean(), 2), "<b>" + fr(g.R2.mean(), 3) + "</b>"]
               for m, g in _cv.groupby("modele", sort=False)],
              [2.2, 2.6, 4.2, 2.4, 2.4, 2.6], surligne=tuple(range(17, 21))),
      Spacer(1, 6),
      KeepTogether([p("B.3 Test, année par année", "h2"),
      tableau([["Modèle", "MAE 2021", "R² 2021", "MAE 2023", "R² 2023"]] +
              [[m, fr(g[g.annee == 2021].MAE.iloc[0], 2), fr(g[g.annee == 2021].R2.iloc[0], 3),
                fr(g[g.annee == 2023].MAE.iloc[0], 2), fr(g[g.annee == 2023].R2.iloc[0], 3)]
               for m, g in PAR_ANNEE.groupby("modele", sort=False)],
              [5.6, 2.7, 2.7, 2.7, 2.7])]),
      Spacer(1, 6),
      KeepTogether([p("B.4 Coefficients de la régression linéaire", "h2"),
      p("Les variables numériques étant ramenées entre 0 et 1, un coefficient se lit comme « l'effet, en kg/hab, de "
        "passer du minimum au maximum observé ». Ordonnée à l'origine : 266,9."),
      tableau([["Variable (après preprocessor)", "Coefficient"]] +
              [[c(r.variable), fr(r.coefficient, 2)] for r in COEFS.itertuples()],
              [10.4, 6.0])]),
      Spacer(1, 4),
      p("Le ratio précédent domine (+602 entre le département le plus bas et le plus haut). Réentraînée sur cette "
        "seule variable, en kg/hab, la régression donne <b>ratio = 0,969 × ratio précédent + 16,7</b> : presque "
        "la valeur précédente telle quelle."),
      KeepTogether([p("B.5 Mise à l'échelle Min-Max : un exemple", "h2"),
      p("Sur l'entraînement, la population va de " + fr(_pop_min, 0) + " à " + fr(_pop_max, 0) + " habitants. Pour "
        "l'Ain en 2019 (" + fr(_ain.VA_POPANNEE, 0) + " habitants) :"),
      code(f"x' = (x − min) / (max − min) = ({_ain.VA_POPANNEE:.0f} − {_pop_min:.0f}) / ({_pop_max:.0f} − {_pop_min:.0f})"
           f" = {(_ain.VA_POPANNEE - _pop_min) / (_pop_max - _pop_min):.3f}")]),
      KeepTogether([p("B.6 Bootstrap du gain sur la baseline naïve", "h2"),
      p("On tire au hasard, avec remise, 198 départements-années parmi les 198 du test, et l'on calcule la différence "
        "de MAE entre la baseline naïve et la régression linéaire. Répété 5 000 fois, cela donne la distribution du "
        "gain. 95 % des tirages donnent un gain entre <b>0,14 et 1,34 kg/hab</b> : l'intervalle ne contient pas 0, "
        "le gain est donc réel, mais il reste très faible face à une erreur moyenne de 36 kg/hab.")]),
      PageBreak()]

# Annexe C : guide d'utilisation
H += [Chapitre("Annexe C. Guide d'utilisation du projet"),
      p("Le dépôt contient un " + c("Makefile") + " qui permet de rejouer chaque étape par une commande simple, "
        "sans connaître le nom des scripts. Chaque commande lance aussi les étapes dont elle dépend."),
      p("Prérequis", "h2"),
      *puces(["Python 3.12 (ou une version récente de Python 3) ;", c("make") + " et " + c("git") + " ;",
              "aucun téléchargement de données : les fichiers sources sont inclus dans le dépôt."]),
      p("Installation", "h2"),
      code("git clone https://github.com/Rxdy/ML-DMA.git\ncd ML-DMA\nmake installer     # crée .venv et installe les dépendances\nmake               # affiche la liste des commandes"),
      p("Les commandes", "h2"),
      tableau([["Commande", "Script", "Ce qu'elle fait"],
               [c("make explorer"), "01", "Exploration : dimensions, valeurs manquantes, doublons"],
               [c("make clustering"), "02", "Profil de traitement et k-means"],
               [c("make preparation"), "03", "Jointure avec le cluster, variables d'historique"],
               [c("make correlation"), "04", "Matrice de corrélation (image dans data/figures/)"],
               [c("make decoupage"), "05", "Découpage train/test et mise à l'échelle (version manuelle)"],
               [c("make modeles"), "06", "Baseline et premiers modèles (version manuelle)"],
               [c("make revenu"), "08", "Test du revenu médian"],
               [c("make pipeline"), "09", "Preprocessor, pipeline, validation croisée, test"],
               [c("make details"), "10", "Détail des calculs de MAE, RMSE et R²"],
               [c("make erreurs"), "11", "Analyse des erreurs"],
               [c("make supervise"), "09 à 11", "Tout le volet supervisé"],
               [c("make all"), "01 à 11", "Rejoue tout le projet"],
               [c("make rapport-pdf"), "—", "Régénère ce rapport"]],
              [4.2, 2.0, 10.2]),
      Spacer(1, 6),
      p("Parcours conseillé pour tester le projet : " + c("make installer") + ", puis " + c("make supervise") + ". La "
        "console affiche les scores de chaque modèle, le détail des calculs et les plus grosses erreurs. Les résultats sont aussi enregistrés en CSV dans " + c("dechets/data/processed/") + "."),
      encadre("Bon à savoir",
              "Le script " + c("07_income_prep.py") + " n'a pas de commande : il assemble des fichiers INSEE téléchargés "
              "à la main, non conservés. Son résultat, " + c("insee_revenu_median_dep_2013_2021.csv") + ", est inclus "
              "dans le dépôt, ce qui suffit à " + c("make revenu") + ". Le rapport PDF nécessite en plus la bibliothèque "
              + c("reportlab") + " (incluse dans requirements.txt) et les polices Noto."),
      p("Organisation du dépôt", "h2"),
      code("ML-DMA/\n"
           "├── Makefile              commandes du projet\n"
           "├── requirements.txt      dépendances Python\n"
           "├── JOURNAL.md            journal de bord, au fil de l'eau\n"
           "└── dechets/\n"
           "    ├── data/raw/         données sources, jamais modifiées\n"
           "    ├── data/processed/   fichiers produits par les scripts\n"
           "    ├── data/figures/     graphiques\n"
           "    ├── scripts/          01 à 11\n"
           "    ├── rapport/          ce rapport (PDF, générateur, logo), rapport HTML\n"
           "    └── COMPTE_RENDU.md   synthèse du volet supervisé")]

doc = Rapport(SORTIE)
doc.multiBuild(H)
print(f"PDF généré -> dechets/{SORTIE}")
