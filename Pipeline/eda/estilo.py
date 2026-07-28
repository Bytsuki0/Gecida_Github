"""Paleta e estilo dos graficos.

Paleta categorica validada (CVD-safe) usada em ordem fixa, nunca ciclada.
Graficos de dispersao/pares usam no maximo os 3 primeiros slots, que sao os
que passam a validacao considerando todos os pares simultaneamente.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- superficies e tinta ---------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_SUAVE = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

# --- paleta categorica (ordem fixa) ----------------------------------------
CATEGORICA = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 laranja
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 amarelo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 vermelho
]
SERIE_1, SERIE_2, SERIE_3 = CATEGORICA[0], CATEGORICA[1], CATEGORICA[2]

# --- status ----------------------------------------------------------------
STATUS = {
    "bom": "#0ca30c",
    "atencao": "#fab219",
    "serio": "#ec835a",
    "critico": "#d03b3b",
}

# --- rampas ----------------------------------------------------------------
# Sequencial: um unico matiz, claro -> escuro.
SEQUENCIAL = LinearSegmentedColormap.from_list(
    "azul_seq",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
)
# Divergente: azul <-> vermelho com cinza neutro no meio (para correlacoes).
DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_verm_div",
    ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#ec8c8b", "#d03b3b", "#7d1f1f"],
)


def aplicar_estilo() -> None:
    """Aplica o tema aos rcParams do matplotlib (grade e eixos recessivos)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": TINTA_PRIMARIA,
            "axes.labelsize": 9,
            "axes.labelcolor": TINTA_SECUNDARIA,
            "axes.edgecolor": EIXO,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRADE,
            "grid.linewidth": 0.7,
            "xtick.color": TINTA_SUAVE,
            "ytick.color": TINTA_SUAVE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 4,
        }
    )
