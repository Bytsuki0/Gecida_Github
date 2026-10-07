"""Fase 3.6 - Normalizacao / escalonamento.

Bloco do diagrama: "StandardScaler · MinMaxScaler (por modelo)". A escala
certa depende do modelo, entao este bloco NAO grava um unico dataset
escalonado. Ele entrega:

    - `ESCALONADOR_POR_MODELO`: qual escalonador cada familia da Fase 4 usa;
    - `escalonador_para(modelo, colunas)`: o transformador nao ajustado, para
      entrar no `Pipeline` do modelo e ser reajustado em cada fold do K-Fold;
    - os dois escalonadores ajustados no train, com o efeito medido em train e
      test (e, opcionalmente, as versoes escalonadas dos dados).

So as colunas com mais de dois valores distintos sao escalonadas: dummies,
flags e indicadoras ja estao em {0, 1} e ficam como estao.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import EDAConfig
from .estilo import EIXO, SERIE_1, SERIE_2, TINTA_SECUNDARIA, TINTA_SUAVE
from .utils import amostrar, df_para_markdown, salvar_figura, salvar_tabela

ESCALONADORES = {"standard": StandardScaler, "minmax": MinMaxScaler}

# Familia de modelo -> (escalonador | None, motivo). As chaves seguem os ramos
# da Fase 4 do diagrama; "outros" cobre os regressores ainda a descobrir.
ESCALONADOR_POR_MODELO: dict[str, tuple[str | None, str]] = {
    "linear": ("standard",
               "OLS nao muda as previsoes com a escala, mas os coeficientes ficam comparaveis"),
    "ridge_lasso": ("standard",
                    "a penalidade age sobre os coeficientes: sem escala comum, pune mais "
                    "as colunas de escala pequena"),
    "svr": ("standard", "kernel e margem dependem de distancias entre linhas"),
    "knn": ("minmax", "distancia euclidiana; [0, 1] limita o peso maximo de cada coluna"),
    "mlp": ("minmax", "entradas limitadas estabilizam o gradiente"),
    "random_forest": (None, "arvores dividem por limiares - invariantes a escala"),
    "xgboost": (None, "arvores de gradiente - invariantes a escala"),
    "lightgbm": (None, "arvores de gradiente - invariantes a escala"),
}


@dataclass
class ResultadoEscalonamento:
    """Artefatos do escalonamento (3.6)."""

    colunas: list[str]
    escalonadores: dict[str, object]
    estatisticas: pd.DataFrame
    por_modelo: pd.DataFrame
    escalonados: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] = field(default_factory=dict)
    figuras: dict[str, str] = field(default_factory=dict)


def colunas_para_escalar(df: pd.DataFrame) -> list[str]:
    """Numericas com mais de dois valores distintos (binarias ficam de fora)."""
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 2]


def escalonador_para(modelo: str, colunas: list[str]) -> ColumnTransformer | str:
    """Transformador NAO ajustado para a familia `modelo`, pronto para um Pipeline.

    Devolve "passthrough" quando o modelo nao precisa de escala (arvores).

        Pipeline([("escala", escalonador_para("linear", cols)), ("modelo", LinearRegression())])
    """
    if modelo not in ESCALONADOR_POR_MODELO:
        raise KeyError(f"modelo desconhecido: {modelo!r} (opcoes: {', '.join(ESCALONADOR_POR_MODELO)})")
    nome = ESCALONADOR_POR_MODELO[modelo][0]
    if nome is None:
        return "passthrough"
    return ColumnTransformer(
        [("escala", ESCALONADORES[nome](), colunas)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _aplicar(escalonador: object, df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    saida = df.copy()
    saida[colunas] = escalonador.transform(df[colunas])
    return saida


def _figura(
    original: pd.DataFrame, versoes: dict[str, pd.DataFrame], colunas: list[str], cfg: EDAConfig
) -> dict[str, str]:
    paineis = {"original": original, **versoes}
    fig, axes = plt.subplots(1, len(paineis), figsize=(5.2 * len(paineis), 0.42 * len(colunas) + 1.8),
                             sharey=True)
    titulos = {"original": "Escala original", "standard": "StandardScaler (media 0, desvio 1)",
               "minmax": "MinMaxScaler ([0, 1] no train)"}
    for ax, (nome, df) in zip(np.atleast_1d(axes), paineis.items()):
        caixa = ax.boxplot([df[c].to_numpy() for c in colunas], vert=False, widths=0.6,
                           patch_artist=True, showfliers=False)
        for patch in caixa["boxes"]:
            patch.set(facecolor=SERIE_1 if nome == "original" else SERIE_2, edgecolor=EIXO, alpha=0.85)
        for elemento in ("whiskers", "caps"):
            for artista in caixa[elemento]:
                artista.set(color=TINTA_SUAVE, linewidth=1.0)
        for mediana in caixa["medians"]:
            mediana.set(color=TINTA_SECUNDARIA, linewidth=1.6)
        ax.set_yticks(range(1, len(colunas) + 1), colunas)
        ax.set_title(titulos.get(nome, nome))
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fase 3.6 - Colunas escalonadas (train, sem fliers)",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return {"escalonamento": salvar_figura(fig, "f3_6_escalonamento", cfg)}


def executar(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cfg: EDAConfig,
    manter_escalonados: bool = False,
) -> ResultadoEscalonamento:
    """Ajusta StandardScaler e MinMaxScaler no train e mede o efeito em train/test."""
    colunas = colunas_para_escalar(train)
    escalonadores = {nome: classe().fit(train[colunas]) for nome, classe in ESCALONADORES.items()}

    versoes_train = {nome: _aplicar(esc, train, colunas) for nome, esc in escalonadores.items()}
    versoes_test = ({nome: _aplicar(esc, test, colunas) for nome, esc in escalonadores.items()}
                    if test is not None else {})

    linhas = []
    for c in colunas:
        linha: dict[str, object] = {
            "coluna": c,
            "media_train": train[c].mean(),
            "desvio_train": train[c].std(ddof=0),
            "min_train": train[c].min(),
            "max_train": train[c].max(),
        }
        if test is not None:
            std_test, mm_test = versoes_test["standard"][c], versoes_test["minmax"][c]
            linha |= {
                "standard_media_test": std_test.mean(),
                "standard_desvio_test": std_test.std(ddof=0),
                "minmax_min_test": mm_test.min(),
                "minmax_max_test": mm_test.max(),
                "minmax_test_fora_de_0_1": int(((mm_test < 0) | (mm_test > 1)).sum()),
            }
        linhas.append(linha)
    estatisticas = pd.DataFrame(linhas)

    por_modelo = pd.DataFrame([
        {"modelo": modelo, "escalonador": nome or "nenhum", "motivo": motivo}
        for modelo, (nome, motivo) in ESCALONADOR_POR_MODELO.items()
    ])

    salvar_tabela(estatisticas, "f3_6_escalonamento", cfg)
    salvar_tabela(por_modelo, "f3_6_escalonador_por_modelo", cfg)

    amostra = amostrar(train, cfg.amostra_boxplot, cfg.random_state)
    figuras = _figura(
        amostra[colunas],
        {nome: v.loc[amostra.index, colunas] for nome, v in versoes_train.items()},
        colunas,
        cfg,
    )

    escalonados = ({nome: (versoes_train[nome], versoes_test.get(nome)) for nome in escalonadores}
                   if manter_escalonados else {})
    return ResultadoEscalonamento(
        colunas=colunas,
        escalonadores=escalonadores,
        estatisticas=estatisticas,
        por_modelo=por_modelo,
        escalonados=escalonados,
        figuras=figuras,
    )


def markdown_escalonamento(resultado: ResultadoEscalonamento) -> str:
    partes = [
        "**Escalonador por familia de modelo** - na Fase 4 use "
        "`escalonador_para(modelo, colunas)` dentro do `Pipeline` do modelo, para "
        "que a escala seja reajustada em cada fold do K-Fold.\n",
        df_para_markdown(resultado.por_modelo),
        f"\n**Colunas escalonadas ({len(resultado.colunas)}):** "
        + ", ".join(f"`{c}`" for c in resultado.colunas)
        + ". Binarias (dummies, flags, indicadoras) ficam em {0, 1}.\n",
        "Ajuste no train; o test usa os mesmos parametros. Por isso a media/desvio "
        "do test so ficam *perto* de 0/1, e o MinMax do test pode sair de [0, 1] "
        "quando o test tem valores alem dos extremos do train.\n",
        df_para_markdown(resultado.estatisticas),
    ]
    if "escalonamento" in resultado.figuras:
        partes.append(f"\n![escalonamento]({resultado.figuras['escalonamento']})\n")
    return "\n".join(partes)
