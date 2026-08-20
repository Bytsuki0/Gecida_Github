"""Fase 3.1 - Tratamento de outliers.

Cobre o bloco "Tratamento de outliers" do diagrama de desenvolvimento:

    3.1.0  Correcao de valores fora do dominio (percentuais > 100, duracao <= 0)
    3.1.1  Ajuste dos limites (IQR clipping | winsorization | z-score > 3)
    3.1.2  Aplicacao dos limites e comparacao antes/depois

Regra central do modulo: os limites sao sempre ajustados **apenas no train**
e depois aplicados, sem reajuste, ao test. Calcular quantis/z-score usando o
test vazaria informacao do conjunto de avaliacao para o pre-processamento.

O codigo nao conhece o dataset de podcasts - as regras de dominio e o metodo
por coluna entram por parametro (ver `Pipeline/run_outliers.py` para os
valores especificos deste projeto).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .config import EDAConfig
from .estilo import EIXO, SERIE_1, SERIE_2, TINTA_SECUNDARIA, TINTA_SUAVE
from .utils import (
    classificar_colunas,
    df_para_markdown,
    grade_de_paineis,
    limites_iqr,
    salvar_figura,
    salvar_tabela,
)

MetodoOutlier = Literal["iqr", "winsor", "zscore"]


@dataclass
class ResultadoFase3Outliers:
    """Artefatos produzidos pelo tratamento de outliers."""

    train: pd.DataFrame
    test: pd.DataFrame | None
    limites: pd.DataFrame
    resumo: pd.DataFrame
    correcao_dominio: pd.DataFrame
    figuras: dict[str, str] = field(default_factory=dict)
    markdown: str = ""


# ---------------------------------------------------------------------------
# 3.1.0 Correcao de valores fora do dominio
# ---------------------------------------------------------------------------
# Isto NAO e tratamento de outliers estatistico: sao valores logicamente
# impossiveis (percentual > 100%, duracao <= 0), entao entram corrigidos
# antes do IQR/winsor/z-score, que tratam de extremos plausiveis.
def corrigir_fora_do_dominio(
    df: pd.DataFrame, regras: dict[str, dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recorta ou remove valores fora de um intervalo logicamente valido.

    `regras` mapeia coluna -> {"inf": float | None, "sup": float | None,
    "acao": "clip" | "nan", "inf_exclusivo": bool, "sup_exclusivo": bool}.
    `acao="nan"` e para casos em que recortar para o limite ainda deixaria um
    valor sem sentido de dominio (ex.: duracao 0). `*_exclusivo=True` torna o
    proprio limite invalido (ex.: duracao > 0 exige inf=0, inf_exclusivo=True
    para pegar tambem o valor exato 0, e nao so os negativos).
    """
    saida = df.copy()
    linhas: list[dict[str, object]] = []
    for coluna, regra in regras.items():
        if coluna not in saida.columns:
            continue
        inf = regra.get("inf")
        sup = regra.get("sup")
        inf_val = -np.inf if inf is None else float(inf)
        sup_val = np.inf if sup is None else float(sup)
        acao = regra.get("acao", "clip")

        serie = saida[coluna]
        abaixo = serie <= inf_val if regra.get("inf_exclusivo") else serie < inf_val
        acima = serie >= sup_val if regra.get("sup_exclusivo") else serie > sup_val
        fora = serie.notna() & (abaixo | acima)
        n = int(fora.sum())
        if n:
            if acao == "nan":
                saida.loc[fora, coluna] = np.nan
            else:
                saida.loc[fora, coluna] = serie[fora].clip(
                    lower=inf if inf is not None else None,
                    upper=sup if sup is not None else None,
                )

        limite_txt = f"[{inf if inf is not None else '-inf'}, {sup if sup is not None else 'inf'}]"
        linhas.append(
            {
                "coluna": coluna,
                "regra": limite_txt,
                "acao": acao,
                "n_corrigidos": n,
                "pct_corrigidos": round(100 * n / len(serie), 4) if len(serie) else 0.0,
            }
        )
    return saida, pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 3.1.1 Calculo de limites (ajustados apenas no train)
# ---------------------------------------------------------------------------
def limites_winsor(serie: pd.Series, p_inf: float = 0.01, p_sup: float = 0.99) -> tuple[float, float]:
    """Limites por percentil (winsorization)."""
    limpa = serie.dropna()
    if limpa.empty:
        return float("nan"), float("nan")
    return float(limpa.quantile(p_inf)), float(limpa.quantile(p_sup))


def limites_zscore(serie: pd.Series, limiar: float = 3.0) -> tuple[float, float]:
    """Limites media +/- limiar*desvio (equivalente a cortar por z-score)."""
    limpa = serie.dropna()
    if limpa.empty or limpa.std() == 0:
        return float("nan"), float("nan")
    media, desvio = limpa.mean(), limpa.std()
    return float(media - limiar * desvio), float(media + limiar * desvio)


def ajustar_limites(
    train: pd.DataFrame,
    metodo_por_coluna: dict[str, MetodoOutlier],
    k_iqr: float = 1.5,
    p_winsor: tuple[float, float] = (0.01, 0.99),
    limiar_zscore: float = 3.0,
) -> pd.DataFrame:
    """Calcula limite inferior/superior por coluna, usando SOMENTE o train."""
    linhas = []
    for coluna, metodo in metodo_por_coluna.items():
        if coluna not in train.columns:
            continue
        serie = train[coluna]
        if metodo == "iqr":
            inf, sup = limites_iqr(serie.dropna(), k=k_iqr) if serie.notna().any() else (np.nan, np.nan)
        elif metodo == "winsor":
            inf, sup = limites_winsor(serie, *p_winsor)
        elif metodo == "zscore":
            inf, sup = limites_zscore(serie, limiar_zscore)
        else:
            raise ValueError(f"metodo desconhecido: {metodo!r} (use 'iqr', 'winsor' ou 'zscore')")
        linhas.append({"coluna": coluna, "metodo": metodo, "limite_inf": inf, "limite_sup": sup})
    return pd.DataFrame(linhas)


def aplicar_limites(df: pd.DataFrame, limites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recorta (clip) usando limites ja ajustados e conta linhas afetadas."""
    saida = df.copy()
    linhas = []
    for _, r in limites.iterrows():
        coluna, inf, sup = r["coluna"], r["limite_inf"], r["limite_sup"]
        if coluna not in saida.columns or pd.isna(inf) or pd.isna(sup):
            continue
        serie = saida[coluna]
        n_validos = int(serie.notna().sum())
        afetados = serie.notna() & ((serie < inf) | (serie > sup))
        n = int(afetados.sum())
        saida[coluna] = serie.clip(inf, sup)
        linhas.append(
            {
                "coluna": coluna,
                "n_afetados": n,
                "pct_afetados": round(100 * n / n_validos, 4) if n_validos else 0.0,
            }
        )
    return saida, pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------
def _figura_antes_depois(
    antes: pd.DataFrame, depois: pd.DataFrame, colunas: list[str], cfg: EDAConfig
) -> dict[str, str]:
    if not colunas:
        return {}
    fig, axes = grade_de_paineis(len(colunas), cfg)
    for ax, c in zip(axes, colunas):
        dados = [antes[c].dropna(), depois[c].dropna()]
        caixa = ax.boxplot(
            dados,
            vert=False,
            widths=0.6,
            patch_artist=True,
            tick_labels=["antes", "depois"],
            flierprops={
                "marker": ".",
                "markersize": 2,
                "markerfacecolor": TINTA_SUAVE,
                "markeredgecolor": "none",
                "alpha": 0.25,
            },
        )
        for patch, cor in zip(caixa["boxes"], (SERIE_2, SERIE_1)):
            patch.set(facecolor=cor, edgecolor=EIXO, alpha=0.85)
        for elemento in ("whiskers", "caps"):
            for artista in caixa[elemento]:
                artista.set(color=TINTA_SUAVE, linewidth=1.2)
        for mediana in caixa["medians"]:
            mediana.set(color=TINTA_SECUNDARIA, linewidth=2.0)
        ax.set_title(c)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Fase 3.1 - Outliers antes vs. depois do tratamento",
        fontsize=13,
        fontweight="semibold",
        color=TINTA_SECUNDARIA,
    )
    fig.tight_layout()
    return {"antes_depois_outliers": salvar_figura(fig, "f3_1_antes_depois_outliers", cfg)}


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def executar(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cfg: EDAConfig,
    metodo_por_coluna: dict[str, MetodoOutlier] | None = None,
    regras_dominio: dict[str, dict[str, object]] | None = None,
    k_iqr: float = 1.5,
    p_winsor: tuple[float, float] = (0.01, 0.99),
    limiar_zscore: float = 3.0,
) -> ResultadoFase3Outliers:
    """Roda a correcao de dominio + o tratamento de outliers.

    `metodo_por_coluna` mapeia coluna -> "iqr" | "winsor" | "zscore". Colunas
    numericas nao listadas usam "iqr" por padrao. `regras_dominio` segue o
    formato de `corrigir_fora_do_dominio`.
    """
    cfg.preparar()

    colunas_numericas = classificar_colunas(train, cfg).numericas
    if metodo_por_coluna is None:
        metodo_por_coluna = {c: "iqr" for c in colunas_numericas}
    else:
        metodo_por_coluna = {c: metodo_por_coluna.get(c, "iqr") for c in colunas_numericas}

    # 3.1.0 - dominio (aplicado em train e test, cada um com seus proprios
    # valores impossiveis; a REGRA em si e a mesma, entao nao ha vazamento)
    regras = regras_dominio or {}
    train_dom, correcao_dominio = corrigir_fora_do_dominio(train, regras)
    test_dom = corrigir_fora_do_dominio(test, regras)[0] if test is not None else None

    # 3.1.1 - limites ajustados apenas no train
    limites = ajustar_limites(
        train_dom, metodo_por_coluna, k_iqr=k_iqr, p_winsor=p_winsor, limiar_zscore=limiar_zscore
    )

    # 3.1.2 - aplica os MESMOS limites no train e no test
    train_tratado, resumo_train = aplicar_limites(train_dom, limites)
    resumo_train = resumo_train.rename(
        columns={"n_afetados": "n_afetados_train", "pct_afetados": "pct_afetados_train"}
    )
    resumo = limites.merge(resumo_train, on="coluna", how="left")

    if test_dom is not None:
        test_tratado, resumo_test = aplicar_limites(test_dom, limites)
        resumo_test = resumo_test.rename(
            columns={"n_afetados": "n_afetados_test", "pct_afetados": "pct_afetados_test"}
        )
        resumo = resumo.merge(resumo_test, on="coluna", how="left")
    else:
        test_tratado = None

    figuras = _figura_antes_depois(train_dom, train_tratado, list(metodo_por_coluna), cfg)

    salvar_tabela(resumo, "f3_1_limites_outliers", cfg)
    if not correcao_dominio.empty:
        salvar_tabela(correcao_dominio, "f3_0_correcao_dominio", cfg)

    markdown = _montar_markdown(resumo, correcao_dominio, figuras, cfg)

    return ResultadoFase3Outliers(
        train=train_tratado,
        test=test_tratado,
        limites=limites,
        resumo=resumo,
        correcao_dominio=correcao_dominio,
        figuras=figuras,
        markdown=markdown,
    )


def _montar_markdown(
    resumo: pd.DataFrame, correcao_dominio: pd.DataFrame, figuras: dict[str, str], cfg: EDAConfig
) -> str:
    partes = ["## Fase 3.1 - Tratamento de outliers\n"]

    if not correcao_dominio.empty:
        partes += [
            "### 3.1.0 Correcao de valores fora do dominio\n",
            "Valores logicamente impossiveis (ex.: percentual > 100%, duracao <= 0) "
            "nao sao outliers estatisticos - sao corrigidos antes, separadamente do "
            "IQR/winsorization/z-score.\n",
            df_para_markdown(correcao_dominio),
        ]

    partes += [
        "\n### 3.1.1 Limites ajustados no train e linhas afetadas\n",
        "Limites calculados apenas no train (`ajustar_limites`) e aplicados sem "
        "reajuste ao test (`aplicar_limites`), para nao vazar informacao do "
        "conjunto de teste para o pre-processamento.\n",
        df_para_markdown(resumo),
    ]
    if "antes_depois_outliers" in figuras:
        partes.append(f"\n![antes_depois_outliers]({figuras['antes_depois_outliers']})\n")
    return "\n".join(partes)
