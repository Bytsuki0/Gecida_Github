"""Fase 2 - Analise Exploratoria (EDA).

Cobre os blocos do diagrama, exceto engenharia de features (deliberadamente
fora deste escopo):

    2.1 Descricao das variaveis   - dtypes, describe, cardinalidade, % NaN
    2.2 Distribuicao do alvo      - histograma, boxplot, skewness, kurtosis
    2.3 Variaveis numericas       - distribuicoes, outliers IQR, scatter vs alvo
    2.4 Variaveis categoricas     - frequencias, barplots, boxplot por categoria
    2.5 Matriz de correlacao      - Pearson, Spearman, heatmap
    2.6 Consolidacao de alertas   - insumo para as decisoes da Fase 3

Nenhuma funcao modifica o DataFrame recebido.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import EDAConfig
from .estilo import (
    DIVERGENTE,
    EIXO,
    SERIE_1,
    SERIE_2,
    SERIE_3,
    STATUS,
    TINTA_SECUNDARIA,
    TINTA_SUAVE,
)
from .utils import (
    Colunas,
    amostrar,
    classificar_colunas,
    contar_outliers_iqr,
    df_para_markdown,
    eta_quadrado,
    grade_de_paineis,
    limites_iqr,
    salvar_figura,
    salvar_tabela,
)


@dataclass
class ResultadoFase2:
    """Artefatos produzidos pela Fase 2."""

    colunas: Colunas
    descricao: pd.DataFrame
    alvo: pd.DataFrame
    numericas: pd.DataFrame
    correlacao_alvo: pd.DataFrame
    categoricas: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    pares_redundantes: pd.DataFrame
    alertas: pd.DataFrame
    figuras: dict[str, str] = field(default_factory=dict)
    markdown: str = ""


# ---------------------------------------------------------------------------
# 2.1 Descricao das variaveis
# ---------------------------------------------------------------------------
def descrever_variaveis(df: pd.DataFrame, cfg: EDAConfig, colunas: Colunas) -> pd.DataFrame:
    """Uma linha por coluna: tipo, ausentes, cardinalidade e valor dominante."""
    papeis = {
        **{c: "numerica" for c in colunas.numericas},
        **{c: "categorica" for c in colunas.categoricas},
        **{c: "alta cardinalidade" for c in colunas.alta_cardinalidade},
        **{c: "constante" for c in colunas.constantes},
        **{c: "identificador" for c in colunas.identificadores},
    }

    n = len(df)
    linhas = []
    for nome in df.columns:
        serie = df[nome]
        contagem = serie.value_counts(dropna=True)
        top_valor = contagem.index[0] if not contagem.empty else None
        top_freq = float(contagem.iloc[0]) if not contagem.empty else 0.0
        nunique = int(serie.nunique(dropna=True))

        papel = papeis.get(nome)
        if papel is None:
            papel = "alvo" if nome == cfg.target else "id"

        linhas.append(
            {
                "coluna": nome,
                "papel": papel,
                "dtype": str(serie.dtype),
                "n_preenchidos": int(serie.notna().sum()),
                "n_nan": int(serie.isna().sum()),
                "nan_pct": round(100 * serie.isna().mean(), 3),
                "nunique": nunique,
                "unicidade_pct": round(100 * nunique / n, 3) if n else 0.0,
                "valor_dominante": top_valor,
                "dominante_pct": round(100 * top_freq / n, 3) if n else 0.0,
                "memoria_MB": round(serie.memory_usage(deep=True) / 1024**2, 3),
            }
        )
    return pd.DataFrame(linhas)


def _figura_nan(descricao: pd.DataFrame, cfg: EDAConfig) -> dict[str, str]:
    com_nan = descricao[descricao["nan_pct"] > 0].sort_values("nan_pct")
    if com_nan.empty:
        return {}

    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.42 * len(com_nan) + 1.2)))
    posicoes = np.arange(len(com_nan))
    ax.barh(posicoes, com_nan["nan_pct"], color=SERIE_1, height=0.62)
    ax.set_yticks(posicoes, com_nan["coluna"])
    ax.set_xlabel("% de valores ausentes")
    ax.set_title("Fase 2.1 - Valores ausentes por coluna")
    ax.axvline(cfg.limite_nan_coluna, color=STATUS["critico"], linewidth=1.5, linestyle="--")
    ax.set_ylim(-0.9, len(com_nan) - 0.4)
    ax.text(cfg.limite_nan_coluna, -0.8,
            f" limite de descarte ({cfg.limite_nan_coluna:.0f}%)",
            color=STATUS["critico"], fontsize=8, va="bottom")
    ax.set_xlim(0, max(100, com_nan["nan_pct"].max() * 1.15))
    ax.grid(axis="y", visible=False)
    for y, valor in zip(posicoes, com_nan["nan_pct"]):
        ax.text(valor + 0.8, y, f"{valor:.2f}%", va="center", fontsize=8, color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return {"valores_ausentes": salvar_figura(fig, "f2_1_valores_ausentes", cfg)}


# ---------------------------------------------------------------------------
# 2.2 Distribuicao da variavel alvo
# ---------------------------------------------------------------------------
def analisar_alvo(df: pd.DataFrame, cfg: EDAConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    """Estatisticas de forma + painel de distribuicao da variavel alvo."""
    alvo = df[cfg.target].dropna()

    n_out, pct_out = contar_outliers_iqr(alvo)
    stats_alvo = pd.DataFrame(
        [
            {
                "metrica": "n",
                "valor": len(alvo),
            },
            {"metrica": "media", "valor": float(alvo.mean())},
            {"metrica": "mediana", "valor": float(alvo.median())},
            {"metrica": "desvio_padrao", "valor": float(alvo.std())},
            {"metrica": "minimo", "valor": float(alvo.min())},
            {"metrica": "p01", "valor": float(alvo.quantile(0.01))},
            {"metrica": "p25", "valor": float(alvo.quantile(0.25))},
            {"metrica": "p75", "valor": float(alvo.quantile(0.75))},
            {"metrica": "p99", "valor": float(alvo.quantile(0.99))},
            {"metrica": "maximo", "valor": float(alvo.max())},
            {"metrica": "skewness", "valor": float(alvo.skew())},
            {"metrica": "kurtosis (excesso)", "valor": float(alvo.kurt())},
            {"metrica": "zeros_pct", "valor": float(100 * (alvo == 0).mean())},
            {"metrica": "negativos_pct", "valor": float(100 * (alvo < 0).mean())},
            {"metrica": "outliers_IQR", "valor": n_out},
            {"metrica": "outliers_IQR_pct", "valor": pct_out},
        ]
    )

    amostra = alvo.sample(n=min(5000, len(alvo)), random_state=cfg.random_state)
    _, p_normal = stats.normaltest(amostra)
    stats_alvo.loc[len(stats_alvo)] = {"metrica": "p-valor normalidade (D'Agostino, n=5k)", "valor": float(p_normal)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    ax = axes[0, 0]
    ax.hist(alvo, bins=60, color=SERIE_1, edgecolor="none")
    ax.axvline(alvo.mean(), color=SERIE_2, linewidth=2.0, label=f"media {alvo.mean():.2f}")
    ax.axvline(alvo.median(), color=SERIE_3, linewidth=2.0, linestyle="--",
               label=f"mediana {alvo.median():.2f}")
    ax.set_title(f"Histograma de {cfg.target}")
    ax.set_xlabel(cfg.target)
    ax.set_ylabel("frequencia")
    ax.legend()

    ax = axes[0, 1]
    caixa = ax.boxplot(alvo, vert=False, widths=0.5, patch_artist=True, showfliers=True,
                       flierprops={"marker": ".", "markersize": 2,
                                   "markerfacecolor": TINTA_SUAVE,
                                   "markeredgecolor": "none", "alpha": 0.3})
    caixa["boxes"][0].set(facecolor=SERIE_1, edgecolor=EIXO, alpha=0.85)
    for elemento in ("whiskers", "caps"):
        for artista in caixa[elemento]:
            artista.set(color=EIXO, linewidth=1.2)
    caixa["medians"][0].set(color=SERIE_2, linewidth=2.0)
    ax.set_title("Boxplot (dispersao e outliers IQR)")
    ax.set_xlabel(cfg.target)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)

    ax = axes[1, 0]
    ordenado = np.sort(alvo.to_numpy())
    ax.plot(ordenado, np.arange(1, len(ordenado) + 1) / len(ordenado), color=SERIE_1)
    ax.set_title("Distribuicao acumulada (ECDF)")
    ax.set_xlabel(cfg.target)
    ax.set_ylabel("proporcao acumulada")

    ax = axes[1, 1]
    (osm, osr), (inclinacao, intercepto, _) = stats.probplot(amostra, dist="norm")
    ax.scatter(osm, osr, s=6, color=SERIE_1, alpha=0.5, edgecolors="none",
               label="quantis observados")
    ax.plot(osm, inclinacao * osm + intercepto, color=SERIE_2, label="normal teorica")
    ax.set_title("Q-Q plot vs normal (amostra 5k)")
    ax.set_xlabel("quantis teoricos")
    ax.set_ylabel("quantis observados")
    ax.legend()

    fig.suptitle(f"Fase 2.2 - Distribuicao do alvo: {cfg.target}",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return stats_alvo, {"distribuicao_alvo": salvar_figura(fig, "f2_2_distribuicao_alvo", cfg)}


# ---------------------------------------------------------------------------
# 2.3 Variaveis numericas
# ---------------------------------------------------------------------------
def analisar_numericas(
    df: pd.DataFrame, cfg: EDAConfig, colunas: Colunas
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Estatisticas estendidas, outliers IQR e relacao de cada numerica com o alvo."""
    nums = colunas.numericas
    if not nums:
        return pd.DataFrame(), pd.DataFrame(), {}

    linhas = []
    for c in nums:
        serie = df[c]
        limpa = serie.dropna()
        inf, sup = limites_iqr(limpa) if not limpa.empty else (np.nan, np.nan)
        n_out, pct_out = contar_outliers_iqr(serie)
        linhas.append(
            {
                "coluna": c,
                "n": int(limpa.size),
                "nan_pct": round(100 * serie.isna().mean(), 3),
                "media": limpa.mean(),
                "mediana": limpa.median(),
                "desvio": limpa.std(),
                "min": limpa.min(),
                "max": limpa.max(),
                "skew": limpa.skew(),
                "kurtosis": limpa.kurt(),
                "iqr_inf": inf,
                "iqr_sup": sup,
                "outliers_iqr": n_out,
                "outliers_pct": round(pct_out, 3),
                "zeros_pct": round(100 * (limpa == 0).mean(), 3) if limpa.size else 0.0,
                "n_valores_distintos": int(limpa.nunique()),
            }
        )
    resumo = pd.DataFrame(linhas)

    # Correlacao de cada numerica com o alvo (linear e monotonica).
    corr_linhas = []
    base = df[nums + [cfg.target]]
    for c in nums:
        par = base[[c, cfg.target]].dropna()
        if len(par) < 3:
            corr_linhas.append({"coluna": c, "pearson": np.nan, "spearman": np.nan, "n_pares": len(par)})
            continue
        corr_linhas.append(
            {
                "coluna": c,
                "pearson": float(par[c].corr(par[cfg.target])),
                "spearman": float(par[c].corr(par[cfg.target], method="spearman")),
                "n_pares": len(par),
            }
        )
    corr_alvo = pd.DataFrame(corr_linhas)
    corr_alvo["abs_pearson"] = corr_alvo["pearson"].abs()
    corr_alvo = corr_alvo.sort_values("abs_pearson", ascending=False).drop(columns="abs_pearson")

    figuras: dict[str, str] = {}

    # Histogramas
    fig, axes = grade_de_paineis(len(nums), cfg)
    for ax, c in zip(axes, nums):
        ax.hist(df[c].dropna(), bins=50, color=SERIE_1, edgecolor="none")
        ax.set_title(c)
        ax.set_ylabel("frequencia")
    fig.suptitle("Fase 2.3 - Distribuicoes das variaveis numericas",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    figuras["hist_numericas"] = salvar_figura(fig, "f2_3_hist_numericas", cfg)

    # Boxplots (escala propria por painel)
    fig, axes = grade_de_paineis(len(nums), cfg)
    for ax, c in zip(axes, nums):
        limpa = df[c].dropna()
        caixa = ax.boxplot(limpa, vert=False, widths=0.5, patch_artist=True,
                           flierprops={"marker": ".", "markersize": 2,
                                       "markerfacecolor": TINTA_SUAVE,
                                       "markeredgecolor": "none", "alpha": 0.25})
        caixa["boxes"][0].set(facecolor=SERIE_1, edgecolor=EIXO, alpha=0.85)
        for elemento in ("whiskers", "caps"):
            for artista in caixa[elemento]:
                artista.set(color=TINTA_SUAVE, linewidth=1.2)
        caixa["medians"][0].set(color=SERIE_2, linewidth=2.0)
        n_out, pct_out = contar_outliers_iqr(limpa)
        # A contagem absoluta importa: em amostras grandes um punhado de erros
        # graves ainda arredonda para 0.00%.
        ax.set_title(f"{c}  -  {n_out:,} outliers IQR ({pct_out:.3f}%)".replace(",", "."))
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fase 2.3 - Outliers pela regra IQR",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    figuras["box_numericas"] = salvar_figura(fig, "f2_3_box_numericas", cfg)

    # Dispersao vs alvo, com media do alvo por decil da preditora
    amostra = amostrar(df[nums + [cfg.target]], cfg.amostra_scatter, cfg.random_state)
    fig, axes = grade_de_paineis(len(nums), cfg)
    for ax, c in zip(axes, nums):
        par = amostra[[c, cfg.target]].dropna()
        ax.scatter(par[c], par[cfg.target], s=4, alpha=0.18, color=SERIE_1,
                   edgecolors="none", label="observacoes")
        completo = df[[c, cfg.target]].dropna()
        if completo[c].nunique() > 10:
            faixas = pd.qcut(completo[c], q=10, duplicates="drop")
            perfil = completo.groupby(faixas, observed=True).agg(
                x=(c, "mean"), y=(cfg.target, "mean")
            )
            ax.plot(perfil["x"], perfil["y"], color=SERIE_2, marker="o",
                    markersize=4, label="media do alvo por decil")
        ax.set_title(c)
        ax.set_xlabel(c)
        ax.set_ylabel(cfg.target)
        ax.legend()
    fig.suptitle(f"Fase 2.3 - Relacao com o alvo (amostra de {len(amostra):,} linhas)".replace(",", "."),
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    figuras["scatter_numericas"] = salvar_figura(fig, "f2_3_scatter_vs_alvo", cfg)

    return resumo, corr_alvo, figuras


# ---------------------------------------------------------------------------
# 2.4 Variaveis categoricas
# ---------------------------------------------------------------------------
def analisar_categoricas(
    df: pd.DataFrame, cfg: EDAConfig, colunas: Colunas
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Frequencias, efeito no alvo por categoria e forca de associacao (eta^2)."""
    cats = colunas.todas_categoricas
    if not cats:
        return pd.DataFrame(), {}

    figuras: dict[str, str] = {}
    resumo_linhas = []
    detalhes: list[pd.DataFrame] = []

    amostra_box = amostrar(df, cfg.amostra_boxplot, cfg.random_state)

    for c in cats:
        # Estatisticas do alvo por categoria (dataset completo).
        agrupado = (
            df.groupby(c, observed=True)[cfg.target]
            .agg(n="count", media="mean", mediana="median", desvio="std")
            .reset_index()
        )
        agrupado["freq_pct"] = round(100 * agrupado["n"] / len(df), 3)
        agrupado.insert(0, "variavel", c)
        detalhes.append(agrupado.sort_values("media", ascending=False))

        eta2 = eta_quadrado(df, c, cfg.target)
        resumo_linhas.append(
            {
                "coluna": c,
                "n_categorias": int(df[c].nunique(dropna=True)),
                "nan_pct": round(100 * df[c].isna().mean(), 3),
                "categoria_dominante": agrupado.loc[agrupado["n"].idxmax(), c],
                "dominante_pct": float(agrupado["freq_pct"].max()),
                "media_alvo_min": float(agrupado["media"].min()),
                "media_alvo_max": float(agrupado["media"].max()),
                "amplitude_media_alvo": float(agrupado["media"].max() - agrupado["media"].min()),
                "eta2_vs_alvo": round(eta2, 5) if pd.notna(eta2) else np.nan,
            }
        )

        # Figura: frequencia + distribuicao do alvo por categoria.
        top = agrupado.nlargest(cfg.top_n_categorias, "n").sort_values("n")
        niveis = top[c].tolist()
        altura = max(3.2, 0.32 * len(niveis) + 1.6)
        fig, (ax_freq, ax_alvo) = plt.subplots(1, 2, figsize=(12.5, altura))

        pos = np.arange(len(niveis))
        ax_freq.barh(pos, top["n"], color=SERIE_1, height=0.62)
        ax_freq.set_yticks(pos, [str(v) for v in niveis])
        ax_freq.set_xlabel("contagem")
        ax_freq.set_title(f"{c} - frequencia (top {len(niveis)})")
        ax_freq.grid(axis="y", visible=False)

        grupos = [
            amostra_box.loc[amostra_box[c] == nivel, cfg.target].dropna().to_numpy()
            for nivel in niveis
        ]
        caixa = ax_alvo.boxplot(grupos, vert=False, widths=0.6, patch_artist=True,
                                flierprops={"marker": ".", "markersize": 1.5,
                                            "markerfacecolor": TINTA_SUAVE,
                                            "markeredgecolor": "none", "alpha": 0.2})
        for artista in caixa["boxes"]:
            artista.set(facecolor=SERIE_1, edgecolor=EIXO, alpha=0.8)
        for elemento in ("whiskers", "caps"):
            for artista in caixa[elemento]:
                artista.set(color=TINTA_SUAVE, linewidth=1.0)
        for artista in caixa["medians"]:
            artista.set(color=SERIE_2, linewidth=1.8)
        ax_alvo.set_yticks(pos + 1, [str(v) for v in niveis])
        ax_alvo.set_xlabel(cfg.target)
        rotulo_eta = f"eta2 = {eta2:.4f}" if pd.notna(eta2) else "eta2 indefinido"
        ax_alvo.set_title(f"{cfg.target} por {c}  -  {rotulo_eta}")
        ax_alvo.grid(axis="y", visible=False)

        fig.suptitle(f"Fase 2.4 - Variavel categorica: {c}",
                     fontsize=12.5, fontweight="semibold", color=TINTA_SECUNDARIA)
        fig.tight_layout()
        figuras[f"categorica_{c}"] = salvar_figura(fig, f"f2_4_categorica_{c}", cfg)

    resumo = pd.DataFrame(resumo_linhas).sort_values("eta2_vs_alvo", ascending=False)
    salvar_tabela(pd.concat(detalhes, ignore_index=True), "f2_4_alvo_por_categoria", cfg)

    # Comparativo de poder explicativo entre as categoricas.
    ordenado = resumo.dropna(subset=["eta2_vs_alvo"]).sort_values("eta2_vs_alvo")
    if not ordenado.empty:
        fig, ax = plt.subplots(figsize=(7.5, max(2.4, 0.45 * len(ordenado) + 1.2)))
        pos = np.arange(len(ordenado))
        ax.barh(pos, ordenado["eta2_vs_alvo"], color=SERIE_1, height=0.6)
        ax.set_yticks(pos, ordenado["coluna"])
        ax.set_xlabel(f"eta^2 - variancia de {cfg.target} explicada")
        ax.set_title("Fase 2.4 - Poder explicativo das variaveis categoricas")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(ordenado["eta2_vs_alvo"].max()) * 1.18 or 1.0)
        for y, valor in zip(pos, ordenado["eta2_vs_alvo"]):
            ax.text(valor, y, f" {valor:.4f}", va="center", fontsize=8, color=TINTA_SECUNDARIA)
        fig.tight_layout()
        figuras["eta2_categoricas"] = salvar_figura(fig, "f2_4_eta2_categoricas", cfg)

    return resumo, figuras


# ---------------------------------------------------------------------------
# 2.5 Matriz de correlacao
# ---------------------------------------------------------------------------
def matriz_correlacao(
    df: pd.DataFrame, cfg: EDAConfig, colunas: Colunas
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Pearson (linear) e Spearman (monotonica) + pares redundantes."""
    nums = colunas.numericas + [cfg.target]
    if len(nums) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    base = df[nums]
    pearson = base.corr(method="pearson")
    spearman = base.corr(method="spearman")

    # Pares redundantes ENTRE PREDITORAS (insumo para selecao de features).
    # O alvo fica de fora: correlacao alta com ele e sinal desejavel, nao
    # motivo para descartar coluna.
    preditoras = [c for c in pearson.columns if c != cfg.target]
    pares = []
    for i, a in enumerate(preditoras):
        for b in preditoras[i + 1:]:
            r = pearson.loc[a, b]
            if abs(r) >= cfg.limite_correlacao_alta:
                pares.append({"variavel_a": a, "variavel_b": b, "pearson": round(float(r), 4)})
    redundantes = pd.DataFrame(pares)

    def desenhar(ax: plt.Axes, matriz: pd.DataFrame, titulo: str, rotular_y: bool):
        imagem = ax.imshow(matriz.to_numpy(), cmap=DIVERGENTE, vmin=-1, vmax=1)
        rotulos = [str(c) for c in matriz.columns]
        ax.set_xticks(range(len(rotulos)), rotulos, rotation=40, ha="right")
        # O painel da direita repete as mesmas linhas: rotular so o da esquerda
        # evita que os textos invadam o heatmap vizinho.
        ax.set_yticks(range(len(rotulos)), rotulos if rotular_y else [""] * len(rotulos))
        ax.set_title(titulo)
        ax.grid(visible=False)
        for i in range(len(rotulos)):
            for j in range(len(rotulos)):
                valor = matriz.iat[i, j]
                cor = "#ffffff" if abs(valor) > 0.55 else TINTA_SECUNDARIA
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8, color=cor)
        return imagem

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2 * 2, 5.6))
    desenhar(ax1, pearson, "Pearson (relacao linear)", rotular_y=True)
    imagem = desenhar(ax2, spearman, "Spearman (relacao monotonica)", rotular_y=False)
    barra = fig.colorbar(imagem, ax=[ax1, ax2], fraction=0.025, pad=0.02)
    barra.set_label("coeficiente de correlacao", color=TINTA_SECUNDARIA, fontsize=8.5)
    barra.outline.set_visible(False)
    fig.suptitle("Fase 2.5 - Matriz de correlacao entre variaveis numericas",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    figuras = {"correlacao": salvar_figura(fig, "f2_5_matriz_correlacao", cfg)}

    return pearson, spearman, redundantes, figuras


# ---------------------------------------------------------------------------
# 2.6 Consolidacao de alertas
# ---------------------------------------------------------------------------
def consolidar_alertas(
    df: pd.DataFrame,
    cfg: EDAConfig,
    colunas: Colunas,
    descricao: pd.DataFrame,
    resumo_num: pd.DataFrame,
    corr_alvo: pd.DataFrame,
    resumo_cat: pd.DataFrame,
    redundantes: pd.DataFrame,
) -> pd.DataFrame:
    """Converte os achados da EDA em uma lista priorizada de acoes para a Fase 3."""
    alertas: list[dict[str, object]] = []

    def add(severidade: str, coluna: str, achado: str, acao: str) -> None:
        alertas.append({"severidade": severidade, "coluna": coluna, "achado": achado, "acao_sugerida": acao})

    # Ausentes
    for _, r in descricao.iterrows():
        if r["nan_pct"] > cfg.limite_nan_coluna:
            add("critico", r["coluna"], f"{r['nan_pct']:.2f}% ausentes",
                "candidata a descarte pela regra do UML (> 50%)")
        # A contagem absoluta e o gatilho: um punhado de NaN em 750k linhas
        # arredonda para 0.00% e escaparia de um teste sobre o percentual.
        elif r["n_nan"] > 0 and r["coluna"] != cfg.target:
            add("atencao", r["coluna"], f"{int(r['n_nan'])} ausentes ({r['nan_pct']:.3f}%)",
                "imputar (mediana/moda/KNN) e criar flag de ausencia")
        if r["papel"] == "constante":
            add("critico", r["coluna"], "coluna constante", "descartar - nao carrega informacao")
        if r["papel"] == "identificador":
            add("info", r["coluna"], f"unicidade de {r['unicidade_pct']:.1f}%",
                "tratar como identificador, fora do conjunto de features")

    # Numericas
    for _, r in resumo_num.iterrows():
        if r["outliers_pct"] > cfg.limite_outliers_pct:
            add("atencao", r["coluna"], f"{r['outliers_pct']:.2f}% de outliers IQR",
                "avaliar clipping IQR / winsorizacao")
        if abs(r["skew"]) > cfg.limite_skew:
            add("atencao", r["coluna"], f"assimetria de {r['skew']:.2f}",
                "avaliar transformacao (log1p, Box-Cox) para modelos lineares")
        if r["n_valores_distintos"] <= cfg.max_valores_discretos:
            add("info", r["coluna"], f"apenas {r['n_valores_distintos']} valores distintos",
                "numerica discreta - considerar tratamento categorico")
        # Heuristica: colunas de percentual fora de [0, 100].
        if any(t in r["coluna"].lower() for t in ("percentage", "percent", "_pct", "pct_")):
            if r["max"] > 100 or r["min"] < 0:
                add("critico", r["coluna"],
                    f"valores fora de [0, 100] (min {r['min']:.2f}, max {r['max']:.2f})",
                    "valores impossiveis para um percentual - corrigir ou tratar como ausentes")

    # Categoricas
    for _, r in resumo_cat.iterrows():
        if r["dominante_pct"] > cfg.limite_categoria_dominante:
            add("atencao", r["coluna"],
                f"categoria '{r['categoria_dominante']}' concentra {r['dominante_pct']:.1f}%",
                "coluna quase constante - avaliar descarte")
        if r["n_categorias"] > cfg.max_cardinalidade_categorica:
            add("atencao", r["coluna"], f"{r['n_categorias']} categorias",
                "OHE explode a dimensionalidade - usar target/ordinal encoding ou agrupamento")
        if pd.notna(r["eta2_vs_alvo"]) and r["eta2_vs_alvo"] < 0.001:
            add("info", r["coluna"], f"eta2 = {r['eta2_vs_alvo']:.5f}",
                "praticamente nenhuma associacao com o alvo - baixa prioridade")

    # Forca do sinal de cada preditora numerica
    for _, r in corr_alvo.iterrows():
        if pd.notna(r["pearson"]) and abs(r["pearson"]) >= cfg.limite_correlacao_alta:
            add("info", r["coluna"], f"pearson {r['pearson']:.3f} com o alvo",
                "preditora dominante - verificar se nao ha vazamento e usar como baseline")
        elif pd.notna(r["pearson"]) and abs(r["pearson"]) < 0.02:
            add("info", r["coluna"], f"pearson {r['pearson']:.3f} com o alvo",
                "sinal linear desprezivel - checar relacao nao linear antes de descartar")

    # Redundancia entre preditoras
    for _, r in redundantes.iterrows():
        add("atencao", f"{r['variavel_a']} / {r['variavel_b']}",
            f"correlacao de {r['pearson']:.3f}",
            "par redundante - manter apenas uma das duas")

    # Regras logicas de dominio
    for regra in cfg.regras_logicas:
        try:
            violacoes = len(df.query(regra))
        except Exception as exc:  # regra invalida nao deve derrubar a EDA
            add("info", regra, f"regra nao avaliada ({exc})", "revisar a sintaxe da regra")
            continue
        if violacoes:
            add("critico", regra, f"{violacoes} linhas violam a regra ({100 * violacoes / len(df):.3f}%)",
                "inconsistencia logica - investigar antes de modelar")

    ordem = {"critico": 0, "atencao": 1, "info": 2}
    resultado = pd.DataFrame(alertas)
    if resultado.empty:
        return resultado
    return (
        resultado.assign(_ordem=resultado["severidade"].map(ordem))
        .sort_values(["_ordem", "coluna"])
        .drop(columns="_ordem")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Orquestracao da fase
# ---------------------------------------------------------------------------
def executar(df: pd.DataFrame, cfg: EDAConfig) -> ResultadoFase2:
    """Roda os blocos 2.1 a 2.6 e grava tabelas/figuras em `cfg.outdir`."""
    if cfg.target not in df.columns:
        raise KeyError(f"coluna alvo '{cfg.target}' nao existe no DataFrame")

    cfg.preparar()
    colunas = classificar_colunas(df, cfg)
    figuras: dict[str, str] = {}

    descricao = descrever_variaveis(df, cfg, colunas)
    figuras |= _figura_nan(descricao, cfg)

    stats_alvo, fig_alvo = analisar_alvo(df, cfg)
    figuras |= fig_alvo

    resumo_num, corr_alvo, fig_num = analisar_numericas(df, cfg, colunas)
    figuras |= fig_num

    resumo_cat, fig_cat = analisar_categoricas(df, cfg, colunas)
    figuras |= fig_cat

    pearson, spearman, redundantes, fig_corr = matriz_correlacao(df, cfg, colunas)
    figuras |= fig_corr

    alertas = consolidar_alertas(df, cfg, colunas, descricao, resumo_num, corr_alvo,
                                 resumo_cat, redundantes)

    salvar_tabela(descricao, "f2_1_descricao_variaveis", cfg)
    salvar_tabela(stats_alvo, "f2_2_estatisticas_alvo", cfg)
    salvar_tabela(resumo_num, "f2_3_resumo_numericas", cfg)
    salvar_tabela(corr_alvo, "f2_3_correlacao_com_alvo", cfg)
    salvar_tabela(resumo_cat, "f2_4_resumo_categoricas", cfg)
    salvar_tabela(pearson.reset_index(names="variavel"), "f2_5_pearson", cfg)
    salvar_tabela(spearman.reset_index(names="variavel"), "f2_5_spearman", cfg)
    if not redundantes.empty:
        salvar_tabela(redundantes, "f2_5_pares_redundantes", cfg)
    salvar_tabela(alertas, "f2_6_alertas", cfg)

    markdown = _montar_markdown(cfg, colunas, descricao, stats_alvo, resumo_num,
                               corr_alvo, resumo_cat, redundantes, alertas, figuras)

    return ResultadoFase2(
        colunas=colunas,
        descricao=descricao,
        alvo=stats_alvo,
        numericas=resumo_num,
        correlacao_alvo=corr_alvo,
        categoricas=resumo_cat,
        pearson=pearson,
        spearman=spearman,
        pares_redundantes=redundantes,
        alertas=alertas,
        figuras=figuras,
        markdown=markdown,
    )


def _montar_markdown(
    cfg: EDAConfig,
    colunas: Colunas,
    descricao: pd.DataFrame,
    stats_alvo: pd.DataFrame,
    resumo_num: pd.DataFrame,
    corr_alvo: pd.DataFrame,
    resumo_cat: pd.DataFrame,
    redundantes: pd.DataFrame,
    alertas: pd.DataFrame,
    figuras: dict[str, str],
) -> str:
    def bloco(nome: str) -> str:
        caminho = figuras.get(nome)
        return f"\n![{nome}]({caminho})\n" if caminho else ""

    partes = [
        "## Fase 2 - Analise Exploratoria (EDA)\n",
        "> Engenharia de features fica fora deste escopo, conforme combinado.\n",
        "### 2.1 Descricao das variaveis\n",
        f"Papeis detectados automaticamente: **{len(colunas.numericas)} numericas**, "
        f"**{len(colunas.categoricas)} categoricas**, "
        f"**{len(colunas.alta_cardinalidade)} de alta cardinalidade**, "
        f"**{len(colunas.identificadores)} identificadores**, "
        f"**{len(colunas.constantes)} constantes**.\n",
        df_para_markdown(descricao),
        bloco("valores_ausentes"),
        "\n### 2.2 Distribuicao da variavel alvo\n",
        df_para_markdown(stats_alvo),
        bloco("distribuicao_alvo"),
        "\n### 2.3 Variaveis numericas\n",
        df_para_markdown(resumo_num),
        "\n**Correlacao de cada numerica com o alvo**\n",
        df_para_markdown(corr_alvo),
        bloco("hist_numericas"),
        bloco("box_numericas"),
        bloco("scatter_numericas"),
        "\n### 2.4 Variaveis categoricas\n",
        "`eta2` = fracao da variancia do alvo explicada pela variavel (ANOVA de um fator).\n",
        df_para_markdown(resumo_cat),
        bloco("eta2_categoricas"),
    ]
    for nome in figuras:
        if nome.startswith("categorica_"):
            partes.append(bloco(nome))

    partes += [
        "\n### 2.5 Matriz de correlacao\n",
        bloco("correlacao"),
        "\n**Pares de preditoras acima do limite de redundancia "
        f"(|r| >= {cfg.limite_correlacao_alta}); o alvo fica de fora**\n",
        df_para_markdown(redundantes) if not redundantes.empty
        else "_Nenhum par de preditoras redundante._",
        "\n### 2.6 Alertas consolidados (entrada da Fase 3)\n",
        df_para_markdown(alertas, max_linhas=100),
    ]
    return "\n".join(partes)
