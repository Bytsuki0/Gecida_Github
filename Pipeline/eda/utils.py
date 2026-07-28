"""Utilitarios compartilhados pelas Fases 1 e 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EDAConfig


# ---------------------------------------------------------------------------
# Classificacao de colunas
# ---------------------------------------------------------------------------
@dataclass
class Colunas:
    """Particao das colunas do dataset por papel na analise."""

    numericas: list[str] = field(default_factory=list)
    categoricas: list[str] = field(default_factory=list)
    alta_cardinalidade: list[str] = field(default_factory=list)
    constantes: list[str] = field(default_factory=list)
    identificadores: list[str] = field(default_factory=list)

    @property
    def todas_categoricas(self) -> list[str]:
        """Categoricas + alta cardinalidade (ambas entram nas analises)."""
        return self.categoricas + self.alta_cardinalidade


def classificar_colunas(df: pd.DataFrame, cfg: EDAConfig) -> Colunas:
    """Separa as colunas por papel, ignorando alvo e id.

    A classificacao e puramente estrutural (dtype + cardinalidade), de modo que
    funciona em qualquer dataset sem conhecimento de dominio.
    """
    col = Colunas()
    n = max(len(df), 1)
    ignorar = {cfg.target}
    if cfg.id_col:
        ignorar.add(cfg.id_col)

    for nome in df.columns:
        if nome in ignorar:
            continue
        serie = df[nome]
        nunique = serie.nunique(dropna=True)

        if nunique <= 1:
            col.constantes.append(nome)
        elif nunique / n >= cfg.razao_identificador:
            col.identificadores.append(nome)
        elif pd.api.types.is_numeric_dtype(serie):
            col.numericas.append(nome)
        elif nunique > cfg.max_cardinalidade_categorica:
            col.alta_cardinalidade.append(nome)
        else:
            col.categoricas.append(nome)

    return col


# ---------------------------------------------------------------------------
# Estatisticas reutilizaveis
# ---------------------------------------------------------------------------
def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Limites inferior/superior da regra IQR (Tukey)."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def contar_outliers_iqr(serie: pd.Series, k: float = 1.5) -> tuple[int, float]:
    """Numero e percentual de outliers pela regra IQR."""
    limpa = serie.dropna()
    if limpa.empty:
        return 0, 0.0
    inf, sup = limites_iqr(limpa, k)
    n = int(((limpa < inf) | (limpa > sup)).sum())
    return n, 100.0 * n / len(limpa)


def eta_quadrado(df: pd.DataFrame, categoria: str, alvo: str) -> float:
    """Eta^2: fracao da variancia do alvo explicada pela variavel categorica.

    Equivale ao R^2 de uma ANOVA de um fator. Serve como medida de forca de
    associacao categorica -> alvo continuo (0 = nenhuma, 1 = total).
    """
    dados = df[[categoria, alvo]].dropna()
    if dados.empty or dados[categoria].nunique() < 2:
        return float("nan")
    media_geral = dados[alvo].mean()
    grupos = dados.groupby(categoria, observed=True)[alvo]
    entre = (grupos.count() * (grupos.mean() - media_geral) ** 2).sum()
    total = ((dados[alvo] - media_geral) ** 2).sum()
    return float(entre / total) if total > 0 else float("nan")


def psi(esperado: pd.Series, observado: pd.Series, bins: int = 10) -> float:
    """Population Stability Index entre duas distribuicoes.

    < 0.10 estavel | 0.10-0.25 atencao | > 0.25 deslocamento forte.
    """
    esp, obs = esperado.dropna(), observado.dropna()
    if esp.empty or obs.empty:
        return float("nan")

    if pd.api.types.is_numeric_dtype(esp):
        cortes = np.unique(esp.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
        if len(cortes) < 3:
            return float("nan")
        cortes[0], cortes[-1] = -np.inf, np.inf
        p_esp = pd.cut(esp, cortes).value_counts(normalize=True, sort=False)
        p_obs = pd.cut(obs, cortes).value_counts(normalize=True, sort=False)
    else:
        p_esp = esp.astype(str).value_counts(normalize=True)
        p_obs = obs.astype(str).value_counts(normalize=True)

    p_obs = p_obs.reindex(p_esp.index).fillna(0.0)
    eps = 1e-6
    p_esp = p_esp.clip(lower=eps)
    p_obs = p_obs.clip(lower=eps)
    return float(((p_obs - p_esp) * np.log(p_obs / p_esp)).sum())


def amostrar(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Amostra sem reposicao quando o dataset excede `n` linhas."""
    return df.sample(n=n, random_state=random_state) if len(df) > n else df


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def salvar_tabela(df: pd.DataFrame, nome: str, cfg: EDAConfig) -> Path:
    """Grava a tabela em CSV dentro de `outdir/tabelas`."""
    caminho = cfg.dir_tabelas / f"{nome}.csv"
    df.to_csv(caminho, index=False, encoding="utf-8")
    return caminho


def salvar_figura(fig: plt.Figure, nome: str, cfg: EDAConfig) -> str:
    """Grava a figura e devolve o caminho relativo ao relatorio."""
    caminho = cfg.dir_figuras / f"{nome}.{cfg.formato_figura}"
    fig.savefig(caminho, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return f"figuras/{caminho.name}"


def grade_de_paineis(n: int, cfg: EDAConfig) -> tuple[plt.Figure, list[plt.Axes]]:
    """Cria uma grade de subplots dimensionada para `n` paineis."""
    ncols = min(cfg.colunas_grade, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cfg.largura_painel * ncols, cfg.altura_painel * nrows),
    )
    lista = np.atleast_1d(axes).ravel().tolist()
    for ax in lista[n:]:
        ax.set_visible(False)
    return fig, lista[:n]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def df_para_markdown(df: pd.DataFrame, max_linhas: int = 60) -> str:
    """Converte um DataFrame em tabela markdown (sem dependencia externa)."""
    if df.empty:
        return "_(vazio)_"

    visao = df.head(max_linhas)
    cabecalho = [str(c) for c in visao.columns]

    def fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.4g}"
        return str(v).replace("|", "\\|")

    linhas = ["| " + " | ".join(cabecalho) + " |",
              "|" + "|".join(["---"] * len(cabecalho)) + "|"]
    for _, row in visao.iterrows():
        linhas.append("| " + " | ".join(fmt(v) for v in row) + " |")

    if len(df) > max_linhas:
        linhas.append(f"\n_Exibindo {max_linhas} de {len(df)} linhas; tabela completa em `tabelas/`._")
    return "\n".join(linhas)
