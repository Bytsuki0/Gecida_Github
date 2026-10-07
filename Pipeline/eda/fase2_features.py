"""Fase 2.7 - Engenharia de features (EDA).

Ultimo bloco da Fase 2 do diagrama ("Is_Weekend · interacoes Length x Host_Pop"):

    2.7.1  Construcao - indicadoras, numeros extraidos de texto, interacoes
    2.7.2  Avaliacao  - correlacao/eta2 com o alvo e ganho de R^2 de cada interacao
    2.7.3  Figura     - media do alvo por grupo, por valor e por decil

Aqui as features sao apenas *avaliadas*. Quem as materializa no dataset final e
a Fase 3 (`fase3_preprocessamento`), depois da limpeza e da imputacao: uma
interacao calculada sobre valores sujos (duracao de 78 milhoes de minutos no
test) ou ausentes herdaria esses problemas.

O codigo nao conhece o dataset de podcasts: o que construir vem em `EspecFeatures`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import EDAConfig
from .estilo import SERIE_1, SERIE_2, SERIE_3, TINTA_SECUNDARIA, TINTA_SUAVE
from .utils import df_para_markdown, eta_quadrado, grade_de_paineis, salvar_figura, salvar_tabela


@dataclass
class EspecFeatures:
    """Quais features derivadas construir.

    Attributes:
        indicadoras: nome -> (coluna, valores); vale 1 quando a coluna esta em `valores`.
        numeros_em_texto: nome -> coluna de texto; extrai o primeiro inteiro do texto.
        interacoes: nome -> (coluna_a, coluna_b); produto a * b.
    """

    indicadoras: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    numeros_em_texto: dict[str, str] = field(default_factory=dict)
    interacoes: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def nomes(self) -> list[str]:
        return [*self.indicadoras, *self.numeros_em_texto, *self.interacoes]


@dataclass
class ResultadoFase2Features:
    """Artefatos produzidos pelo bloco 2.7."""

    avaliacao: pd.DataFrame
    figuras: dict[str, str] = field(default_factory=dict)
    markdown: str = ""


# ---------------------------------------------------------------------------
# 2.7.1 Construcao
# ---------------------------------------------------------------------------
def criar_features(df: pd.DataFrame, espec: EspecFeatures) -> pd.DataFrame:
    """Devolve uma copia de `df` com as features de `espec` adicionadas.

    Funcao pura e sem estado: o mesmo calculo vale para train e test, entao nao
    ha nada a ajustar. NaN na coluna de origem gera NaN na feature.
    """
    saida = df.copy()

    for nome, (coluna, valores) in espec.indicadoras.items():
        serie = saida[coluna]
        indicadora = serie.isin(valores)
        saida[nome] = (indicadora.astype("int8") if serie.notna().all()
                       else indicadora.astype(float).where(serie.notna()))

    for nome, coluna in espec.numeros_em_texto.items():
        extraido = saida[coluna].astype(str).str.extract(r"(\d+)", expand=False)
        numeros = pd.to_numeric(extraido, errors="coerce")
        saida[nome] = numeros.astype("int64") if numeros.notna().all() else numeros

    for nome, (a, b) in espec.interacoes.items():
        saida[nome] = saida[a] * saida[b]

    return saida


# ---------------------------------------------------------------------------
# 2.7.2 Avaliacao
# ---------------------------------------------------------------------------
def _r2_ols(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 de uma regressao linear com intercepto (minimos quadrados)."""
    x = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuo = y - x @ coef
    centrado = y - y.mean()
    return float(1 - (residuo @ residuo) / (centrado @ centrado))


def avaliar_features(
    df: pd.DataFrame, espec: EspecFeatures, cfg: EDAConfig, preditoras: list[str]
) -> pd.DataFrame:
    """Uma linha por feature: forca da relacao com o alvo e redundancia.

    - indicadoras: eta2 da feature vs eta2 da coluna categorica de origem
      (quanto do poder explicativo da origem a versao binaria retem);
    - numeros em texto: correlacao linear/monotonica vs eta2 da coluna de texto;
    - interacoes: R^2 de `alvo ~ a + b` contra `alvo ~ a + b + a*b` (casos
      completos). O ganho isola o que o produto acrescenta aos efeitos principais.

    `preditoras` sao as numericas originais usadas na checagem de redundancia.
    """
    alvo = cfg.target
    linhas: list[dict[str, object]] = []

    def base(nome: str, tipo: str, definicao: str) -> dict[str, object]:
        par = df[[nome, alvo]].dropna()
        linha: dict[str, object] = {
            "feature": nome,
            "tipo": tipo,
            "definicao": definicao,
            "nan_pct": round(100 * df[nome].isna().mean(), 3),
            "pearson_alvo": float(par[nome].corr(par[alvo])),
            "spearman_alvo": float(par[nome].corr(par[alvo], method="spearman")),
        }
        candidatas = [p for p in preditoras if p in df.columns and p != nome]
        if candidatas:
            corr = df[candidatas].corrwith(df[nome]).abs()
            linha["preditora_mais_correlacionada"] = corr.idxmax()
            linha["corr_com_ela"] = round(float(corr.max()), 4)
        return linha

    for nome, (coluna, valores) in espec.indicadoras.items():
        linha = base(nome, "indicadora", f"{coluna} em {{{', '.join(map(str, valores))}}}")
        linha["eta2_alvo"] = eta_quadrado(df, nome, alvo)
        linha["eta2_coluna_origem"] = eta_quadrado(df, coluna, alvo)
        linhas.append(linha)

    for nome, coluna in espec.numeros_em_texto.items():
        linha = base(nome, "numero em texto", f"inteiro extraido de {coluna}")
        linha["eta2_coluna_origem"] = eta_quadrado(df, coluna, alvo)
        linhas.append(linha)

    for nome, (a, b) in espec.interacoes.items():
        linha = base(nome, "interacao", f"{a} x {b}")
        completo = df[[a, b, alvo]].dropna()
        xa, xb, y = (completo[c].to_numpy(dtype=float) for c in (a, b, alvo))
        r2_sem = _r2_ols(np.column_stack([xa, xb]), y)
        r2_com = _r2_ols(np.column_stack([xa, xb, xa * xb]), y)
        linha |= {
            "r2_efeitos_principais": round(r2_sem, 5),
            "r2_com_interacao": round(r2_com, 5),
            "ganho_r2": round(r2_com - r2_sem, 5),
        }
        linhas.append(linha)

    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 2.7.3 Figura
# ---------------------------------------------------------------------------
def _figura_features(
    df: pd.DataFrame, espec: EspecFeatures, avaliacao: pd.DataFrame, cfg: EDAConfig
) -> dict[str, str]:
    paineis = espec.nomes
    if not paineis:
        return {}
    alvo = cfg.target
    linha_de = avaliacao.set_index("feature")
    media_geral = float(df[alvo].mean())

    fig, axes = grade_de_paineis(len(paineis), cfg)
    eixos = dict(zip(paineis, axes))

    for nome in espec.indicadoras:
        ax = eixos[nome]
        grupos = df.groupby(nome, observed=True)[alvo].agg(["mean", "count"])
        pos = np.arange(len(grupos))
        ax.bar(pos, grupos["mean"], color=SERIE_1, width=0.55)
        ax.set_xticks(pos, [f"{nome} = {int(v)}" for v in grupos.index])
        for x, (media, n) in zip(pos, grupos.itertuples(index=False)):
            ax.text(x, media, f"{media:.2f}\n(n = {n:,})".replace(",", "."),
                    ha="center", va="bottom", fontsize=8, color=TINTA_SECUNDARIA)
        ax.set_ylim(0, grupos["mean"].max() * 1.25)
        ax.set_ylabel(f"media de {alvo}")
        ax.set_title(f"{nome}  -  eta2 = {linha_de.loc[nome, 'eta2_alvo']:.5f}")
        ax.grid(axis="x", visible=False)

    for nome in espec.numeros_em_texto:
        ax = eixos[nome]
        perfil = df.groupby(nome, observed=True)[alvo].mean()
        ax.plot(perfil.index, perfil.to_numpy(), color=SERIE_1, marker="o", markersize=3,
                label=f"media por {nome}")
        ax.axhline(media_geral, color=TINTA_SUAVE, linestyle="--", linewidth=1.2,
                   label=f"media geral {media_geral:.2f}")
        ax.set_xlabel(nome)
        ax.set_ylabel(f"media de {alvo}")
        ax.set_title(f"{nome}  -  pearson = {linha_de.loc[nome, 'pearson_alvo']:+.4f}")
        ax.legend()

    # Interacao: media do alvo por decil de `a`, uma linha por faixa de `b`.
    # Linhas paralelas = efeitos so aditivos; leque abrindo = interacao.
    for nome, (a, b) in espec.interacoes.items():
        ax = eixos[nome]
        dados = df[[a, b, alvo]].dropna()
        if dados[b].nunique() <= 6:
            faixas_b = dados[b].round(2).astype(str)
            ordem = sorted(faixas_b.unique(), key=float)
        else:
            faixas_b = pd.qcut(dados[b], q=3, labels=["terco inferior", "terco medio", "terco superior"])
            ordem = list(faixas_b.cat.categories)
        decis_a = pd.qcut(dados[a], q=10, duplicates="drop")
        perfil = dados.groupby([faixas_b, decis_a], observed=True).agg(x=(a, "mean"), y=(alvo, "mean"))

        # No maximo 3 faixas recebem cor (slots validados para linhas sobrepostas);
        # marcadores distintos sao a codificacao secundaria.
        cores, marcadores = (SERIE_1, SERIE_2, SERIE_3), ("o", "s", "^")
        exibidas = ordem if len(ordem) <= 3 else [ordem[0], ordem[len(ordem) // 2], ordem[-1]]
        for faixa, cor, marcador in zip(exibidas, cores, marcadores):
            trecho = perfil.loc[faixa]
            rotulo = f"{b} = {faixa}" if dados[b].nunique() <= 6 else f"{b}: {faixa}"
            ax.plot(trecho["x"], trecho["y"], color=cor, marker=marcador, markersize=4, label=rotulo)
            ax.annotate(str(faixa).split(" ")[-1], (trecho["x"].iloc[-1], trecho["y"].iloc[-1]),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7.5, color=TINTA_SECUNDARIA)
        ax.set_xlabel(f"{a} (media do decil)")
        ax.set_ylabel(f"media de {alvo}")
        ax.set_title(f"{nome}  -  ganho de R2 = {linha_de.loc[nome, 'ganho_r2']:+.5f}")
        ax.legend(fontsize=7.5)

    fig.suptitle("Fase 2.7 - Engenharia de features: relacao das candidatas com o alvo",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return {"features_candidatas": salvar_figura(fig, "f2_7_features_candidatas", cfg)}


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def executar(
    df: pd.DataFrame,
    cfg: EDAConfig,
    espec: EspecFeatures,
    preditoras: list[str],
    origem_dos_dados: str = "",
) -> ResultadoFase2Features:
    """Constroi as candidatas sobre `df` (com o alvo), avalia e grava tabela/figura.

    `origem_dos_dados` descreve no relatorio qual versao do train foi usada.
    """
    if cfg.target not in df.columns:
        raise KeyError(f"coluna alvo '{cfg.target}' nao existe no DataFrame")
    cfg.preparar()

    com_features = criar_features(df, espec)
    avaliacao = avaliar_features(com_features, espec, cfg, preditoras)
    salvar_tabela(avaliacao, "f2_7_features_candidatas", cfg)
    figuras = _figura_features(com_features, espec, avaliacao, cfg)

    partes = [
        "## Fase 2.7 - Engenharia de features (EDA)\n",
        "Ultimo bloco da Fase 2. As candidatas sao avaliadas aqui e materializadas "
        "na Fase 3 (secao 3.4), depois da limpeza.\n",
    ]
    if origem_dos_dados:
        partes.append(f"Dados avaliados: {origem_dos_dados}\n")
    partes += [
        "- **indicadora**: `eta2_alvo` da versao binaria contra `eta2_coluna_origem` "
        "(quanto do sinal da coluna original ela retem);",
        "- **numero em texto**: `pearson_alvo`^2 e o R^2 de uma reta; `eta2_coluna_origem` "
        "e o teto que a coluna de texto atinge tratada como categoria;",
        "- **interacao**: `ganho_r2` = R^2(alvo ~ a + b + a*b) - R^2(alvo ~ a + b), em "
        "casos completos - o que o produto acrescenta aos efeitos principais.\n",
        df_para_markdown(avaliacao),
    ]
    if "features_candidatas" in figuras:
        partes.append(f"\n![features_candidatas]({figuras['features_candidatas']})\n")

    return ResultadoFase2Features(avaliacao=avaliacao, figuras=figuras, markdown="\n".join(partes))
