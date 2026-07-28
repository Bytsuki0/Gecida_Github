"""Fase 1 - Carregamento de dados.

`pd.read_csv` + inspecao inicial de integridade: shape, memoria, tipos,
alinhamento entre train e test, duplicatas e estabilidade das distribuicoes
(PSI). Nada aqui altera os dados - a fase apenas descreve o que chegou.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import EDAConfig
from .estilo import SERIE_1, SERIE_2, TINTA_SECUNDARIA
from .utils import (
    classificar_colunas,
    df_para_markdown,
    grade_de_paineis,
    psi,
    salvar_figura,
    salvar_tabela,
)


@dataclass
class ResultadoFase1:
    """Artefatos produzidos pela Fase 1."""

    train: pd.DataFrame
    test: pd.DataFrame | None
    resumo: pd.DataFrame
    integridade: pd.DataFrame
    alinhamento: pd.DataFrame
    estabilidade: pd.DataFrame
    figuras: dict[str, str]
    markdown: str


def carregar(caminho: str | Path, **kwargs: object) -> pd.DataFrame:
    """Le um CSV e devolve o DataFrame bruto."""
    return pd.read_csv(caminho, **kwargs)


def _resumo_datasets(train: pd.DataFrame, test: pd.DataFrame | None) -> pd.DataFrame:
    linhas = []
    for nome, df in [("train", train), ("test", test)]:
        if df is None:
            continue
        linhas.append(
            {
                "dataset": nome,
                "linhas": len(df),
                "colunas": df.shape[1],
                "memoria_MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
                "col_numericas": df.select_dtypes(include="number").shape[1],
                "col_texto": df.select_dtypes(include=["object", "category"]).shape[1],
                "celulas_nan_pct": round(100 * df.isna().to_numpy().mean(), 3),
                "linhas_com_nan_pct": round(100 * df.isna().any(axis=1).mean(), 3),
            }
        )
    return pd.DataFrame(linhas)


def _integridade(
    train: pd.DataFrame, test: pd.DataFrame | None, cfg: EDAConfig
) -> pd.DataFrame:
    checagens: list[dict[str, object]] = []

    def add(item: str, valor: object, ok: bool, obs: str = "") -> None:
        checagens.append({"checagem": item, "valor": valor, "status": "OK" if ok else "ATENCAO", "observacao": obs})

    alvo_ok = cfg.target in train.columns
    add("alvo presente no train", cfg.target, alvo_ok, "" if alvo_ok else "coluna alvo ausente")

    if alvo_ok:
        n_nan = int(train[cfg.target].isna().sum())
        add("NaN no alvo", n_nan, n_nan == 0, "linhas sem alvo devem ser descartadas" if n_nan else "")

    if test is not None:
        vazou = cfg.target in test.columns
        add("alvo ausente no test", not vazou, not vazou, "alvo presente no test indica vazamento" if vazou else "")

    if cfg.id_col and cfg.id_col in train.columns:
        dup = int(train[cfg.id_col].duplicated().sum())
        add(f"{cfg.id_col} unico no train", dup, dup == 0, "ids repetidos" if dup else "")
        if test is not None and cfg.id_col in test.columns:
            sobrep = len(set(train[cfg.id_col]) & set(test[cfg.id_col]))
            add("sobreposicao de id train/test", sobrep, sobrep == 0, "mesmos ids nos dois conjuntos" if sobrep else "")

    cols = [c for c in train.columns if c != cfg.id_col]
    dup_linhas = int(train.duplicated(subset=cols).sum())
    add("linhas duplicadas no train (sem id)", dup_linhas, dup_linhas == 0,
        "duplicatas inflam a validacao" if dup_linhas else "")

    return pd.DataFrame(checagens)


def _alinhamento(train: pd.DataFrame, test: pd.DataFrame | None, cfg: EDAConfig) -> pd.DataFrame:
    if test is None:
        return pd.DataFrame()

    so_train = [c for c in train.columns if c not in test.columns]
    so_test = [c for c in test.columns if c not in train.columns]
    comuns = [c for c in train.columns if c in test.columns]

    tipos_divergentes = [
        {"coluna": c, "dtype_train": str(train[c].dtype), "dtype_test": str(test[c].dtype)}
        for c in comuns
        if train[c].dtype != test[c].dtype
    ]

    linhas: list[dict[str, object]] = [
        {"item": "colunas comuns", "quantidade": len(comuns), "colunas": ""},
        {"item": "so no train", "quantidade": len(so_train), "colunas": ", ".join(so_train)},
        {"item": "so no test", "quantidade": len(so_test), "colunas": ", ".join(so_test)},
        {
            "item": "dtypes divergentes",
            "quantidade": len(tipos_divergentes),
            "colunas": ", ".join(d["coluna"] for d in tipos_divergentes),
        },
    ]
    return pd.DataFrame(linhas)


def _estabilidade(
    train: pd.DataFrame, test: pd.DataFrame | None, cfg: EDAConfig
) -> pd.DataFrame:
    """Compara train e test coluna a coluna (PSI + NaN + categorias novas)."""
    if test is None:
        return pd.DataFrame()

    comuns = [c for c in train.columns if c in test.columns and c not in {cfg.id_col, cfg.target}]
    linhas = []
    for c in comuns:
        valor_psi = psi(train[c], test[c])
        categorias_novas = ""
        fora_do_intervalo = 0
        max_train = max_test = float("nan")

        if pd.api.types.is_numeric_dtype(train[c]):
            # O PSI e baseado em quantis e nao enxerga valores extremos raros;
            # a contagem abaixo cobre essa cegueira.
            min_train, max_train = float(train[c].min()), float(train[c].max())
            max_test = float(test[c].max())
            fora_do_intervalo = int(((test[c] < min_train) | (test[c] > max_train)).sum())
        else:
            novas = set(test[c].dropna().unique()) - set(train[c].dropna().unique())
            categorias_novas = ", ".join(map(str, sorted(novas)[:5]))

        linhas.append(
            {
                "coluna": c,
                "nan_pct_train": round(100 * train[c].isna().mean(), 3),
                "nan_pct_test": round(100 * test[c].isna().mean(), 3),
                "psi": round(valor_psi, 4) if pd.notna(valor_psi) else float("nan"),
                "diagnostico": _classificar_psi(valor_psi, cfg),
                "max_train": max_train,
                "max_test": max_test,
                "test_fora_do_intervalo_train": fora_do_intervalo,
                "categorias_novas_no_test": categorias_novas,
            }
        )
    return pd.DataFrame(linhas).sort_values("psi", ascending=False, na_position="last")


def _classificar_psi(valor: float, cfg: EDAConfig) -> str:
    if pd.isna(valor):
        return "nao avaliado"
    if valor < cfg.limite_psi:
        return "estavel"
    if valor < 0.25:
        return "atencao"
    return "deslocamento forte"


def _figura_train_vs_test(
    train: pd.DataFrame, test: pd.DataFrame | None, cfg: EDAConfig
) -> dict[str, str]:
    """Sobrepoe as distribuicoes de train e test para cada coluna comum."""
    if test is None:
        return {}

    colunas = classificar_colunas(train, cfg)
    alvo_numericas = [c for c in colunas.numericas if c in test.columns]
    if not alvo_numericas:
        return {}

    fig, axes = grade_de_paineis(len(alvo_numericas), cfg)
    for ax, c in zip(axes, alvo_numericas):
        serie_train, serie_test = train[c].dropna(), test[c].dropna()
        juntas = pd.concat([serie_train, serie_test])

        # Um unico valor absurdo comprime todo o histograma; a janela robusta
        # mantem a comparacao legivel e o titulo informa quantos ficaram fora.
        inf, sup = juntas.quantile(0.001), juntas.quantile(0.999)
        margem = (sup - inf) * 0.05 or 1.0
        inf, sup = inf - margem, sup + margem
        n_fora = int(((juntas < inf) | (juntas > sup)).sum())

        faixas = np.linspace(inf, sup, 51)
        ax.hist(serie_train.clip(inf, sup), bins=faixas, density=True, color=SERIE_1,
                alpha=0.55, label="train", edgecolor="none")
        ax.hist(serie_test.clip(inf, sup), bins=faixas, density=True, histtype="step",
                color=SERIE_2, linewidth=2.0, label="test")
        sufixo = f"  ({n_fora} fora da janela p0.1-p99.9)" if n_fora else ""
        ax.set_title(f"{c}{sufixo}")
        ax.set_ylabel("densidade")
        ax.legend()
    fig.suptitle("Fase 1 - Distribuicoes: train vs test",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return {"train_vs_test": salvar_figura(fig, "f1_train_vs_test", cfg)}


def executar(
    caminho_train: str | Path,
    caminho_test: str | Path | None,
    cfg: EDAConfig,
) -> ResultadoFase1:
    """Roda a Fase 1 completa e grava tabelas/figuras em `cfg.outdir`."""
    cfg.preparar()

    train = carregar(caminho_train)
    test = carregar(caminho_test) if caminho_test else None

    resumo = _resumo_datasets(train, test)
    integridade = _integridade(train, test, cfg)
    alinhamento = _alinhamento(train, test, cfg)
    estabilidade = _estabilidade(train, test, cfg)

    salvar_tabela(resumo, "f1_resumo_datasets", cfg)
    salvar_tabela(integridade, "f1_integridade", cfg)
    if not alinhamento.empty:
        salvar_tabela(alinhamento, "f1_alinhamento_colunas", cfg)
    if not estabilidade.empty:
        salvar_tabela(estabilidade, "f1_estabilidade_train_test", cfg)

    figuras = _figura_train_vs_test(train, test, cfg)

    partes = [
        "## Fase 1 - Carregamento de dados\n",
        f"Origem: `{caminho_train}`" + (f" e `{caminho_test}`" if caminho_test else "") + "\n",
        "### 1.1 Dimensoes e memoria\n",
        df_para_markdown(resumo),
        "\n### 1.2 Integridade\n",
        df_para_markdown(integridade),
    ]
    if not alinhamento.empty:
        partes += ["\n### 1.3 Alinhamento de colunas train/test\n", df_para_markdown(alinhamento)]
    if not estabilidade.empty:
        partes += [
            "\n### 1.4 Estabilidade das distribuicoes (PSI)\n",
            "PSI < 0.10 estavel | 0.10-0.25 atencao | > 0.25 deslocamento forte.\n",
            df_para_markdown(estabilidade),
        ]
        extrapolam = estabilidade[estabilidade["test_fora_do_intervalo_train"] > 0]
        if not extrapolam.empty:
            partes.append(
                "\n**Atencao:** o test contem valores fora do intervalo observado no train "
                "(o PSI, por ser baseado em quantis, nao sinaliza esses casos):\n"
            )
            for _, r in extrapolam.iterrows():
                partes.append(
                    f"- `{r['coluna']}`: {int(r['test_fora_do_intervalo_train'])} linha(s); "
                    f"max train {r['max_train']:,.2f} vs max test {r['max_test']:,.2f}"
                )
    for titulo, caminho in figuras.items():
        partes.append(f"\n![{titulo}]({caminho})\n")

    return ResultadoFase1(
        train=train,
        test=test,
        resumo=resumo,
        integridade=integridade,
        alinhamento=alinhamento,
        estabilidade=estabilidade,
        figuras=figuras,
        markdown="\n".join(partes),
    )
