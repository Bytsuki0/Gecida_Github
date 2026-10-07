"""Fase 3 - Tratamento de valores ausentes (NaN).

Bloco do diagrama de desenvolvimento:

    NaN na coluna > 50%?  sim -> drop coluna
    NaN na linha  > 25%?  sim -> drop linha
    senao                 -> mediana (num.) · moda (cat.) · KNN Imputer

O bloco e executado em dois momentos da Fase 3:

    3.1  descarte   - colunas e linhas removidas, antes de qualquer tratamento
    3.3  imputacao  - preenche o que sobrou, DEPOIS dos outliers (3.2)

A imputacao vem depois dos outliers por tres motivos: a correcao de dominio da
3.2 transforma valores impossiveis em NaN, que tambem precisam ser imputados;
os limites de outlier devem ser calculados so sobre valores observados (uma
massa de valores imputados na mediana estreitaria o IQR); e o KNN mede
distancias, que valores extremos distorcem.

Regras de vazamento: tudo e decidido/ajustado no train. O test nunca perde
linhas (o submission precisa de uma previsao por id) e e imputado com os
valores aprendidos no train. O alvo nunca entra na imputacao: ele nao existe
no test, entao imputar o train com ele criaria uma relacao que o test nao tem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .config import EDAConfig
from .estilo import SERIE_1, SERIE_2, TINTA_SECUNDARIA
from .utils import df_para_markdown, grade_de_paineis, salvar_figura, salvar_tabela

ESTRATEGIAS = ("mediana", "media", "moda", "knn")


# ---------------------------------------------------------------------------
# 3.1 Descarte de colunas e linhas
# ---------------------------------------------------------------------------
@dataclass
class ResultadoDescarte:
    """Decisoes da arvore de descarte do diagrama."""

    colunas_descartadas: list[str]
    linhas_descartadas: int
    por_coluna: pd.DataFrame
    por_linha: pd.DataFrame


def descartar(
    train: pd.DataFrame, test: pd.DataFrame | None, cfg: EDAConfig, colunas: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame | None, ResultadoDescarte]:
    """Aplica a arvore de decisao do diagrama sobre as `colunas` de features.

    A regra da coluna vem primeiro (como no diagrama); a fracao de NaN por linha
    e medida sobre as colunas que sobreviveram. Linhas so saem do train.
    """
    nan_train = 100 * train[colunas].isna().mean()
    nan_test = 100 * test[colunas].isna().mean() if test is not None else None
    colunas_fora = [c for c in colunas if nan_train[c] > cfg.limite_nan_coluna]

    linhas_col = []
    for c in colunas:
        if c in colunas_fora:
            decisao = f"descartar (> {cfg.limite_nan_coluna:g}% NaN)"
        elif nan_train[c] > 0 or (nan_test is not None and nan_test[c] > 0):
            decisao = "manter - imputar na 3.3"
        else:
            decisao = "manter - completa"
        linhas_col.append({
            "coluna": c,
            "nan_pct_train": round(float(nan_train[c]), 4),
            "nan_pct_test": round(float(nan_test[c]), 4) if nan_test is not None else np.nan,
            "decisao": decisao,
        })

    restantes = [c for c in colunas if c not in colunas_fora]
    n_nan_train = train[restantes].isna().sum(axis=1)
    acima = 100 * n_nan_train / max(len(restantes), 1) > cfg.limite_nan_linha
    n_nan_test = test[restantes].isna().sum(axis=1) if test is not None else None

    linhas_lin = []
    for k in range(int(max(n_nan_train.max(), n_nan_test.max() if n_nan_test is not None else 0)) + 1):
        pct = 100 * k / max(len(restantes), 1)
        excede = pct > cfg.limite_nan_linha
        n_tr = int((n_nan_train == k).sum())
        n_te = int((n_nan_test == k).sum()) if n_nan_test is not None else 0
        if n_tr == 0 and n_te == 0:
            continue
        linhas_lin.append({
            "nan_na_linha": k,
            "pct_das_features": round(pct, 1),
            "linhas_train": n_tr,
            "linhas_test": n_te,
            "acao_train": "descartar" if excede else "manter",
            "acao_test": "manter e imputar" if k else "manter",
        })

    train_out = train.loc[~acima].drop(columns=colunas_fora)
    test_out = test.drop(columns=colunas_fora) if test is not None else None
    resultado = ResultadoDescarte(
        colunas_descartadas=colunas_fora,
        linhas_descartadas=int(acima.sum()),
        por_coluna=pd.DataFrame(linhas_col),
        por_linha=pd.DataFrame(linhas_lin),
    )
    return train_out, test_out, resultado


# ---------------------------------------------------------------------------
# 3.3 Imputacao
# ---------------------------------------------------------------------------
class Imputador(BaseEstimator, TransformerMixin):
    """Mediana / media / moda / KNN por coluna, com flag de ausencia.

    Compativel com `sklearn.pipeline.Pipeline` (entra e sai DataFrame).

    Args:
        estrategias: coluna -> "mediana" | "media" | "moda" | "knn". Colunas nao
            listadas usam mediana (numericas) ou moda (demais).
        limite_flag: colunas com ao menos esse % de NaN no fit ganham a coluna
            `<coluna>_missing` (1 = valor ausente/invalido na origem).
        vizinhos: k do KNN Imputer.
        amostra_knn: linhas do train usadas como doadoras do KNN. O KNN compara
            cada linha com todas as doadoras; com 750k doadoras a imputacao
            levaria horas.
    """

    def __init__(
        self,
        estrategias: dict[str, str] | None = None,
        limite_flag: float = 1.0,
        vizinhos: int = 5,
        amostra_knn: int = 20_000,
        random_state: int = 42,
    ) -> None:
        self.estrategias = estrategias
        self.limite_flag = limite_flag
        self.vizinhos = vizinhos
        self.amostra_knn = amostra_knn
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: object = None) -> Imputador:
        pedidas = dict(self.estrategias or {})
        invalidas = {c: e for c, e in pedidas.items() if e not in ESTRATEGIAS}
        if invalidas:
            raise ValueError(f"estrategia desconhecida: {invalidas} (use {', '.join(ESTRATEGIAS)})")

        self.estrategia_por_coluna_: dict[str, str] = {}
        self.valores_: dict[str, object] = {}
        for c in X.columns:
            numerica = pd.api.types.is_numeric_dtype(X[c])
            estrategia = pedidas.get(c, "mediana" if numerica else "moda")
            if estrategia in ("mediana", "media", "knn") and not numerica:
                raise ValueError(f"'{estrategia}' exige coluna numerica; '{c}' nao e")
            self.estrategia_por_coluna_[c] = estrategia
            if estrategia == "mediana":
                self.valores_[c] = X[c].median()
            elif estrategia == "media":
                self.valores_[c] = X[c].mean()
            elif estrategia == "moda":
                moda = X[c].mode(dropna=True)
                self.valores_[c] = moda.iloc[0] if not moda.empty else np.nan

        self.colunas_knn_ = [c for c, e in self.estrategia_por_coluna_.items() if e == "knn"]
        if self.colunas_knn_:
            # Distancias so fazem sentido em escala comum: sem padronizar, uma
            # coluna em minutos (0-300) apagaria uma contagem (0-3).
            self.base_knn_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            doadoras = X[self.base_knn_].sample(
                n=min(self.amostra_knn, len(X)), random_state=self.random_state
            )
            self.escala_knn_ = StandardScaler().fit(doadoras)
            self.knn_ = KNNImputer(n_neighbors=self.vizinhos, keep_empty_features=True).fit(
                self.escala_knn_.transform(doadoras)
            )

        self.flags_ = [c for c in X.columns if 100 * X[c].isna().mean() >= self.limite_flag]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        saida = X.copy()
        for c in self.flags_:
            saida[f"{c}_missing"] = X[c].isna().astype("int8")

        if self.colunas_knn_:
            linhas = X[self.colunas_knn_].isna().any(axis=1)
            if linhas.any():
                base = X.loc[linhas, self.base_knn_]
                preenchido = pd.DataFrame(
                    self.escala_knn_.inverse_transform(
                        self.knn_.transform(self.escala_knn_.transform(base))
                    ),
                    index=base.index,
                    columns=self.base_knn_,
                )
                for c in self.colunas_knn_:
                    saida.loc[linhas, c] = base[c].fillna(preenchido[c])

        for c, valor in self.valores_.items():
            if saida[c].isna().any():
                saida[c] = saida[c].fillna(valor)
        return saida


def experimento_imputacao(
    X: pd.DataFrame,
    colunas: list[str],
    cfg: EDAConfig,
    n_mascarados: int = 2_000,
    n_doadores: int = 20_000,
    vizinhos: tuple[int, ...] = (5, 50),
) -> pd.DataFrame:
    """Compara mediana, media e KNN escondendo valores que se conhece.

    Para cada coluna: sorteia `n_mascarados` linhas com valor observado, apaga
    esse valor, imputa usando `n_doadores` outras linhas e mede o erro contra o
    valor verdadeiro. `rmse / desvio` ~ 1 significa que o metodo nao acerta
    mais do que chutar a media - as outras colunas nao preveem esta.
    """
    rng = np.random.default_rng(cfg.random_state)
    numericas = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    linhas: list[dict[str, object]] = []

    for coluna in colunas:
        observadas = X.index[X[coluna].notna()].to_numpy()
        if len(observadas) < 10:
            continue
        mascaradas = rng.choice(observadas, size=min(n_mascarados, len(observadas) // 2), replace=False)
        resto = np.setdiff1d(observadas, mascaradas)
        doadoras = rng.choice(resto, size=min(n_doadores, len(resto)), replace=False)
        verdade = X.loc[mascaradas, coluna].to_numpy(dtype=float)
        desvio = float(verdade.std())

        palpites: dict[str, np.ndarray] = {
            "mediana": np.full(len(verdade), X.loc[doadoras, coluna].median()),
            "media": np.full(len(verdade), X.loc[doadoras, coluna].mean()),
        }
        escala = StandardScaler().fit(X.loc[doadoras, numericas])
        consulta = X.loc[mascaradas, numericas].copy()
        consulta[coluna] = np.nan
        for k in vizinhos:
            knn = KNNImputer(n_neighbors=k, keep_empty_features=True).fit(
                escala.transform(X.loc[doadoras, numericas])
            )
            imputado = escala.inverse_transform(knn.transform(escala.transform(consulta)))
            palpites[f"knn (k={k})"] = imputado[:, numericas.index(coluna)]

        for metodo, palpite in palpites.items():
            erro = palpite - verdade
            rmse = float(np.sqrt(np.mean(erro**2)))
            linhas.append({
                "coluna": coluna,
                "metodo": metodo,
                "rmse": round(rmse, 3),
                "mae": round(float(np.mean(np.abs(erro))), 3),
                "rmse_sobre_desvio": round(rmse / desvio, 3) if desvio else np.nan,
                "n_mascarados": len(verdade),
                "n_doadores": len(doadoras),
            })
    return pd.DataFrame(linhas)


@dataclass
class ResultadoImputacao:
    """Artefatos da imputacao (3.3)."""

    imputador: Imputador
    por_coluna: pd.DataFrame
    experimento: pd.DataFrame
    figuras: dict[str, str] = field(default_factory=dict)


def _figura_imputacao(
    antes: pd.DataFrame, depois: pd.DataFrame, colunas: list[str], cfg: EDAConfig
) -> dict[str, str]:
    if not colunas:
        return {}
    fig, axes = grade_de_paineis(len(colunas), cfg)
    for ax, c in zip(axes, colunas):
        observados, completos = antes[c].dropna(), depois[c]
        faixas = np.linspace(completos.min(), completos.max(), 61)
        ax.hist(completos, bins=faixas, color=SERIE_2, alpha=0.55, edgecolor="none",
                label="apos imputacao")
        ax.hist(observados, bins=faixas, histtype="step", color=SERIE_1, linewidth=2.0,
                label="so observados")
        pct = 100 * antes[c].isna().mean()
        ax.set_title(f"{c}  ({pct:.1f}% imputado)")
        ax.set_ylabel("frequencia")
        ax.legend()
    fig.suptitle("Fase 3.3 - Distribuicao antes e depois da imputacao (train)",
                 fontsize=13, fontweight="semibold", color=TINTA_SECUNDARIA)
    fig.tight_layout()
    return {"imputacao": salvar_figura(fig, "f3_3_imputacao", cfg)}


def imputar(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cfg: EDAConfig,
    estrategias: dict[str, str] | None = None,
    vizinhos: int = 5,
    amostra_knn: int = 20_000,
    rodar_experimento: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, ResultadoImputacao]:
    """Ajusta o `Imputador` no train e aplica em train e test."""
    imputador = Imputador(
        estrategias=estrategias,
        limite_flag=cfg.limite_flag_ausencia,
        vizinhos=vizinhos,
        amostra_knn=amostra_knn,
        random_state=cfg.random_state,
    ).fit(train)

    train_imp = imputador.transform(train)
    test_imp = imputador.transform(test) if test is not None else None

    linhas = []
    for c, estrategia in imputador.estrategia_por_coluna_.items():
        n_train = int(train[c].isna().sum())
        n_test = int(test[c].isna().sum()) if test is not None else 0
        if not (n_train or n_test):
            continue
        valor = imputador.valores_.get(c)
        linhas.append({
            "coluna": c,
            "estrategia": estrategia,
            "valor_imputado": f"KNN k={vizinhos} ({amostra_knn:,} doadoras)".replace(",", ".")
            if estrategia == "knn" else (round(valor, 4) if isinstance(valor, float) else valor),
            "n_imputados_train": n_train,
            "n_imputados_test": n_test,
            "flag": f"{c}_missing" if c in imputador.flags_ else "-",
        })
    por_coluna = pd.DataFrame(linhas)

    numericas_com_nan = [
        c for c in train.columns
        if pd.api.types.is_numeric_dtype(train[c]) and train[c].isna().any()
    ]
    experimento = (experimento_imputacao(train, numericas_com_nan, cfg)
                   if rodar_experimento and numericas_com_nan else pd.DataFrame())

    restantes = int(train_imp.isna().sum().sum()) + (int(test_imp.isna().sum().sum()) if test_imp is not None else 0)
    if restantes:
        raise RuntimeError(f"{restantes} NaN sobreviveram a imputacao - revisar as estrategias")

    salvar_tabela(por_coluna, "f3_3_imputacao", cfg)
    if not experimento.empty:
        salvar_tabela(experimento, "f3_3_experimento_imputacao", cfg)
    com_flag = [c for c in numericas_com_nan if c in imputador.flags_]
    figuras = _figura_imputacao(train, train_imp, com_flag, cfg)

    return train_imp, test_imp, ResultadoImputacao(
        imputador=imputador, por_coluna=por_coluna, experimento=experimento, figuras=figuras
    )


def markdown_descarte(resultado: ResultadoDescarte, cfg: EDAConfig) -> str:
    partes = [
        f"### 3.1.1 Colunas: NaN na coluna > {cfg.limite_nan_coluna:g}%?\n",
        df_para_markdown(resultado.por_coluna),
        f"\n**Colunas descartadas:** {', '.join(resultado.colunas_descartadas) or 'nenhuma'}.\n",
        f"\n### 3.1.2 Linhas: NaN na linha > {cfg.limite_nan_linha:g}%?\n",
        "Fracao medida sobre as colunas de features que sobreviveram. So o train "
        "perde linhas: o test precisa de uma previsao por `id` no submission, entao "
        "suas linhas incompletas sao imputadas.\n",
        df_para_markdown(resultado.por_linha),
        f"\n**Linhas descartadas do train:** {resultado.linhas_descartadas:,}.\n".replace(",", "."),
    ]
    return "\n".join(partes)


def markdown_imputacao(resultado: ResultadoImputacao) -> str:
    partes = [
        "Valores aprendidos no train e aplicados ao test. A flag `<coluna>_missing` "
        "preserva a informacao de que o valor nao existia - o modelo pode tratar "
        "essas linhas de forma diferente.\n",
        df_para_markdown(resultado.por_coluna),
    ]
    if not resultado.experimento.empty:
        partes += [
            "\n**Mediana x media x KNN - valores conhecidos escondidos e reimputados.** "
            "`rmse_sobre_desvio` ~ 1 indica que o metodo nao acerta mais que a media: "
            "as outras colunas nao preveem esta.\n",
            df_para_markdown(resultado.experimento),
        ]
    if "imputacao" in resultado.figuras:
        partes.append(f"\n![imputacao]({resultado.figuras['imputacao']})\n")
    return "\n".join(partes)
