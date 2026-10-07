"""Fase 3.5 - Encoding de variaveis categoricas.

Bloco do diagrama: "Ordinal (Pub_Time) · OHE (Genre, Day)". Cada coluna
categorica recebe um metodo em `regras`:

    ordinal   - categorias com ordem natural; a ordem e sempre explicita
                (alfabetica poria Afternoon < Evening < Morning < Night)
    onehot    - nominais de baixa cardinalidade; a primeira categoria vira a
                referencia e e descartada, para que as dummies nao somem 1 em
                toda linha (colinearidade perfeita com o intercepto -> VIF
                infinito na selecao de features)
    target    - nominais de cardinalidade media/alta: media do alvo por
                categoria, suavizada e com cross-fitting no train (cada linha
                e codificada por folds que nao a contem, evitando vazamento)
    descartar - coluna substituida por uma feature derivada (Fase 2.7)

Colunas sem regra: onehot ate `cfg.max_cardinalidade_categorica` categorias,
target acima disso. Tudo e ajustado no train e aplicado ao test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from .config import EDAConfig
from .utils import df_para_markdown, salvar_tabela

METODOS = ("ordinal", "onehot", "target", "descartar")


@dataclass
class ResultadoEncoding:
    """Artefatos do encoding (3.5)."""

    codificador: ColumnTransformer
    regras: dict[str, dict[str, object]]
    tabela: pd.DataFrame


def resolver_regras(
    X: pd.DataFrame, regras: dict[str, dict[str, object]] | None, cfg: EDAConfig
) -> dict[str, dict[str, object]]:
    """Completa as regras para todas as colunas categoricas de `X`."""
    regras = dict(regras or {})
    resolvidas: dict[str, dict[str, object]] = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        if c in regras:
            regra = {**regras[c], "origem": "definida"}
        else:
            alta = X[c].nunique() > cfg.max_cardinalidade_categorica
            regra = {"metodo": "target" if alta else "onehot", "origem": "padrao (cardinalidade)"}
        if regra["metodo"] not in METODOS:
            raise ValueError(f"metodo de encoding desconhecido para '{c}': {regra['metodo']!r}")
        if regra["metodo"] == "ordinal" and not regra.get("categorias"):
            raise ValueError(f"'{c}': encoding ordinal exige 'categorias' na ordem desejada")
        resolvidas[c] = regra
    return resolvidas


def construir_codificador(regras: dict[str, dict[str, object]], cfg: EDAConfig) -> ColumnTransformer:
    """ColumnTransformer nao ajustado, um transformador por coluna.

    Colunas que nao estao em `regras` (numericas, flags, features) passam direto.
    """
    transformadores: list[tuple[str, object, list[str]]] = []
    for c, regra in regras.items():
        metodo = regra["metodo"]
        if metodo == "ordinal":
            enc = OrdinalEncoder(categories=[list(regra["categorias"])],
                                 handle_unknown="use_encoded_value", unknown_value=-1)
        elif metodo == "onehot":
            enc = OneHotEncoder(
                categories=[list(regra["categorias"])] if regra.get("categorias") else "auto",
                drop="first" if regra.get("drop_primeira", True) else None,
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.int8,
            )
        elif metodo == "target":
            enc = TargetEncoder(
                target_type="continuous",
                cv=KFold(n_splits=5, shuffle=True, random_state=cfg.random_state),
            )
        else:
            enc = "drop"
        transformadores.append((f"{metodo}_{c}", enc, [c]))

    return ColumnTransformer(
        transformadores, remainder="passthrough", verbose_feature_names_out=False
    ).set_output(transform="pandas")


def _detalhe(regra: dict[str, object], enc: object) -> str:
    metodo = regra["metodo"]
    if metodo == "ordinal":
        return " < ".join(map(str, regra["categorias"])) + " -> 0.." + str(len(regra["categorias"]) - 1)
    if metodo == "onehot":
        cats = list(enc.categories_[0])
        return f"referencia (descartada): {cats[0]}" if enc.drop is not None else "sem categoria de referencia"
    if metodo == "target":
        cod = enc.encodings_[0]
        return f"media suavizada do alvo, {cod.min():.2f} a {cod.max():.2f}; cross-fitting 5 folds"
    return str(regra.get("motivo", "descartada"))


def executar(
    train: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame | None,
    cfg: EDAConfig,
    regras: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, ResultadoEncoding]:
    """Ajusta o encoding no train (com `y`, pelo target encoding) e aplica no test.

    O train passa por `fit_transform`: e ele que ativa o cross-fitting do
    TargetEncoder. `fit` seguido de `transform` no proprio train codificaria
    cada linha com uma media que inclui o seu proprio alvo.
    """
    regras = resolver_regras(train, regras, cfg)
    codificador = construir_codificador(regras, cfg)
    train_enc = codificador.fit_transform(train, y)
    test_enc = codificador.transform(test) if test is not None else None

    # Mantem a ordem original (ordinal e target preservam o nome da coluna);
    # as dummies novas vao para o fim.
    ordem = [c for c in train.columns if c in train_enc.columns]
    ordem += [c for c in train_enc.columns if c not in ordem]
    train_enc = train_enc[ordem]
    test_enc = test_enc[ordem] if test_enc is not None else None

    nao_numericas = [c for c in train_enc.columns if not pd.api.types.is_numeric_dtype(train_enc[c])]
    if nao_numericas:
        raise RuntimeError(f"colunas ainda nao numericas apos o encoding: {nao_numericas}")

    linhas = []
    for c, regra in regras.items():
        enc = codificador.named_transformers_[f"{regra['metodo']}_{c}"]
        geradas = (list(enc.get_feature_names_out()) if regra["metodo"] != "descartar" else [])
        desconhecidas = 0
        if test is not None:
            desconhecidas = int((~test[c].isin(train[c].dropna().unique()) & test[c].notna()).sum())
        linhas.append({
            "coluna": c,
            "metodo": regra["metodo"],
            "regra": regra["origem"],
            "n_categorias": int(train[c].nunique()),
            "detalhe": _detalhe(regra, enc),
            "n_colunas_geradas": len(geradas),
            "colunas_geradas": ", ".join(geradas) if len(geradas) <= 4
            else f"{', '.join(geradas[:3])}, ... (+{len(geradas) - 3})",
            "linhas_test_categoria_nova": desconhecidas,
        })
    tabela = pd.DataFrame(linhas)
    salvar_tabela(tabela, "f3_5_encoding", cfg)

    return train_enc, test_enc, ResultadoEncoding(codificador=codificador, regras=regras, tabela=tabela)


def markdown_encoding(resultado: ResultadoEncoding) -> str:
    return "\n".join([
        "Ajustado no train, aplicado ao test. Categoria nunca vista no train vira "
        "`-1` (ordinal), zeros em todas as dummies (onehot) ou a media global do alvo "
        "(target) - a coluna `linhas_test_categoria_nova` mostra quantas linhas isso afetou.\n",
        df_para_markdown(resultado.tabela),
    ])
