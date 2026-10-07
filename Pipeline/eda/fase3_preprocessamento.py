"""Engenharia de features (2.7) + Fase 3 do diagrama, ate antes da selecao.

Ordem de execucao (e do relatorio):

    3.1  Descarte        - NaN na coluna > 50%, NaN na linha > 25% e regras de
                           consistencia com o alvo; linhas so saem do train
    3.2  Outliers        - correcao de dominio + IQR/winsor/z-score (fase3_outliers)
    2.7  Avaliacao das features candidatas (sobre o train ja corrigido na 3.2)
    3.3  Imputacao       - mediana / moda / KNN + flags de ausencia
    3.4  Materializacao  - as features da 2.7, calculadas sobre colunas limpas
    3.5  Encoding        - ordinal / onehot / target
    3.6  Escalonamento   - StandardScaler / MinMaxScaler por modelo

Unica troca em relacao ao diagrama: a imputacao (ultima caixa do bloco de NaN)
roda depois dos outliers - os motivos estao em `fase3_ausentes`. A selecao de
features (VIF, importancia, correlacao > 0.9) fica para a proxima etapa.

`id` e alvo nunca passam pelas transformacoes: sao separados no inicio e
recolocados no fim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import fase2_features, fase3_ausentes, fase3_encoding, fase3_escalonamento, fase3_outliers
from .config import EDAConfig
from .fase2_features import EspecFeatures, ResultadoFase2Features
from .utils import classificar_colunas, df_para_markdown, salvar_tabela


@dataclass
class EspecPreprocessamento:
    """Tudo o que e especifico do dataset na Fase 3 (o runner preenche)."""

    features: EspecFeatures = field(default_factory=EspecFeatures)
    encoding: dict[str, dict[str, object]] = field(default_factory=dict)
    imputacao: dict[str, str] = field(default_factory=dict)
    # Condicoes `DataFrame.query` que removem linhas do train (podem usar o alvo).
    regras_descarte_linhas: list[str] = field(default_factory=list)
    metodo_outliers: dict[str, str] = field(default_factory=dict)
    regras_dominio: dict[str, dict[str, object]] = field(default_factory=dict)
    k_iqr: float = 1.5
    p_winsor: tuple[float, float] = (0.01, 0.99)
    limiar_zscore: float = 3.0
    vizinhos_knn: int = 5
    amostra_knn: int = 20_000
    experimento_imputacao: bool = True
    manter_escalonados: bool = False
    # Observacao do dataset acrescentada ao relatorio da 2.7.
    nota_features: str = ""


@dataclass
class ResultadoFase3:
    """Dados prontos para a selecao de features + artefatos de cada bloco."""

    train: pd.DataFrame
    test: pd.DataFrame | None
    colunas_features: list[str]
    features: ResultadoFase2Features
    descarte: fase3_ausentes.ResultadoDescarte
    descarte_regras: pd.DataFrame
    outliers: fase3_outliers.ResultadoFase3Outliers
    imputacao: fase3_ausentes.ResultadoImputacao
    materializacao: pd.DataFrame
    encoding: fase3_encoding.ResultadoEncoding
    escalonamento: fase3_escalonamento.ResultadoEscalonamento
    avisos: list[str]
    markdown: str = ""


def _descartar_por_regras(
    train: pd.DataFrame, regras: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove do train as linhas que satisfazem alguma regra (condicao invalida)."""
    remover = pd.Series(False, index=train.index)
    linhas = []
    for regra in regras:
        try:
            alvo = train.eval(regra).fillna(False).astype(bool)
        except Exception as exc:  # regra invalida nao derruba a Fase 3
            linhas.append({"regra": regra, "linhas_removidas": 0, "pct": 0.0,
                           "observacao": f"nao avaliada: {exc}"})
            continue
        novas = alvo & ~remover
        remover |= alvo
        linhas.append({"regra": regra, "linhas_removidas": int(novas.sum()),
                       "pct": round(100 * float(novas.mean()), 4), "observacao": ""})
    return train.loc[~remover], pd.DataFrame(linhas)


def _materializar(
    train: pd.DataFrame, test: pd.DataFrame | None, y: pd.Series, espec: EspecFeatures
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Cria as features da 2.7 sobre colunas ja limpas e imputadas.

    Uma feature que saia com NaN (ex.: titulo sem numero no test) e preenchida
    com a mediana do train e contada na tabela - nenhum NaN segue adiante.
    """
    train_f = fase2_features.criar_features(train, espec)
    test_f = fase2_features.criar_features(test, espec) if test is not None else None
    linhas = []
    for nome in espec.nomes:
        mediana = train_f[nome].median()
        n_train, n_test = int(train_f[nome].isna().sum()), int(test_f[nome].isna().sum()) if test_f is not None else 0
        if n_train:
            train_f[nome] = train_f[nome].fillna(mediana)
        if n_test:
            test_f[nome] = test_f[nome].fillna(mediana)
        linhas.append({
            "feature": nome,
            "pearson_alvo": round(float(train_f[nome].corr(y)), 5),
            "media_train": round(float(train_f[nome].mean()), 4),
            "nan_preenchidos_train": n_train,
            "nan_preenchidos_test": n_test,
        })
    return train_f, test_f, pd.DataFrame(linhas)


def _avisos_de_redundancia(espec: EspecPreprocessamento, regras_enc: dict[str, dict[str, object]]) -> list[str]:
    """Redundancias exatas criadas por construcao (insumo para a selecao)."""
    avisos = []
    for nome, (coluna, valores) in espec.features.indicadoras.items():
        if regras_enc.get(coluna, {}).get("metodo") == "onehot":
            avisos.append(
                f"`{nome}` e combinacao linear exata das dummies de `{coluna}` "
                f"(soma das dummies de {', '.join(valores)}): VIF infinito na selecao. "
                f"Manter `{nome}` ou as dummies de `{coluna}`, nao os dois."
            )
    for nome, coluna in espec.features.numeros_em_texto.items():
        if regras_enc.get(coluna, {}).get("metodo") not in (None, "descartar"):
            avisos.append(f"`{coluna}` foi codificada e tambem gerou `{nome}`: informacao duplicada.")
    return avisos


def executar(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cfg: EDAConfig,
    espec: EspecPreprocessamento,
) -> ResultadoFase3:
    """Roda 3.1 -> 3.6 (com a avaliacao 2.7) e devolve os dados prontos para a selecao."""
    if cfg.target not in train.columns:
        raise KeyError(f"coluna alvo '{cfg.target}' nao existe no train")
    cfg.preparar()
    fora = {cfg.target, cfg.id_col}
    colunas = [c for c in train.columns if c not in fora]

    # 3.1 - descarte (colunas, linhas por NaN, linhas por regra de consistencia)
    train, test, descarte = fase3_ausentes.descartar(train, test, cfg, colunas)
    train, descarte_regras = _descartar_por_regras(train, espec.regras_descarte_linhas)
    salvar_tabela(descarte.por_coluna, "f3_1_descarte_colunas", cfg)
    salvar_tabela(descarte.por_linha, "f3_1_descarte_linhas", cfg)
    if not descarte_regras.empty:
        salvar_tabela(descarte_regras, "f3_1_descarte_regras", cfg)

    y = train[cfg.target]
    ids_train = train[[cfg.id_col]] if cfg.id_col in train.columns else None
    ids_test = test[[cfg.id_col]] if test is not None and cfg.id_col in test.columns else None
    restantes = [c for c in colunas if c not in descarte.colunas_descartadas]
    X_train = train[restantes]
    X_test = test[restantes] if test is not None else None
    preditoras_originais = classificar_colunas(X_train, cfg).numericas

    # 3.2 - outliers (modulo da equipe: dominio + limites ajustados no train)
    outliers = fase3_outliers.executar(
        X_train, X_test, cfg,
        metodo_por_coluna=espec.metodo_outliers,
        regras_dominio=espec.regras_dominio,
        k_iqr=espec.k_iqr, p_winsor=espec.p_winsor, limiar_zscore=espec.limiar_zscore,
    )
    X_train, X_test = outliers.train, outliers.test

    # 2.7 - avaliacao das candidatas sobre o train ja sem erros de dominio
    features = fase2_features.executar(
        X_train.assign(**{cfg.target: y}), cfg, espec.features, preditoras_originais,
        origem_dos_dados=(
            "train apos o descarte (3.1) e o tratamento de outliers (3.2), antes da "
            "imputacao, casos completos - valores impossiveis distorceriam a "
            "avaliacao. " + espec.nota_features
        ).strip(),
    )

    # 3.3 - imputacao
    X_train, X_test, imputacao = fase3_ausentes.imputar(
        X_train, X_test, cfg, estrategias=espec.imputacao,
        vizinhos=espec.vizinhos_knn, amostra_knn=espec.amostra_knn,
        rodar_experimento=espec.experimento_imputacao,
    )

    # 3.4 - materializacao das features
    X_train, X_test, materializacao = _materializar(X_train, X_test, y, espec.features)
    salvar_tabela(materializacao, "f3_4_features_materializadas", cfg)

    # 3.5 - encoding
    X_train, X_test, encoding = fase3_encoding.executar(X_train, y, X_test, cfg, espec.encoding)
    avisos = _avisos_de_redundancia(espec, encoding.regras)

    # 3.6 - escalonamento
    escalonamento = fase3_escalonamento.executar(X_train, X_test, cfg, espec.manter_escalonados)

    colunas_features = list(X_train.columns)
    partes_train = [p for p in (ids_train, X_train, y.to_frame()) if p is not None]
    train_final = pd.concat(partes_train, axis=1).reset_index(drop=True)
    test_final = None
    if X_test is not None:
        test_final = pd.concat([p for p in (ids_test, X_test) if p is not None], axis=1).reset_index(drop=True)

    resumo_saida = pd.DataFrame([
        {"dataset": "train", "linhas": len(train_final), "features": len(colunas_features),
         "nan": int(X_train.isna().sum().sum())},
        *([{"dataset": "test", "linhas": len(test_final), "features": len(colunas_features),
            "nan": int(X_test.isna().sum().sum())}] if test_final is not None else []),
    ])
    salvar_tabela(resumo_saida, "f3_saida", cfg)

    resultado = ResultadoFase3(
        train=train_final, test=test_final, colunas_features=colunas_features,
        features=features, descarte=descarte, descarte_regras=descarte_regras,
        outliers=outliers, imputacao=imputacao, materializacao=materializacao,
        encoding=encoding, escalonamento=escalonamento, avisos=avisos,
    )
    resultado.markdown = _montar_markdown(resultado, resumo_saida, cfg)
    return resultado


def _montar_markdown(r: ResultadoFase3, resumo_saida: pd.DataFrame, cfg: EDAConfig) -> str:
    roteiro = pd.DataFrame([
        {"bloco do diagrama": "Engenharia de features (EDA)", "secao": "2.7 e 3.4",
         "observacao": "avaliada na 2.7, construida na 3.4 sobre colunas ja limpas"},
        {"bloco do diagrama": "NaN: drop coluna > 50% / drop linha > 25%", "secao": "3.1",
         "observacao": "linhas so saem do train"},
        {"bloco do diagrama": "Tratamento de outliers", "secao": "3.2",
         "observacao": "`fase3_outliers.py` (dominio + IQR/winsor/z-score)"},
        {"bloco do diagrama": "Mediana num. · moda cat. · KNN Imputer", "secao": "3.3",
         "observacao": "depois da 3.2: preenche tambem os NaN criados pela correcao de dominio"},
        {"bloco do diagrama": "Encoding de variaveis categoricas", "secao": "3.5", "observacao": ""},
        {"bloco do diagrama": "Normalizacao / escalonamento", "secao": "3.6",
         "observacao": "escalonador escolhido por modelo, aplicado na Fase 4"},
        {"bloco do diagrama": "Selecao de features", "secao": "-",
         "observacao": "proxima etapa (fora deste escopo)"},
    ])

    partes = [
        "## Roteiro: diagrama x execucao\n",
        df_para_markdown(roteiro),
        "",
        r.features.markdown,
        "\n## Fase 3.1 - Descarte de colunas e linhas\n",
        fase3_ausentes.markdown_descarte(r.descarte, cfg),
    ]
    if not r.descarte_regras.empty:
        partes += [
            "\n### 3.1.3 Linhas do train que violam regras de consistencia\n",
            "Condicoes impossiveis envolvendo o alvo, avaliadas nos valores brutos. "
            "Como dependem do alvo, valem so para o train.\n",
            df_para_markdown(r.descarte_regras),
        ]
    partes += [
        "",
        r.outliers.markdown,
        "\n## Fase 3.3 - Imputacao (mediana · moda · KNN)\n",
        fase3_ausentes.markdown_imputacao(r.imputacao),
        "\n## Fase 3.4 - Materializacao das features da 2.7\n",
        "Calculadas depois da imputacao: a interacao usa a duracao ja imputada e "
        "recortada. `pearson_alvo` medido no train final.\n",
        df_para_markdown(r.materializacao),
        "\n## Fase 3.5 - Encoding de variaveis categoricas\n",
        fase3_encoding.markdown_encoding(r.encoding),
        "\n## Fase 3.6 - Normalizacao / escalonamento\n",
        fase3_escalonamento.markdown_escalonamento(r.escalonamento),
        "\n## Saida - entrada da selecao de features\n",
        df_para_markdown(resumo_saida),
        f"\n**{len(r.colunas_features)} features:** "
        + ", ".join(f"`{c}`" for c in r.colunas_features) + "\n",
        "Os arquivos gravados estao **sem escala** (servem direto para as arvores); "
        "os modelos que precisam de escala a recebem pelo `escalonador_para` (3.6).\n",
    ]
    if r.avisos:
        partes.append("**Redundancias por construcao - a selecao de features (VIF) vai aponta-las:**\n")
        partes += [f"- {a}" for a in r.avisos]
    return "\n".join(partes)
