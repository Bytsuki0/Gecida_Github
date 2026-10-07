"""Engenharia de features (2.7) + Fase 3 (pre-processamento) ate antes da selecao.

Exemplos
--------
Dataset do projeto (valores padrao ja apontam para ele):

    python Pipeline/run_preprocessamento.py

Trocando a estrategia de imputacao de uma coluna (mediana | media | moda | knn):

    python Pipeline/run_preprocessamento.py --imputacao Episode_Length_minutes=knn

Gravando tambem as versoes escalonadas (StandardScaler e MinMaxScaler):

    python Pipeline/run_preprocessamento.py --salvar-escalonados

Saidas: relatorio + figuras + tabelas em --outdir; train/test prontos para a
selecao de features em --dados (pasta ignorada pelo git - arquivos grandes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from Pipeline.eda import EDAConfig  # noqa: E402
from Pipeline.eda import fase1_carregamento, fase3_preprocessamento  # noqa: E402
from Pipeline.eda.estilo import aplicar_estilo  # noqa: E402
from Pipeline.eda.fase2_features import EspecFeatures  # noqa: E402
from Pipeline.eda.fase3_ausentes import ESTRATEGIAS  # noqa: E402
from Pipeline.run_outliers import METODOS_PADRAO_PODCAST, REGRAS_DOMINIO_PODCAST  # noqa: E402

ALVO_PODCAST = "Listening_Time_minutes"

# 2.7 - Is_Weekend e a interacao Length x Host_Pop vem do diagrama. Os demais
# candidatos vem da EDA: Episode_Title e "Episode N" (um numero disfarcado de
# texto) e Number_of_Ads / Guest_Popularity sao as outras numericas - a
# interacao com Ads e a que mais acrescenta R^2. A selecao decide quem fica.
FEATURES_PODCAST = EspecFeatures(
    indicadoras={"Is_Weekend": ("Publication_Day", ["Saturday", "Sunday"])},
    numeros_em_texto={"Episode_Number": "Episode_Title"},
    interacoes={
        "Length_x_Host_Pop": ("Episode_Length_minutes", "Host_Popularity_percentage"),
        "Length_x_Ads": ("Episode_Length_minutes", "Number_of_Ads"),
        "Length_x_Guest_Pop": ("Episode_Length_minutes", "Guest_Popularity_percentage"),
    },
)
NOTA_FEATURES_PODCAST = (
    "No train bruto, os 9 valores corrompidos de `Number_of_Ads` (ate 103) derrubam "
    "o ganho medido de `Length_x_Ads` de 0,0022 para 0,0013."
)

# 3.1 - a regra bruta "escuta > duracao" pega 2.568 linhas, mas 96,7% delas sao
# arredondamento de quem ouviu o episodio inteiro (EDA, secao 2.6). Com
# tolerancia de 5 min sobram as 85 com o alvo de fato impossivel.
REGRAS_DESCARTE_LINHAS_PODCAST = [
    "Listening_Time_minutes > Episode_Length_minutes + 5",
]

# 3.3 - vazio = mediana para todas as numericas (moda para categoricas). O
# experimento da 3.3 mostra que nenhuma coluna preve Episode_Length_minutes ou
# Guest_Popularity_percentage: o KNN com k=5 erra MAIS que a mediana e com k=50
# apenas empata. A flag `_missing` guarda o que a mediana apaga.
IMPUTACAO_PODCAST: dict[str, str] = {}

# 3.5 - Ordinal (Pub_Time) e OHE (Genre, Day) vem do diagrama.
ENCODING_PODCAST = {
    "Publication_Time": {"metodo": "ordinal", "categorias": ["Morning", "Afternoon", "Evening", "Night"]},
    "Episode_Sentiment": {"metodo": "ordinal", "categorias": ["Negative", "Neutral", "Positive"]},
    "Genre": {"metodo": "onehot"},
    # Ordem explicita so para a referencia descartada ser Monday, nao Friday.
    "Publication_Day": {"metodo": "onehot", "categorias": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
    # 48 nomes: OHE criaria 47 colunas para um eta2 de 0,0026 (EDA 2.4).
    "Podcast_Name": {"metodo": "target"},
    "Episode_Title": {"metodo": "descartar", "motivo": "substituida por Episode_Number (2.7)"},
}


def _pares(itens: list[str] | None, opcao: str) -> dict[str, str]:
    pares = {}
    for item in itens or []:
        if "=" not in item:
            print(f"[aviso] ignorando {opcao} mal formado: '{item}' (use COLUNA=VALOR)")
            continue
        coluna, valor = item.split("=", 1)
        pares[coluna] = valor
    return pares


def montar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Engenharia de features (2.7) + Fase 3 ate antes da selecao de features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train", default=str(RAIZ / "train.csv"), help="CSV de treino")
    p.add_argument("--test", default=str(RAIZ / "test.csv"), help="CSV de teste (opcional)")
    p.add_argument("--sem-test", action="store_true", help="rodar apenas com o train")
    p.add_argument("--target", default=ALVO_PODCAST, help="coluna alvo")
    p.add_argument("--id", dest="id_col", default="id", help="coluna identificadora")
    p.add_argument("--outdir", default=str(RAIZ / "outputs" / "eda"),
                   help="diretorio do relatorio, figuras e tabelas")
    p.add_argument("--dados", default=str(RAIZ / "outputs" / "dados"),
                   help="diretorio dos CSVs pre-processados")
    p.add_argument("--imputacao", action="append", metavar="COLUNA=ESTRATEGIA",
                   help=f"estrategia por coluna ({' | '.join(ESTRATEGIAS)}); repetivel")
    p.add_argument("--metodo", action="append", metavar="COLUNA=METODO",
                   help="metodo de outlier por coluna (iqr | winsor | zscore); repetivel")
    p.add_argument("--vizinhos-knn", type=int, default=5, help="k do KNN Imputer")
    p.add_argument("--amostra-knn", type=int, default=20_000, help="linhas doadoras do KNN Imputer")
    p.add_argument("--sem-experimento", action="store_true",
                   help="pular o experimento mediana x KNN da 3.3")
    p.add_argument("--salvar-escalonados", action="store_true",
                   help="gravar tambem train/test com StandardScaler e MinMaxScaler")
    p.add_argument("--sem-regras-padrao", action="store_true",
                   help="nao aplicar as regras de dominio/consistencia do dataset de podcasts")
    return p


def main(argv: list[str] | None = None) -> int:
    args = montar_parser().parse_args(argv)
    inicio = time.time()

    cfg = EDAConfig(target=args.target, id_col=args.id_col, outdir=Path(args.outdir))
    aplicar_estilo()
    cfg.preparar()

    podcast = args.target == ALVO_PODCAST
    regras = podcast and not args.sem_regras_padrao
    espec = fase3_preprocessamento.EspecPreprocessamento(
        features=FEATURES_PODCAST if podcast else EspecFeatures(),
        encoding=ENCODING_PODCAST if podcast else {},
        imputacao={**(IMPUTACAO_PODCAST if podcast else {}), **_pares(args.imputacao, "--imputacao")},
        regras_descarte_linhas=REGRAS_DESCARTE_LINHAS_PODCAST if regras else [],
        metodo_outliers={**(METODOS_PADRAO_PODCAST if podcast else {}), **_pares(args.metodo, "--metodo")},
        regras_dominio=REGRAS_DOMINIO_PODCAST if regras else {},
        vizinhos_knn=args.vizinhos_knn,
        amostra_knn=args.amostra_knn,
        experimento_imputacao=not args.sem_experimento,
        manter_escalonados=args.salvar_escalonados,
        nota_features=NOTA_FEATURES_PODCAST if podcast else "",
    )

    caminho_test = None
    if not args.sem_test:
        caminho_test = args.test if args.test and Path(args.test).exists() else None
        if args.test and caminho_test is None:
            print(f"[aviso] test nao encontrado em '{args.test}' - seguindo apenas com o train.")

    print("Carregando dados...")
    train = fase1_carregamento.carregar(args.train)
    test = fase1_carregamento.carregar(caminho_test) if caminho_test else None
    print(f"  train {train.shape}" + (f" | test {test.shape}" if test is not None else ""))

    print("Engenharia de features (2.7) + pre-processamento (3.1 a 3.6)...")
    r = fase3_preprocessamento.executar(train, test, cfg, espec)

    print(f"  3.1 colunas descartadas: {r.descarte.colunas_descartadas or 'nenhuma'} | "
          f"linhas (NaN): {r.descarte.linhas_descartadas} | "
          f"linhas (regras): {int(r.descarte_regras['linhas_removidas'].sum()) if not r.descarte_regras.empty else 0}")
    for _, linha in r.features.avaliacao.iterrows():
        ganho = linha.get("ganho_r2")
        extra = f" | ganho R2 {ganho:+.5f}" if pd.notna(ganho) else ""
        print(f"  2.7 {linha['feature']}: pearson {linha['pearson_alvo']:+.4f}{extra}")
    print(f"  3.5 encoding: {len(r.encoding.regras)} categoricas -> {len(r.colunas_features)} features")
    for aviso in r.avisos:
        print(f"  [selecao] {aviso}")

    relatorio = cfg.outdir / "relatorio_fase3_preprocessamento.md"
    relatorio.write_text(
        f"# Engenharia de features (2.7) e pre-processamento (Fase 3) | alvo: `{cfg.target}`\n\n"
        "Gerado por `Pipeline/run_preprocessamento.py`. Tabelas em `tabelas/`, figuras em "
        "`figuras/`. Para na entrada da selecao de features.\n\n" + r.markdown,
        encoding="utf-8",
    )
    print(f"\nRelatorio: {relatorio}")

    dados = Path(args.dados)
    dados.mkdir(parents=True, exist_ok=True)
    gravar = {"train_preprocessado": r.train, "test_preprocessado": r.test}
    for nome, (train_esc, test_esc) in r.escalonamento.escalonados.items():
        # Mesmas linhas e colunas do arquivo principal, so as features trocadas.
        gravar[f"train_{nome}"] = r.train.assign(**train_esc.reset_index(drop=True)[r.colunas_features])
        if test_esc is not None:
            gravar[f"test_{nome}"] = r.test.assign(**test_esc.reset_index(drop=True)[r.colunas_features])
    for nome, df in gravar.items():
        if df is None:
            continue
        caminho = dados / f"{nome}.csv"
        df.to_csv(caminho, index=False, encoding="utf-8")
        print(f"{nome}: {df.shape} -> {caminho}")

    print(f"Concluido em {time.time() - inicio:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
