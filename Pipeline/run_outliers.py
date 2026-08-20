"""Roda o bloco 'Tratamento de outliers' da Fase 3 sobre train/test.

Exemplos
--------
Dataset do projeto (valores padrao ja apontam para ele):

    python Pipeline/run_outliers.py

Trocando o metodo de uma coluna especifica (ex.: winsorization em vez de IQR):

    python Pipeline/run_outliers.py --metodo Number_of_Ads=winsor

Escolhendo o metodo padrao para todas as colunas nao listadas em --metodo:

    python Pipeline/run_outliers.py --metodo-padrao zscore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from Pipeline.eda import EDAConfig  # noqa: E402
from Pipeline.eda import fase1_carregamento, fase3_outliers  # noqa: E402
from Pipeline.eda.estilo import aplicar_estilo  # noqa: E402

# Regras de dominio especificas do dataset de podcasts: valores logicamente
# impossiveis, corrigidos ANTES do tratamento estatistico de outliers.
REGRAS_DOMINIO_PODCAST = {
    "Host_Popularity_percentage": {"inf": 0.0, "sup": 100.0, "acao": "clip"},
    "Guest_Popularity_percentage": {"inf": 0.0, "sup": 100.0, "acao": "clip"},
    # duracao <= 0 nao tem conserto por clipping (0 continua sem sentido) -
    # vira NaN para ser tratado pela etapa de imputacao (fora do escopo aqui).
    "Episode_Length_minutes": {"inf": 0.0, "inf_exclusivo": True, "sup": None, "acao": "nan"},
}

# Number_of_Ads e uma contagem discreta bem assimetrica (skew ~6, kurtosis
# ~500): IQR classificaria ate 6-10 anuncios legitimos como outlier. Um
# winsor generoso (p99) preserva esses valores e ainda recorta os absurdos
# (103 no train, 2063 no test).
METODOS_PADRAO_PODCAST = {
    "Number_of_Ads": "winsor",
}


def montar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fase 3.1 - tratamento de outliers (correcao de dominio + IQR/winsor/z-score).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train", default=str(RAIZ / "train.csv"), help="CSV de treino")
    p.add_argument("--test", default=str(RAIZ / "test.csv"), help="CSV de teste (opcional)")
    p.add_argument("--sem-test", action="store_true", help="rodar apenas com o train")
    p.add_argument("--target", default="Listening_Time_minutes", help="coluna alvo (excluida do tratamento)")
    p.add_argument("--id", dest="id_col", default="id", help="coluna identificadora (excluida do tratamento)")
    p.add_argument("--outdir", default=str(RAIZ / "outputs" / "eda"), help="diretorio de saida")
    p.add_argument("--metodo-padrao", default="iqr", choices=["iqr", "winsor", "zscore"],
                    help="metodo usado nas colunas nao listadas em --metodo")
    p.add_argument("--metodo", action="append", default=None, metavar="COLUNA=METODO",
                    help="metodo especifico por coluna, ex. Number_of_Ads=winsor (repetivel)")
    p.add_argument("--k-iqr", type=float, default=1.5, help="multiplicador do IQR (regra de Tukey)")
    p.add_argument("--limiar-zscore", type=float, default=3.0, help="limiar de z-score")
    p.add_argument("--sem-regras-padrao", action="store_true",
                    help="nao aplicar as regras de dominio padrao do dataset de podcasts")
    return p


def main(argv: list[str] | None = None) -> int:
    args = montar_parser().parse_args(argv)

    cfg = EDAConfig(target=args.target, id_col=args.id_col, outdir=Path(args.outdir))
    aplicar_estilo()
    cfg.preparar()

    caminho_test = None
    if not args.sem_test:
        caminho_test = args.test if args.test and Path(args.test).exists() else None
        if args.test and caminho_test is None:
            print(f"[aviso] test nao encontrado em '{args.test}' - seguindo apenas com o train.")

    print("Carregando dados (Fase 1)...")
    f1 = fase1_carregamento.executar(args.train, caminho_test, cfg)
    print(f"  train {f1.train.shape}" + (f" | test {f1.test.shape}" if f1.test is not None else ""))

    metodo_por_coluna = dict(METODOS_PADRAO_PODCAST) if args.target == "Listening_Time_minutes" else {}
    if args.metodo_padrao != "iqr":
        # metodo-padrao explicito prevalece sobre os defaults do dataset,
        # exceto onde --metodo tambem foi passado explicitamente (abaixo).
        metodo_por_coluna = {}
    for item in args.metodo or []:
        if "=" not in item:
            print(f"[aviso] ignorando --metodo mal formado: '{item}' (use COLUNA=METODO)")
            continue
        coluna, metodo = item.split("=", 1)
        metodo_por_coluna[coluna] = metodo

    from Pipeline.eda.utils import classificar_colunas
    colunas_numericas = classificar_colunas(f1.train, cfg).numericas
    metodo_por_coluna = {c: metodo_por_coluna.get(c, args.metodo_padrao) for c in colunas_numericas}

    regras_dominio = {} if args.sem_regras_padrao or args.target != "Listening_Time_minutes" else REGRAS_DOMINIO_PODCAST

    print("Tratando outliers (Fase 3.1)...")
    print(f"  metodos: {metodo_por_coluna}")
    f3 = fase3_outliers.executar(
        f1.train,
        f1.test,
        cfg,
        metodo_por_coluna=metodo_por_coluna,
        regras_dominio=regras_dominio,
        k_iqr=args.k_iqr,
        limiar_zscore=args.limiar_zscore,
    )

    if not f3.correcao_dominio.empty:
        print("  correcao de dominio:")
        for _, r in f3.correcao_dominio.iterrows():
            print(f"    {r['coluna']}: {r['n_corrigidos']} linha(s) ({r['acao']}) fora de {r['regra']}")
    print("  outliers tratados por coluna (train):")
    for _, r in f3.resumo.iterrows():
        print(
            f"    {r['coluna']} [{r['metodo']}]: {int(r['n_afetados_train'])} linha(s) "
            f"({r['pct_afetados_train']:.3f}%) recortadas para [{r['limite_inf']:.3f}, {r['limite_sup']:.3f}]"
        )

    relatorio = cfg.outdir / "relatorio_fase3_outliers.md"
    relatorio.write_text(
        f"# Fase 3.1 - Tratamento de outliers | alvo: `{cfg.target}`\n\n"
        "Gerado por `Pipeline/run_outliers.py`. Tabelas em `tabelas/`, figuras em `figuras/`.\n\n"
        + f3.markdown,
        encoding="utf-8",
    )
    print(f"\nRelatorio: {relatorio}")

    saida_train = cfg.outdir.parent / "train_sem_outliers.csv"
    f3.train.to_csv(saida_train, index=False, encoding="utf-8")
    print(f"Train tratado: {saida_train}")
    if f3.test is not None:
        saida_test = cfg.outdir.parent / "test_sem_outliers.csv"
        f3.test.to_csv(saida_test, index=False, encoding="utf-8")
        print(f"Test tratado: {saida_test}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
