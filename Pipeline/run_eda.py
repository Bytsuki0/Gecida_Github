"""Executa as Fases 1 e 2 (carregamento + EDA) sobre qualquer par train/test.

Exemplos
--------
Dataset do projeto (valores padrao ja apontam para ele):

    python Pipeline/run_eda.py

Qualquer outro dataset:

    python Pipeline/run_eda.py --train dados/treino.csv --test dados/teste.csv \\
        --target preco --id id --outdir outputs/eda_precos

Regras logicas de dominio (condicoes que NAO deveriam ocorrer) podem ser
declaradas e sao verificadas ao final da Fase 2:

    --regra "Listening_Time_minutes > Episode_Length_minutes"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from Pipeline.eda import EDAConfig  # noqa: E402
from Pipeline.eda import fase1_carregamento, fase2_eda  # noqa: E402
from Pipeline.eda.estilo import aplicar_estilo  # noqa: E402


def montar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fases 1 e 2 do pipeline: carregamento e analise exploratoria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train", default=str(RAIZ / "train.csv"), help="CSV de treino")
    p.add_argument("--test", default=str(RAIZ / "test.csv"), help="CSV de teste (opcional)")
    p.add_argument("--sem-test", action="store_true",
                   help="rodar apenas com o train, ignorando --test")
    p.add_argument("--target", default="Listening_Time_minutes", help="coluna alvo")
    p.add_argument("--id", dest="id_col", default="id", help="coluna identificadora")
    p.add_argument("--outdir", default=str(RAIZ / "outputs" / "eda"), help="diretorio de saida")
    p.add_argument("--amostra-scatter", type=int, default=20_000,
                   help="linhas amostradas nos graficos de dispersao")
    p.add_argument("--top-n-categorias", type=int, default=20,
                   help="niveis exibidos nos graficos categoricos")
    p.add_argument("--regra", action="append", default=None, metavar="EXPR",
                   help="condicao pandas.query que NAO deveria ocorrer (repetivel)")
    p.add_argument("--sem-regras-padrao", action="store_true",
                   help="nao aplicar as regras logicas padrao do dataset de podcasts")
    return p


# Regras especificas do dataset de podcasts. Sao aplicadas somente quando o alvo
# padrao esta em uso; para outros datasets passe as suas com --regra.
REGRAS_PADRAO_PODCAST = [
    "Listening_Time_minutes > Episode_Length_minutes",
    "Number_of_Ads > 10",
    "Episode_Length_minutes <= 0",
]


def main(argv: list[str] | None = None) -> int:
    args = montar_parser().parse_args(argv)

    regras = list(args.regra or [])
    if not args.sem_regras_padrao and args.target == "Listening_Time_minutes":
        regras = REGRAS_PADRAO_PODCAST + regras

    cfg = EDAConfig(
        target=args.target,
        id_col=args.id_col,
        outdir=Path(args.outdir),
        amostra_scatter=args.amostra_scatter,
        top_n_categorias=args.top_n_categorias,
        regras_logicas=regras,
    )

    aplicar_estilo()
    cfg.preparar()

    caminho_test = None
    if not args.sem_test:
        caminho_test = args.test if args.test and Path(args.test).exists() else None
        if args.test and caminho_test is None:
            print(f"[aviso] test nao encontrado em '{args.test}' - seguindo apenas com o train.")

    print("Fase 1 - carregando dados...")
    f1 = fase1_carregamento.executar(args.train, caminho_test, cfg)
    print(f"  train {f1.train.shape}" + (f" | test {f1.test.shape}" if f1.test is not None else ""))

    if not f1.estabilidade.empty:
        extrapolam = f1.estabilidade[f1.estabilidade["test_fora_do_intervalo_train"] > 0]
        for _, r in extrapolam.iterrows():
            print(f"  [test fora do intervalo do train] {r['coluna']}: "
                  f"{int(r['test_fora_do_intervalo_train'])} linha(s), max {r['max_test']:,.2f}")

    print("Fase 2 - analise exploratoria...")
    f2 = fase2_eda.executar(f1.train, cfg)
    print(f"  {len(f2.figuras)} figuras | {len(f2.alertas)} alertas")

    relatorio = cfg.outdir / "relatorio_eda.md"
    relatorio.write_text(
        "\n".join(
            [
                f"# EDA - Fases 1 e 2 | alvo: `{cfg.target}`\n",
                "Gerado por `Pipeline/run_eda.py`. Tabelas completas em `tabelas/`, "
                "figuras em `figuras/`.\n",
                f1.markdown,
                "\n---\n",
                f2.markdown,
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nRelatorio: {relatorio}")
    if not f2.alertas.empty:
        criticos = f2.alertas[f2.alertas["severidade"] == "critico"]
        print(f"Alertas criticos: {len(criticos)}")
        for _, r in criticos.iterrows():
            print(f"  - {r['coluna']}: {r['achado']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
