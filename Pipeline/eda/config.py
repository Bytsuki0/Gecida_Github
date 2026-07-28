"""Configuracao central das Fases 1 e 2 do diagrama de desenvolvimento.

Tudo o que e especifico de um dataset (alvo, id, caminhos, limiares) mora aqui.
Os modulos de analise consomem apenas `EDAConfig`, o que permite rodar a mesma
EDA sobre qualquer par train/test sem tocar no codigo de analise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EDAConfig:
    """Parametros de execucao da EDA.

    Attributes:
        target: nome da coluna alvo (obrigatorio; deve existir no train).
        id_col: coluna identificadora, excluida das analises estatisticas.
        outdir: diretorio raiz onde figuras, tabelas e relatorio sao gravados.
    """

    target: str
    id_col: str | None = None
    outdir: Path = Path("outputs/eda")

    # --- classificacao automatica de colunas -------------------------------
    # Acima deste numero de categorias a coluna e tratada como "alta
    # cardinalidade": entra nas analises, mas so os `top_n` niveis sao plotados.
    max_cardinalidade_categorica: int = 50
    # Razao nunique/n acima da qual a coluna e considerada identificador.
    razao_identificador: float = 0.95
    # Numericas com poucos valores distintos sao sinalizadas como discretas.
    max_valores_discretos: int = 20
    # Niveis exibidos nos graficos de variaveis categoricas.
    top_n_categorias: int = 20

    # --- limiares de alerta -------------------------------------------------
    # Regra do UML (Fase 3): coluna com mais NaN que isso e candidata a drop.
    limite_nan_coluna: float = 50.0
    # Fracao de outliers IQR a partir da qual a coluna e sinalizada.
    limite_outliers_pct: float = 5.0
    # |skew| acima disso indica assimetria relevante (candidata a transformacao).
    limite_skew: float = 1.0
    # |r| acima disso indica par redundante (Fase 3: selecao de features).
    limite_correlacao_alta: float = 0.9
    # Categoria dominante acima disso indica coluna quase constante.
    limite_categoria_dominante: float = 95.0
    # PSI train vs test: > 0.1 merece atencao, > 0.25 e drift forte.
    limite_psi: float = 0.1

    # --- amostragem para graficos pesados ----------------------------------
    amostra_scatter: int = 20_000
    amostra_boxplot: int = 100_000
    random_state: int = 42

    # --- figuras ------------------------------------------------------------
    dpi: int = 120
    formato_figura: str = "png"
    largura_painel: float = 5.0
    altura_painel: float = 3.6
    colunas_grade: int = 3

    # Regras logicas de dominio no formato aceito por `DataFrame.query`.
    # Cada regra descreve uma condicao que NAO deveria ocorrer.
    regras_logicas: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)

    @property
    def dir_figuras(self) -> Path:
        return self.outdir / "figuras"

    @property
    def dir_tabelas(self) -> Path:
        return self.outdir / "tabelas"

    def preparar(self) -> None:
        """Cria a arvore de saida."""
        self.dir_figuras.mkdir(parents=True, exist_ok=True)
        self.dir_tabelas.mkdir(parents=True, exist_ok=True)
