"""Pipeline de EDA - Fases 1 e 2 do diagrama de desenvolvimento.

Uso tipico:

    from Pipeline.eda import EDAConfig, fase1_carregamento, fase2_eda

    cfg = EDAConfig(target="Listening_Time_minutes", id_col="id")
    f1 = fase1_carregamento.executar("train.csv", "test.csv", cfg)
    f2 = fase2_eda.executar(f1.train, cfg)
"""

from .config import EDAConfig
from .utils import Colunas, classificar_colunas

__all__ = ["EDAConfig", "Colunas", "classificar_colunas"]
