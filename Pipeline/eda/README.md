# Fases 1 e 2 - Carregamento e EDA

> **Versao narrada:** `Pipeline/EDA_Fases_1_e_2.ipynb` reune toda a analise em um
> unico notebook, em portugues, com os graficos e a interpretacao de cada achado.
> Este modulo e a versao automatizada da mesma analise.

Implementa os dois primeiros blocos do `Diagrama_de_desenvolvimento.pdf`. O
codigo nao conhece o dataset de podcasts: tudo o que e especifico entra por
`EDAConfig` ou pela linha de comando, entao o mesmo pipeline roda sobre
qualquer par train/test.

**Fora de escopo por decisao:** engenharia de features (ultimo bloco da Fase 2)
e tudo da Fase 3 em diante. Nenhuma funcao aqui modifica os dados - as fases
apenas descrevem o que chegou e produzem a lista de decisoes para a Fase 3.

## Uso

```bash
# dataset do projeto (padroes ja apontam para train.csv / test.csv na raiz)
python Pipeline/run_eda.py

# qualquer outro dataset
python Pipeline/run_eda.py --train dados/treino.csv --test dados/teste.csv \
    --target preco --id id --outdir outputs/eda_precos

# sem conjunto de teste (a Fase 1 roda so com o train)
python Pipeline/run_eda.py --train dados/treino.csv --sem-test --target preco
```

Regras logicas de dominio (condicoes que **nao** deveriam ocorrer) sao
verificadas ao final da Fase 2:

```bash
python Pipeline/run_eda.py --regra "duracao <= 0" --regra "preco > teto"
```

Para o alvo `Listening_Time_minutes` tres regras ja vem ligadas por padrao
(ver `REGRAS_PADRAO_PODCAST` em `run_eda.py`); `--sem-regras-padrao` desliga.

## Saida

```
outputs/eda/
├── relatorio_eda.md    # relatorio completo, com tabelas e figuras embutidas
├── figuras/            # PNG de cada bloco
└── tabelas/            # CSV de cada tabela (materia-prima do relatorio LaTeX)
```

## Modulos

| Arquivo | Papel |
|---|---|
| `config.py` | `EDAConfig` - alvo, id, limiares, amostragem, parametros de figura |
| `estilo.py` | paleta validada para daltonismo e tema dos graficos |
| `utils.py` | classificacao de colunas, IQR, eta^2, PSI, persistencia, markdown |
| `fase1_carregamento.py` | leitura, integridade, alinhamento train/test, PSI |
| `fase2_eda.py` | blocos 2.1 a 2.5 do diagrama + consolidacao de alertas (2.6) |

## O que cada bloco entrega

- **1.1-1.4** dimensoes, memoria, integridade (alvo, ids, duplicatas), colunas
  divergentes entre train e test, PSI por coluna e contagem de valores do test
  fora do intervalo do train.
- **2.1** uma linha por coluna: papel inferido, dtype, ausentes, cardinalidade,
  valor dominante.
- **2.2** estatisticas de forma do alvo (skewness, kurtosis, normalidade) e
  painel com histograma, boxplot, ECDF e Q-Q plot.
- **2.3** describe estendido, limites e contagem de outliers IQR, dispersao vs
  alvo com a media do alvo por decil da preditora.
- **2.4** frequencias, distribuicao do alvo por categoria e `eta^2` (fracao da
  variancia do alvo explicada pela variavel).
- **2.5** heatmaps de Pearson e Spearman e pares de preditoras redundantes.
- **2.6** alertas priorizados (critico / atencao / info) - a entrada pronta para
  as decisoes da Fase 3.

## Como estender

A classificacao de colunas em `utils.classificar_colunas` e estrutural (dtype +
cardinalidade). Para mudar o que conta como categorica, identificador ou
numerica discreta, ajuste os limiares em `EDAConfig` - nao o codigo de analise.
