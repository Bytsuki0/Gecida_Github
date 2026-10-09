# Roteiro de apresentação — Fases 1, 2 e 3

**Notebook de apoio:** `Pipeline/Parts-1,2,3.ipynb`
**Duração estimada:** 22 a 26 minutos (sem perguntas)
**Como usar:** o texto em parágrafos é para ser **falado**. As linhas entre colchetes
`[ ... ]` são **marcações de cena** — o que rolar na tela, o que apontar. Não leia
as marcações em voz alta.

**Divisão sugerida se a apresentação for em dupla ou trio:**

| Bloco | Conteúdo | Tempo |
|---|---|---|
| Abertura | Contexto e problema | 2 min |
| Fase 1 | Coleta e validação dos dados | 4 min |
| Fase 2 | Análise exploratória | 8 min |
| Fase 3 | Pré-processamento e features | 10 min |
| Fecho | Conclusões e próximos passos | 2 min |

---

## ABERTURA — 2 minutos

[Slide de capa ou primeira célula do notebook]

Bom dia a todos. Vamos apresentar as Fases 1, 2 e 3 do nosso projeto: coleta e
validação dos dados, análise exploratória e pré-processamento.

O problema é o seguinte: temos dados de episódios de podcast e queremos prever
quantos minutos uma pessoa vai efetivamente escutar de um episódio. A variável
alvo se chama `Listening_Time_minutes`. É um problema de **regressão**, com
750 mil linhas de treino e 250 mil de teste.

Antes de entrar nos números, quero adiantar a conclusão geral, porque ela organiza
tudo o que vem depois: **este é um problema quase unidimensional**. Uma única
variável — a duração do episódio — explica 84% da variância do que a pessoa
escuta. Todo o resto do dataset, somado, contribui com menos de 1%.

Isso muda o que faz sentido fazer em cada fase. E boa parte da nossa apresentação
é justamente sobre **descobrir o que não valia a pena fazer** — e por quê, com
número na mão.

Uma observação de método antes de começar: o notebook foi escrito para ser
auditável. As Fases 1 e 2 são **puramente descritivas** — nenhuma célula altera
os dados. A transformação só começa na Fase 3, e sempre sobre cópias, para que
qualquer pessoa consiga reproduzir a análise do início ao fim.

---

## FASE 1 — COLETA E VALIDAÇÃO DOS DADOS — 4 minutos

### 1.1 Leitura dos arquivos

[Mostrar célula 1.1 — `train.head()`]

Começamos carregando os dois arquivos. O `train` tem **750 mil linhas e 12
colunas**; o `test` tem **250 mil linhas e 11 colunas** — uma a menos, porque é
exatamente a coluna que temos que prever.

As colunas descrevem o episódio: nome do podcast, título, duração em minutos,
gênero, popularidade do apresentador e do convidado em percentual, dia e horário
de publicação, número de anúncios e o sentimento do episódio.

### 1.2 Alinhamento entre train e test

[Mostrar a tabela de alinhamento]

Primeira verificação: os dois arquivos falam a mesma língua?

Sim. **Onze colunas em comum**, uma só no train — o alvo — nenhuma só no test, e
**zero divergências de tipo**. Isso importa porque um tipo diferente entre treino
e teste é o clássico bug que só aparece na hora de gerar a submissão.

### 1.3 Estabilidade das distribuições — o PSI

[Mostrar a tabela de PSI]

Segunda verificação, mais fina: o `test` é uma amostra da mesma população do
`train`, ou os dados vieram de contextos diferentes?

Para isso usamos o **PSI**, o *Population Stability Index*. Ele compara a
distribuição de cada coluna nos dois arquivos. A convenção é: abaixo de 0,10 é
estável; entre 0,10 e 0,25 merece atenção; acima de 0,25 é deslocamento forte.

O resultado é tranquilizador: **o maior PSI de todo o dataset é 0,0006**, no
`Episode_Title`. Todas as colunas estão duas ordens de grandeza abaixo do limite
de "estável". As taxas de valores ausentes também batem: 11,6% no train contra
11,5% no test para a duração; 19,47% contra 19,53% para a popularidade do
convidado.

Ou seja: **não há *data drift***. O test é a mesma população.

### 1.4 Mas há algo estranho nos máximos

[Apontar as colunas `max_train` e `max_test` da mesma tabela, depois a tabela de valores fora do intervalo]

Só que a mesma tabela guarda uma surpresa. Olhem as duas últimas colunas: o
máximo de `Episode_Length_minutes` no train é **325 minutos**. No test é
**78 milhões e 486 mil minutos**.

[Pausa]

Isso são cerca de **149 anos de episódio**.

E não é caso isolado. `Number_of_Ads` vai até 103 no train e até **2.063** no test.

Duas linhas do test estão fora do intervalo da duração e uma está fora do
intervalo de anúncios. A razão entre o máximo do test e o do train é de
**241 mil vezes** para a duração.

[Mostrar a célula com as 3 maiores durações do test]

Quando olhamos as três maiores durações do test, o padrão fica claro: 78 milhões,
depois 7.575, e só na terceira posição um valor plausível, 120,73. Não é uma
distribuição com cauda longa — são registros corrompidos isolados.

**Isso é a primeira lição do projeto: o test é mais sujo que o train.** E como o
test é o que vai para produção, qualquer tratamento que a gente construir precisa
funcionar nele também.

### 1.5 Comparação visual

[Mostrar a grade de histogramas train vs test]

Para confirmar, plotamos as distribuições sobrepostas dentro de uma janela robusta,
recortando os 0,1% de cada ponta.

Dentro dessa janela as duas curvas praticamente se **sobrepõem**. Barra azul é o
train, linha laranja é o test. Confirma o PSI: mesma população, com um punhado de
registros corrompidos.

E reparem no **formato** dessas distribuições — isso vai ser importante daqui a
pouco. `Episode_Length_minutes`, `Host_Popularity_percentage` e
`Guest_Popularity_percentage` são **quase uniformes**, não normais. São
retângulos, não sinos.

**Guardem esse detalhe.** Ele é o motivo de a regra do IQR não funcionar neste
dataset, e vamos voltar nele na Fase 2.

---

## FASE 2 — ANÁLISE EXPLORATÓRIA — 8 minutos

### 2.1 Classificação e valores ausentes

[Mostrar a saída de `classificar_colunas` e a tabela `descrever_variaveis`]

A Fase 2 começa classificando as colunas automaticamente:

- **4 numéricas:** duração, popularidade do host, popularidade do convidado e número de anúncios
- **5 categóricas:** nome do podcast, gênero, dia, horário e sentimento
- **1 de alta cardinalidade:** `Episode_Title`, com 100 valores distintos
- **Zero identificadores e zero constantes** — nada para descartar de saída

[Mostrar o gráfico de barras de valores ausentes]

Sobre valores ausentes, só três colunas têm algum, e o gráfico mostra a linha
vermelha do nosso limite de descarte, que é 50%:

- `Guest_Popularity_percentage`: **19,47%** — cerca de 146 mil linhas
- `Episode_Length_minutes`: **11,61%** — cerca de 87 mil linhas
- `Number_of_Ads`: um único valor

Nenhuma chega perto de 50%, então nenhuma coluna é descartada. Mas guardem que a
segunda da lista é a duração — a variável mais importante do dataset. Isso vai
custar caro na Fase 3.

### 2.2 A distribuição do alvo

[Mostrar a tabela de métricas do alvo e o painel de 4 gráficos]

Agora o alvo. Quatro visões: histograma, boxplot, distribuição acumulada e Q-Q plot.

Os números: média de **45,4 minutos**, mediana de **43,4**, desvio padrão de
**27,1**. O alvo vai de **0 a 119,97**.

Três leituras importantes:

**Primeira — a assimetria é leve.** Skewness de **0,35** e kurtosis de **−0,66**,
ou seja, achatada em relação à normal. Isso é relevante porque o reflexo automático
em regressão é aplicar `log1p` num alvo enviesado. Aqui **não é o caso** — a
transformação não é necessária e, como vamos mostrar na 3.6, atrapalharia.

**Segunda — zero outliers pela regra IQR.** O boxplot não tem nenhum ponto fora
dos bigodes. A distribuição é larga e sem caudas longas.

**Terceira — o Q-Q plot mostra um "piso" achatado perto do zero.** Isso é
**1,14% das observações exatamente iguais a zero**: pessoas que abriram o episódio
e não escutaram nada. É um comportamento qualitativamente diferente do resto e
ficou registrado como algo a investigar.

Sobre o teste de normalidade: ele rejeita a hipótese nula com p na ordem de
10 elevado a menos 74. Mas quero ser honesto aqui — **com 750 mil observações,
qualquer desvio mínimo seria rejeitado**. A forma do gráfico informa mais do que
o p-valor.

**Consequência prática para a Fase 4:** distribuição sem cauda longa, sem
transformação necessária. **RMSE** é a métrica principal adequada. O RMSLE que o
diagrama original previa faz menos sentido aqui, porque ele penaliza erro
relativo — útil quando o alvo cobre várias ordens de grandeza, o que não é o caso.

### 2.3 As variáveis numéricas — e por que o IQR falha

[Mostrar a tabela `resumo_num` e a grade de histogramas]

Aqui está o achado técnico mais interessante da Fase 2.

[Mostrar a grade de boxplots com a contagem de outliers no título]

Olhem quantos outliers a regra do IQR encontra em 750 mil linhas:

| coluna | limite superior do IQR | máximo observado | outliers |
|---|---|---|---|
| `Episode_Length_minutes` | 181,6 | 325,2 | **1** em 662.907 |
| `Host_Popularity_percentage` | 139,7 | 119,5 | **0** |
| `Guest_Popularity_percentage` | 148,9 | 119,9 | **0** |
| `Number_of_Ads` | 5,0 | 103,9 | **9** |

Dez linhas no dataset inteiro. **A regra IQR é praticamente inócua aqui.**

E o motivo é aquele detalhe que pedi para guardarem na Fase 1: **as distribuições
são quase uniformes**. Kurtosis de menos 1,2. Uma distribuição uniforme tem os
quartis muito afastados um do outro, e o IQR é justamente a distância entre eles.
Quartis afastados empurram os limites para longe.

Vejam o caso extremo: para `Host_Popularity_percentage`, o limite superior do IQR
é **139,7** — um valor que a variável **nunca alcança**, porque o máximo observado
é 119,5. O limite estatístico está fora do alcance da variável.

**A consequência é direta para o nosso plano.** O diagrama de desenvolvimento
previa "IQR clipping, winsorization, z-score maior que 3" na Fase 3. Nós
demonstramos que **esse tratamento não removeria os erros reais dos dados**.
Percentuais de 119% passariam intactos — porque estatisticamente eles não são
extremos. São apenas **impossíveis**.

`Number_of_Ads` é o caso oposto e igualmente instrutivo: skewness de **6,03** e
kurtosis de **506**, produzidas por apenas **nove valores corrompidos** numa
variável cuja escala real é de 0 a 3. Um punhado de erros distorce completamente
as estatísticas de forma.

### 2.4 A relação com o alvo — o achado central

[Mostrar a grade de dispersões com a linha de média por decil, depois a tabela de correlações]

Agora a pergunta central: o que prevê o alvo?

| variável | Pearson | Spearman |
|---|---|---|
| `Episode_Length_minutes` | **0,917** | **0,932** |
| `Number_of_Ads` | −0,118 | −0,115 |
| `Host_Popularity_percentage` | 0,051 | 0,045 |
| `Guest_Popularity_percentage` | −0,016 | −0,015 |

**`Episode_Length_minutes` sozinha explica cerca de 84% da variância do alvo** —
0,917 ao quadrado dá 0,84.

O gráfico de dispersão mostra por quê: os pontos formam uma **faixa triangular**
bem definida, com o alvo limitado superiormente pela duração do episódio. E faz
todo sentido: **não se escuta mais tempo do que o episódio dura**. A linha laranja,
que é a média do alvo por decil, é praticamente uma reta.

As outras três variáveis somadas não chegam perto. A segunda mais forte, o número
de anúncios, tem correlação de menos 0,12.

### 2.5 As variáveis categóricas

[Mostrar a tabela `resumo_cat` e o gráfico de barras de η²]

Para medir o poder explicativo das categóricas usamos o **eta quadrado**, que é a
fração da variância do alvo explicada pela variável categórica.

| variável | categorias | η² |
|---|---|---|
| `Episode_Title` | 100 | 0,0058 |
| `Podcast_Name` | 48 | 0,0026 |
| `Episode_Sentiment` | 3 | 0,0016 |
| `Publication_Time` | 4 | 0,0006 |
| `Genre` | 10 | 0,0006 |
| `Publication_Day` | 7 | 0,0003 |

**A melhor categórica explica 0,58% da variância.** `Genre`, `Publication_Day` e
`Publication_Time` ficam abaixo de 0,07%.

[Mostrar um ou dois dos boxplots por categoria]

Os boxplots contam a mesma história visualmente: em todas as variáveis as caixas
estão praticamente alinhadas, as medianas mal se deslocam de uma categoria para
outra.

Duas observações que ficaram anotadas para a Fase 3:

**Primeira:** `Episode_Title` tem 100 categorias e `Podcast_Name` tem 48. One-Hot
Encoding nas duas adicionaria **148 colunas** para capturar menos de 1% de
variância. Isso é caro demais.

**Segunda:** `Episode_Title` é literalmente o texto "Episode N", com N de 1 a 100.
**É um número disfarçado de texto.** Extrair o inteiro seria o caminho natural — e
é exatamente o que fizemos na Fase 3.

E nenhuma categoria domina o dataset: a maior concentração é `Publication_Time`
igual a `Night`, com 26%. Então não há coluna quase constante para descartar.

### 2.6 A matriz de correlação

[Mostrar os dois heatmaps, Pearson e Spearman]

Verificação de multicolinearidade entre as preditoras: **nenhum par com
correlação absoluta maior ou igual a 0,9**. Na verdade nenhum par passa de 0,06.

As variáveis originais são **estatisticamente independentes entre si**.

Isso significa que a etapa de seleção por VIF prevista no diagrama **não vai
eliminar nada** — pelo menos não nas variáveis originais. Vamos voltar nisso na
Fase 3, porque a história muda.

### 2.7 As regras lógicas de domínio

[Mostrar a tabela de violações]

E aqui está o ponto que amarra a Fase 2 inteira.

As seções anteriores mostraram que a estatística descritiva **não encontra** os
erros deste dataset. O que encontra são **regras que descrevem o que é impossível
no mundo real**.

Verificamos três condições que não deveriam existir:

1. Escutar mais do que o episódio dura
2. Número de anúncios maior que 10 — a escala real é 0 a 3
3. Duração menor ou igual a zero

Mais uma checagem semântica automática: colunas de percentual fora do intervalo
de 0 a 100.

O resultado:

| regra violada | linhas |
|---|---|
| escuta > duração | **2.568** |
| `Host_Popularity_percentage` fora de [0, 100] | 25 |
| `Guest_Popularity_percentage` fora de [0, 100] | 19 |
| `Number_of_Ads` > 10 | 9 |
| `Episode_Length_minutes` ≤ 0 | 1 |

Comparem: a regra IQR encontrou 10 linhas. As regras de domínio encontram cerca
de 130 erros reais, mais 2.568 linhas logicamente impossíveis.

### 2.8 Mas atenção à magnitude — a decisão mais importante da fase

[Mostrar a tabela de faixas de excesso e os dois grupos de exemplos]

Aqui vem a parte da qual a gente mais se orgulha na Fase 2.

A regra bruta "escuta maior que duração" marca **2.568 linhas**. O reflexo natural
seria descartar todas. **Seria um erro grave.**

Quando olhamos a **magnitude** da violação, ela separa dois grupos completamente
diferentes:

- Excesso **mediano**: 0,02 minuto
- Excesso **médio**: 1,67 minuto
- Excesso **máximo**: 115,54 minutos

E na distribuição acumulada:

| faixa | linhas | % das impossíveis |
|---|---|---|
| excesso ≤ 0,01 min | 1.142 | 44,5% |
| excesso ≤ 1 min | 2.185 | 85,1% |
| **excesso ≤ 5 min** | **2.483** | **96,7%** |

[Mostrar os exemplos dos dois grupos]

Olhem os dois grupos lado a lado. No **Grupo 1**, uma linha com duração de 7,4647
e escuta de 7,4648. Isso é **arredondamento** — a pessoa escutou o episódio
inteiro e a diferença é na quarta casa decimal.

No **Grupo 2**, uma linha com duração de 1,24 minuto e escuta de 116,78. Isso é
**erro de dados** — um dos dois valores está simplesmente errado.

**96,7% das violações são arredondamento.** Só **85 linhas** têm violação acima de
5 minutos.

Se a gente tivesse aplicado a regra sem olhar a magnitude, teria **descartado 2.483
linhas perfeitamente válidas** — justamente as que representam escuta completa do
episódio, que é o comportamento mais informativo do dataset.

### 2.9 Alertas consolidados

[Mostrar a tabela `alertas`]

Fechamos a Fase 2 com uma tabela priorizada de tudo o que a EDA encontrou:
**5 alertas críticos, 5 de atenção e 6 informativos**. Cada linha dessa tabela é
uma decisão a tomar, e ela é literalmente a entrada da Fase 3.

### Resumo da Fase 2 — cinco frases

1. **O problema é quase unidimensional** — a duração explica 84%.
2. **As categóricas são quase irrelevantes** — todos os η² abaixo de 0,006.
3. **Não há multicolinearidade** nas variáveis originais.
4. **Os erros não são estatísticos, são lógicos** — e a magnitude da violação
   importa tanto quanto a violação.
5. **O test é mais sujo que o train** e precisa do mesmo tratamento.

---

## FASE 3 — PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES — 10 minutos

### 3.0 A regra que organiza a fase inteira

[Mostrar a célula de abertura da Fase 3]

Da Fase 3 em diante os dados **são transformados**. Trabalhamos sobre cópias
chamadas `p_train` e `p_test`, e os DataFrames originais continuam intactos —
isso mantém as Fases 1 e 2 reproduzíveis.

E existe uma regra que vale para tudo o que vem a seguir:

> **Todo parâmetro é estimado no `train` e aplicado sem reajuste ao `test`.**

Mediana de imputação, moda, limites de outlier, mínimos e máximos do escalonamento,
médias do target encoding: **tudo calculado só no treino**. Recalcular qualquer um
deles usando o test vazaria informação do conjunto de avaliação para dentro do
pré-processamento, e produziria uma métrica otimista que não se sustentaria na
Fase 5.

Criamos também um **diário de transformações** — cada etapa registra o que fez e
quantas linhas ou colunas afetou, no train e no test. No fim da fase esse diário
vira a auditoria completa.

### 3.1 Correção de valores fora do domínio

[Mostrar a tabela de correções de domínio]

**Esta etapa não estava no diagrama — ela vem antes dele, e a Fase 2 explicou por
quê.**

Definimos o intervalo logicamente válido de cada coluna:

- Percentuais: recortados em [0, 100]
- `Number_of_Ads`: recortado em [0, 3]
- Duração menor ou igual a zero: vira `NaN`, porque recortar para zero não
  resolveria nada

O resultado: **25 e 19 percentuais corrigidos no train**, 12 e 5 no test;
**9 valores de anúncios recortados no train**, 2 no test; e uma duração virou
ausente.

Duas coisas importantes sobre a **ordem**:

Primeiro, essa etapa tem que vir **antes da imputação**, por um motivo mecânico: a
mediana usada para imputar seria calculada sobre valores corrompidos.

Segundo, ela **cria** valores ausentes — aquela duração ≤ 0 que virou `NaN`. A
etapa seguinte vai tratar.

### 3.2 A regra de duas colunas

[Mostrar a tabela de sensibilidade à tolerância]

Aquela regra "escuta maior que duração" olha duas colunas ao mesmo tempo, então
precisou de um tratamento próprio.

Aplicamos a decisão que a Fase 2.8 fundamentou: **tolerância de 5 minutos**.

A tabela mostra quantas linhas cada tolerância descartaria:

| tolerância | linhas descartadas |
|---|---|
| 0 min | 2.567 |
| 1 min | 382 |
| **5 min** | **84** |
| 10 min | 62 |

Com a tolerância de 5 minutos:

- **2.483 linhas** são tratadas como arredondamento: o alvo é **recortado** para
  ser exatamente a duração do episódio. A informação é preservada.
- **84 linhas** são erro real e são **removidas**, porque não dá para imputar um alvo.

Depois disso o `p_train` tem **749.916 linhas** e **zero** casos de escuta maior
que duração.

Uma observação que pode gerar pergunta: por que 2.567 e não os 2.568 da Fase 2?
Porque a correção de domínio acabou de rodar e transformou aquela duração ≤ 0 em
`NaN` — e comparação com `NaN` é falsa. A linha não sumiu, foi reclassificada como
valor ausente.

### 3.3 As regras de descarte do diagrama

[Mostrar o painel duplo de ausentes por coluna e por linha]

O diagrama previa duas regras: descartar coluna com mais de 50% de ausentes e
linha com mais de 25%.

**Nenhuma das duas dispara.** A coluna com mais ausentes tem 19,47% e a linha mais
vazia tem 2 células faltando em 12, ou seja 16,7%.

Executamos mesmo assim e registramos — porque uma regra que não dispara continua
sendo a política correta.

### 3.4 Política de imputação categórica

[Mostrar a tabela de políticas categóricas]

Para as categóricas, a equipe definiu uma política por coluna:

- **`Podcast_Name` e `Episode_Title`:** descartar a linha. Um registro sem o nome
  do programa ou o título é inútil — não há o que imputar.
- **`Genre` e `Episode_Sentiment`:** categoria explícita `Sem_classificacao`. Não
  inventamos um gênero, e a ausência vira algo mensurável.
- **`Publication_Day` e `Publication_Time`:** moda **do próprio podcast**. Dia e
  horário de publicação são um hábito do programa; a moda global ignoraria isso.

**O que os dados dizem sobre essa política:** nenhuma das seis colunas categóricas
tem um único valor ausente. **Zero linhas afetadas.**

[Mostrar a tabela do V de Cramér]

E fomos além: testamos se a moda por podcast **valeria a pena** se houvesse o que
imputar. Escondemos a coluna e medimos o acerto de cada estratégia.

O **V de Cramér** entre podcast e dia de publicação é **0,02** — ou seja, os
programas publicam em dias e horários praticamente **aleatórios**. A moda por
podcast acerta **0,31 ponto percentual** a mais que a moda global no dia, e 0,66
no horário.

A regra é a política correta — só não há estrutura neste dataset para ela
explorar. Foi mantida porque custa nada e protege contra outro dataset.

### 3.5 As duas colunas numéricas com ausentes — o achado central da Fase 3

[Mostrar a tabela de benchmark de imputação e o gráfico de barras com a linha do desvio padrão]

Esta é a parte mais importante da Fase 3.

`Episode_Length_minutes` tem 11,6% de ausentes e é a variável mais preditiva do
dataset. O diagrama sugeria "mediana ou KNN Imputer". A Fase 2 recomendou testar
KNN ou regressão. **Antes de escolher, medimos.**

O protocolo: pegamos só as linhas em que a coluna **é observada**, escondemos uma
fração igual à taxa real de ausência, imputamos e comparamos com o valor
verdadeiro. Testamos nove estratégias.

A referência é o **desvio padrão da própria coluna** — que é o RMSE que se obtém
simplesmente chutando a média. **Qualquer estratégia que não fique claramente
abaixo dele não está usando informação nenhuma.**

[Apontar a linha tracejada do gráfico]

Olhem o gráfico. **Todas as barras azuis encostam na linha tracejada.**

Para `Episode_Length_minutes`, o desvio padrão é **32,88** e o melhor RMSE honesto
é **32,77** — uma melhora de **0,3%** sobre o chute da média. Isso é ruído.

Agrupar por podcast, por gênero, por título, por horário: nada muda. E faz sentido
— a Fase 2.6 já tinha mostrado que nenhuma preditora se correlaciona com outra
acima de 0,06. **Não há de onde tirar informação.**

Para `Guest_Popularity_percentage` é ainda mais claro: **nenhuma** estratégia fica
abaixo do desvio padrão, nem a que usa o alvo.

Três consequências práticas:

**Primeira — o KNN Imputer do diagrama fica *pior* que a média.** RMSE de 35,87
contra 32,88: **9% de erro a mais**, aquela barra vermelha. Não é bug. O KNN mede
distância nas outras colunas numéricas, que aqui são ruído puro. Ele encontra
"vizinhos" arbitrários e importa a variância deles. Some-se a isso que o
`KNNImputer` é O(n²) e **nem roda** em 663 mil linhas — o ajuste precisou de uma
subamostra de 8 mil, o que já é, por si só, um argumento contra.

**Segunda — a única coisa que prevê a duração é o alvo.** Aquela barra laranja:
RMSE de 11,67, uma queda de 64%. Faz total sentido, a correlação é 0,917. E é
exatamente o que **não podemos usar**: o `test.csv` não tem a coluna alvo. Imputar
com o alvo no treino e com a mediana no teste criaria duas distribuições
diferentes e uma validação otimista que quebraria na Fase 5.

**Terceira — então usamos a mediana.** Entre as estratégias que empatam, é a mais
robusta, não inventa estrutura, custa O(n) e é trivial de aplicar identicamente ao
test.

### 3.6 O que a imputação custa

[Mostrar a tabela de R² e o teste t]

Mas o que se perde é real, e vale medir.

| cenário | n | R² |
|---|---|---|
| só casos completos | 662.822 | **0,8410** |
| todas as linhas, mediana | 749.916 | **0,7508** |
| todas as linhas, mediana + flag | 749.916 | 0,7514 |

**Nove pontos de R².** Nos casos completos a duração explica 84,1% da variância;
imputando os 11,6% ausentes com a mediana, cai para 75,1%.

**Esse é o maior custo isolado de toda a Fase 3, e ele é irrecuperável** — porque
a informação não existe em lugar nenhum do dataset.

E tem mais: a ausência **não é aleatória**. O alvo médio quando a duração está
ausente é **43,15 minutos**; quando está presente é **45,73**. Teste t de Welch com
p na ordem de 10 elevado a menos 165.

Linhas sem duração registrada têm alvo sistematicamente menor. **Isso é informação.**
Por isso criamos a flag `Length_Ausente` **antes** de imputar — depois a informação
sumiria.

[Mostrar o gráfico do pico artificial da mediana]

E aqui está a representação visual desses nove pontos: o pico na linha vermelha.
**87 mil episódios passam a ter exatamente a mesma duração**, 63,84 minutos. É por
isso que a flag precisa existir — para o modelo conseguir distinguir "durou 64
minutos" de "não sabemos quanto durou".

**Recomendação para a Fase 4:** modelos de árvore modernos — `HistGradientBoosting`,
LightGBM, XGBoost — tratam `NaN` nativamente e aprendem um caminho próprio para os
ausentes, em vez de recebê-los disfarçados de mediana. Vale treiná-los sobre a
matriz **sem imputar** e comparar. É a única rota que pode recuperar parte desses
nove pontos.

### 3.7 Outliers estatísticos

[Mostrar o painel antes/depois dos boxplots]

Executamos o bloco de IQR do diagrama, com os limites ajustados **só no train** e
aplicados sem recálculo ao test.

**As caixas são indistinguíveis, e isso é o resultado.** Depois da correção de
domínio, sobrou **uma linha** no train inteiro para o IQR recortar.

Confirma a recomendação da Fase 2: o tratamento estatístico de outliers é
supérfluo neste dataset. O trabalho de verdade foi feito pelas regras de domínio.

**Mas a etapa ficou no pipeline**, e por um motivo concreto: ela recortou **duas
linhas no test** — e são justamente as que importam. Foi ela, e não o IQR do train,
que impediu o episódio de 78 milhões de minutos de chegar ao escalonamento.

### 3.8 Engenharia de features

[Mostrar a tabela de features criadas]

Criamos **dez features derivadas**. As principais e a hipótese de cada uma:

- **`Episode_Num`** — o inteiro extraído de `Episode_Title`. A Fase 2.4 notou que
  o texto é "Episode N": uma coluna substitui cem.
- **`Ads_por_minuto`** — número de anúncios dividido pela duração. A hipótese é que
  a **densidade** de interrupção importa mais que a contagem absoluta.
- **`Length_x_Host_Pop`** — a interação sugerida pelo diagrama.
- **`Length_Ausente` e `Sem_Convidado`** — as flags de ausência.
- **`Is_Weekend`**, **`Publication_Time_ord`**, **`Sentimento_ord`**, **`Pop_Media`**
  e **`Pop_Diferenca`**.

[Mostrar o painel duplo: ganho de R² vs correlação]

E aqui está uma das lições mais bonitas do projeto. Avaliamos cada feature de duas
maneiras, e **elas discordam de propósito**:

- À direita, a **correlação com o alvo** — o quanto a feature se move junto com o alvo.
- À esquerda, o **ganho de R²** — o quanto ela acrescenta a um modelo que **já tem**
  as quatro numéricas originais.

**Os dois painéis contam histórias opostas, e o da esquerda é o verdadeiro.**

`Length_x_Host_Pop`, a interação sugerida pelo diagrama, tem correlação de
**+0,678** com o alvo — a segunda maior de todas. E acrescenta **+0,0004** de R².
Menos que uma flag binária.

O motivo é **aritmético**: ela é a duração multiplicada por outra coluna, então
herda a correlação de 0,917 **sem trazer informação nova**. É o caso didático de
por que a correlação isolada não serve para selecionar features.

`Ads_por_minuto` é o oposto: correlação de **−0,45**, menor em valor absoluto, e o
**maior ganho real** do conjunto, **+0,0008**. Ela mistura duas variáveis numa razão
que nenhuma das duas expressa sozinha.

`Length_Ausente` fica em segundo lugar com correlação de só −0,03 — confirmando a
decisão de transformar a ausência em variável.

E no fim da fila, `Pop_Media` e `Pop_Diferenca` têm ganho **negativo**: são funções
exatas de duas colunas que já estão no modelo.

O ganho total é modesto — **R² de 0,7577 para 0,7597** — e era esperado. A Fase 2
já tinha estabelecido que o problema é quase unidimensional.

### 3.9 O efeito dos anúncios

[Mostrar o gráfico da fração escutada e o mapa de calor]

A Fase 2 conjecturou que os anúncios teriam efeito **não linear** sobre a fração
escutada, e que isso seria o grande ganho dos modelos de árvore. Testamos direto.

| anúncios | fração média escutada | queda |
|---|---|---|
| 0 | 70,65% | — |
| 1 | 68,37% | −2,28 pp |
| 2 | 65,70% | −2,68 pp |
| 3 | 62,82% | −2,87 pp |

**A conjectura se confirma só de leve.** Cada anúncio derruba a fração escutada em
cerca de 2,6 pontos percentuais, e os quatro pontos ficam quase sobre a reta. Há
curvatura — o efeito **acelera** — mas é pequena.

O mapa de calor acrescenta a segunda metade da resposta: o efeito é **estável ao
longo das faixas de duração**. As linhas caem juntas, sem interação relevante entre
duração e anúncios — que era exatamente a não linearidade que valeria a pena.

**Isso reduz a expectativa de ganho das árvores sobre a regressão linear.** A Fase 4
deve medir o baseline linear primeiro e exigir dos modelos mais caros uma melhora
que justifique o custo.

### 3.10 Encoding

[Mostrar a tabela de comparação de encoding]

Quatro estratégias, escolhidas por cardinalidade:

- **Ordinal** para `Publication_Time` e `Episode_Sentiment` — eles têm ordem natural.
- **One-Hot** para `Genre` e `Publication_Day` — poucas categorias, com `drop='first'`
  para evitar a armadilha das dummies.
- **Target encoding** para `Podcast_Name` e `Episode_Title` — as de alta cardinalidade.

O target encoding substitui a categoria pela média do alvo naquela categoria, em
**uma coluna**. O risco é vazamento, porque a média inclui a própria linha. O
`TargetEncoder` do scikit-learn resolve com validação cruzada interna: no
`fit_transform` cada linha recebe a média calculada **sem ela**.

**O resultado justifica a escolha:** `Episode_Title_te` correlaciona **+0,074** com
o alvo, o equivalente a um R² de 0,0055 — praticamente o η² de 0,0058 que a Fase 2
mediu para a coluna inteira.

**Uma coluna recupera quase todo o sinal que 99 dummies capturariam.**

Duas colunas em vez de 146.

### 3.11 Normalização e escalonamento

[Mostrar o gráfico de assimetria com as três barras]

O diagrama pedia "StandardScaler ou MinMaxScaler, por modelo". São na verdade
**duas perguntas diferentes**, e a equipe separou bem as duas.

**Pergunta 1: normalizar, no sentido de transformar a forma da distribuição, é
necessário?**

**Não.** Depois da correção de domínio, nenhuma coluna original passa de
`|skew| = 0,35` — e o limite do projeto para assimetria relevante é 1,0. As
distribuições são quase uniformes, e uma uniforme já é **simétrica**. Não há
assimetria para corrigir.

E o `log1p` não só é desnecessário como **inverte o problema**:
`Episode_Length_minutes` sai de skew **+0,005** — praticamente perfeita — para
**−1,17**, uma cauda longa à esquerda que **não existia**. O log comprime os
valores altos e espalha os baixos, entortando uma distribuição que estava plana.

Uma exceção, e ela foi criada por nós: `Ads_por_minuto` tem skew de **+4,06**.
Testamos o `log1p` nela também — a assimetria mal mexe, de 4,06 para 3,51, porque
29% dos valores são zero e o log não separa zeros. E o R² muda na quinta casa
decimal. **Mantida sem transformar.**

[Mostrar a tabela de amplitudes]

**Pergunta 2: escalonar, no sentido de mudar a ordem de grandeza, é necessário?**

**Sim.** A amplitude da maior coluna dividida pela da menor é de **17.721 vezes**.
Num modelo que soma coeficientes vezes valores, ou que mede distância euclidiana,
uma variável que vai de 0 a 12 mil **domina** uma que vai de 0 a 0,67 — não por ser
mais importante, mas por ser numericamente maior.

[Mostrar o painel triplo: antes, MinMax, Standard]

E aqui fizemos um **ajuste ao raciocínio original da equipe**. A conclusão — usar
`MinMaxScaler` — está certa. Mas o motivo apresentado, de que ele "preserva a
distribuição uniforme e não força artificialmente uma curva normal", merece uma
correção.

**O `StandardScaler` também preserva.** Nenhum dos dois pode forçar normalidade,
porque **ambos são transformações afins** — da forma `x` vira `a` vezes `x` mais
`b`. A assimetria depois de escalonar é **idêntica**, e a correlação com a original
é **exatamente 1,000000**.

O `StandardScaler` divide pelo desvio padrão; ele não "normaliza" no sentido de
mudar a forma, apesar de o nome sugerir isso. Quem muda a forma é o
`PowerTransformer` — e já vimos que aqui ele atrapalha.

Os três painéis mostram exatamente isso: as curvas têm o **mesmo desenho** nos três
gráficos, só o eixo horizontal muda.

**Dito isso, o `MinMaxScaler` continua sendo a escolha certa**, por dois motivos que
os dados sustentam:

1. **As variáveis têm limites naturais rígidos.** Percentuais vivem em [0, 100],
   anúncios em [0, 3] — e a etapa 3.1 impôs esses limites explicitamente. Mapear
   para [0, 1] preserva o significado: 0,5 é literalmente "metade do máximo
   possível". Já em uma distribuição uniforme, "um desvio padrão acima da média"
   não é evento raro, é rotina.
2. **Compatibilidade com a Fase 5.** O alvo está restrito a [0, 120] e o plano prevê
   recortar as previsões nesse intervalo. Trabalhar em [0, 1] mantém o raciocínio de
   limites coerente de ponta a ponta.

### 3.12 A demonstração que amarra tudo

[Mostrar a tabela de comparação com e sem tratamento]

E aqui está a demonstração de por que a **ordem** das etapas importa.

O `MinMaxScaler` é o mais sensível a extremos de todos os escalonadores: **um único
valor absurdo no `fit` esmaga todo o resto contra o zero**. Por isso ele vem
**depois** da correção de domínio e do recorte de outliers.

Simulamos o que aconteceria sem as etapas 3.1 e 3.3:

Aquele episódio de 78 milhões de minutos viraria **241.318** na escala que deveria
ir de zero a um. E comprimiria todos os episódios reais num intervalo colado no
zero. `Number_of_Ads` chegaria a **19,85** por causa do valor 2.063.

Com o tratamento: de **3 linhas fora de [0,1]** para **1**.

E essa uma linha que sobra **não é falha**. Os limites vêm do train, e o test tem
uma duração ligeiramente abaixo do mínimo do train. Forçá-la para dentro exigiria
reajustar o escalonador no test — **isso sim seria vazamento**.

### 3.13 Seleção de features — o VIF

[Mostrar a tabela dos 10 pares mais correlacionados, depois a tabela de VIF]

A Fase 2 concluiu que "a seleção por VIF não vai eliminar nada" — e estava certa
**sobre as variáveis originais**. A engenharia de features mudou o quadro.

Primeiro a correlação par a par, que é a regra do diagrama: **zero pares com
correlação absoluta maior ou igual a 0,9**. O par mais correlacionado de todos é
0,75.

Pela regra do diagrama, **nada seria removido**.

Agora o **VIF**, o *Variance Inflation Factor*. Ele mede o quanto **todas as outras
features juntas** explicam uma dada feature. A fórmula é 1 dividido por 1 menos R².
A regra usual é: VIF maior que 10 é multicolinearidade séria.

[Apontar a tabela]

**Sete features com VIF infinito.** E uma com 12,5.

VIF infinito significa **redundância exata**: a feature é combinação linear das
outras, sem sobra. E ele acusa o **grupo** inteiro, não o culpado.

Há dois grupos, ambos criados pela nossa própria engenharia de features:

**Grupo 1 — as popularidades.** `Pop_Media` é a média de host e convidado;
`Pop_Diferenca` é a subtração. Duas quaisquer dessas quatro colunas determinam as
outras duas por álgebra elementar.

**Grupo 2 — o fim de semana.** `Is_Weekend` é **exatamente** a soma das dummies de
sábado e domingo. As três aparecem com VIF infinito — e a correlação par a par mais
alta entre elas é **0,66**.

[Pausa]

**A regra de correlação maior que 0,9 do diagrama jamais encontraria essa
identidade.** É o argumento de existência do VIF: uma feature pode ser combinação
de **várias** outras sem se parecer muito com nenhuma delas em particular.

E `Length_x_Host_Pop`, com VIF 12,5. Não é redundância exata — é um produto, não
uma soma — mas é o bastante para inflar o VIF da duração para 8,3. Já sabíamos que
ela acrescenta +0,0004 de R²; agora sabemos que também **cobra caro** por isso.

[Mostrar o gráfico antes/depois do VIF]

**A poda:** basta quebrar cada dependência, não eliminar o grupo inteiro. Saem
`Pop_Media`, `Pop_Diferenca`, `Is_Weekend` e `Length_x_Host_Pop`.

De **sete features com VIF infinito para zero**.

### 3.14 Importância por permutação

[Mostrar o gráfico de importância]

O VIF cuida da redundância; falta saber o que é **útil**.

A importância por permutação embaralha uma coluna de cada vez e mede o quanto o
erro do modelo piora. Se embaralhar não muda nada, a coluna não estava sendo usada.

**A escala do gráfico conta a história sozinha.**

Embaralhar `Episode_Length_minutes` piora o RMSE em **21,06 minutos** — num alvo
cujo RMSE é cerca de 13. A segunda colocada, `Ads_por_minuto`, custa **0,53
minuto**. **Quarenta vezes menos.**

É a Fase 2 confirmada num terceiro tipo de medida.

Três leituras menos óbvias:

**Primeira — `Number_of_Ads` aparece com importância negativa.** Embaralhá-la chega
a melhorar o modelo, dentro do ruído. Não é que anúncios não importem — a 3.4
mostrou que importam. É que `Ads_por_minuto` já carrega a informação numa forma
melhor. **A permutação mede o que é insubstituível, não o que é útil**: quando duas
colunas dizem a mesma coisa, embaralhar uma não dói.

**Segunda — o target encoding ganha do número extraído, e não é páreo.**
`Episode_Title_te` rende 0,028 e `Episode_Num` rende 0,0001, com desvio de 0,008 —
indistinguível de zero. As duas vêm da mesma coluna original: o efeito do título
**não é monotônico** no número do episódio.

**Terceira — 19 das 27 features têm efeito menor ou igual a 0,01 minuto**, abaixo de
um décimo de por cento do RMSE. As dummies de gênero e dia estão quase todas aí,
coerente com os η² de 0,0006 da Fase 2.

### 3.15 A decisão de seleção

[Mostrar a tabela de features removidas]

O diagrama trata a seleção como passo obrigatório. Aqui ela precisou de um critério,
porque há **duas famílias de modelos com necessidades opostas**:

- **Modelos lineares** sofrem com multicolinearidade — os coeficientes ficam
  instáveis e sem interpretação. Para eles, remover as de VIF alto é obrigatório.
- **Modelos de árvore** são indiferentes: eles simplesmente não escolhem essas
  colunas para dividir.

Como a Fase 4 vai comparar as duas famílias, a matriz final mantém as features com
sinal e remove **apenas as redundantes por construção**.

De **31 candidatas para 27 finais**.

E para dimensionar: se tivéssemos usado One-Hot em tudo, seriam **171 colunas** —
144 a mais, vindas só de `Podcast_Name` e `Episode_Title`.

### 3.16 Matrizes finais e verificação

[Mostrar a tabela de verificações]

Saem quatro artefatos da Fase 3, e a escolha entre eles é do modelo:

- **`X_train` e `X_test`** — sem escalonar, para modelos de árvore
- **`X_train_mm` e `X_test_mm`** — escalonados, para regressão linear, SVM e KNN

E rodamos onze verificações automáticas: sem `NaN`, mesmas colunas em train e test,
tudo numérico, alvo dentro de [0, 120], escalonamento dentro de [0, 1], test com
todas as linhas.

**Nove verificações obrigatórias, nove aprovadas.**

Resultado final: **X_train com 749.916 linhas e 27 colunas**; X_test com 250 mil
linhas e as mesmas 27 colunas. Removemos **84 linhas** do train — **0,011%** do
dataset.

Duas verificações são marcadas como **informativas, não obrigatórias**, e vale
explicar por quê:

- Nas linhas com duração **imputada**, o alvo pode passar da mediana. Isso é
  **esperado** — a mediana substituiu a duração real, então a comparação perde o
  sentido. Por isso a checagem obrigatória só olha as linhas com duração observada,
  e lá temos **zero violações**.
- O `X_test_mm` pode sair de [0, 1], porque os limites vêm do train. Forçá-lo a
  caber seria vazamento.

### 3.17 O ganho real da fase, sem maquiagem

[Mostrar a tabela de comparação final]

E fechamos medindo, com validação cruzada em 5 dobras, o que sai da Fase 3 contra
o que entrou:

| cenário | features | RMSE |
|---|---|---|
| cru + mediana (linear) | 4 | 13,367 |
| **Fase 3 escalonada (linear)** | 27 | **13,289** |
| cru + mediana (HistGB) | 4 | 13,153 |
| **Fase 3 (HistGB)** | 27 | **13,103** |

O RMSE do modelo linear cai **0,078 minuto**; o do HistGB, **0,051**.

**Isso é 0,6% e 0,4%.**

[Pausa]

E eu quero ser muito claro sobre como interpretar esse número: **não é decepção, é
confirmação.** Quando uma variável explica 84% do alvo e nenhuma outra passa de
0,6%, o pré-processamento **não tem de onde tirar mais**.

O valor real da Fase 3 está em outro lugar:

- Nos dados que ela **impediu** de chegar torto à Fase 4 — o episódio de 78 milhões
  de minutos, os percentuais acima de 100, os 2.063 anúncios.
- Nas **2.483 linhas válidas** que ela **não** descartou.
- Nas **sete redundâncias exatas** que ela removeu antes de quebrarem o modelo linear.

**A Fase 3 vale menos pelo que acrescenta do que pelo que impede de passar.**

[Mostrar o diário de transformações e a lista de arquivos salvos]

Tudo isso está registrado no **diário de transformações**, e os artefatos foram
salvos em `outputs/fase3`: as quatro matrizes em Parquet — que preserva os tipos e
ocupa dez vezes menos que CSV — mais cinco tabelas de decisão em CSV, para leitura
fora do Python.

---

## FECHAMENTO — 2 minutos

[Slide de conclusões]

Para fechar, os pontos que a gente gostaria que ficassem:

**Primeiro: o problema é quase unidimensional.** A duração do episódio explica 84%
da variância do alvo. Isso foi confirmado por três medidas independentes:
correlação de Pearson na Fase 2, ganho de R² na 3.4, e importância por permutação
na 3.7.

**Segundo: a imputação é o gargalo, e não tem conserto.** Nos casos completos a
duração explica 84,1%; depois de imputar, 75,1%. Nove pontos de R², e nenhuma das
nove estratégias testadas bate o desvio padrão da própria coluna. **Esses nove
pontos são o teto que a Fase 4 vai encontrar.**

**Terceiro: correlação não seleciona feature.** `Length_x_Host_Pop` tem a segunda
maior correlação com o alvo e acrescenta praticamente nada, com VIF 12,5. Foi
removida. `Ads_por_minuto`, com correlação menor, é a feature nova mais útil.

**Quarto: as ferramentas estatísticas precisam do contexto certo.** O IQR não
funciona em distribuição uniforme. A correlação par a par não vê identidade entre
três colunas. O `StandardScaler` não normaliza, apesar do nome. Em cada um desses
casos a gente executou a ferramenta do plano, mediu o resultado e documentou o
porquê.

### Os cinco ajustes que propomos ao plano original

1. **O `KNN Imputer` sai do plano.** Ele fica pior que a média e é O(n²) num dataset
   de 750 mil linhas. No lugar: mediana mais flag de ausência.
2. **A correção de domínio entra antes da imputação** — senão a mediana é calculada
   sobre valores corrompidos, e a regra "escuta maior que duração" deixa de ser
   verificável.
3. **O clipping por IQR vira secundário** — ele encontra uma linha no train. Quem
   faz o trabalho são as regras de domínio, e sempre inspecionando a **magnitude**
   da violação antes de decidir o tratamento.
4. **O K-Fold estratificado da Fase 4 vira K-Fold simples.** Estratificação é para
   classificação; com alvo contínuo, ou usamos `KFold` simples ou estratificamos
   por faixas do alvo.
5. **O RMSLE sai das métricas.** O alvo não tem cauda longa nem cobre várias ordens
   de grandeza. RMSE e MAE bastam.

### O que esperamos da Fase 4

Como o sinal linear é forte e as preditoras são independentes, **a regressão linear
deve chegar perto do teto do que é linearmente extraível**. Por isso vamos medir o
baseline linear **primeiro**, e exigir dos modelos mais caros uma melhora que
justifique o custo.

O ganho das árvores deve vir de duas coisas: a **restrição de teto** — o alvo nunca
excede a duração do episódio — e o tratamento **nativo de `NaN`**, que é a única
rota que pode recuperar parte dos nove pontos de R² que a imputação custou.

Obrigado. Estamos abertos a perguntas.

---

## APÊNDICE — Perguntas prováveis e respostas prontas

**"Por que vocês não usaram o KNN Imputer, se estava no plano?"**
Nós usamos — e medimos. Ele ficou com RMSE de 35,87 contra 32,88 da média simples,
ou seja, 9% pior. O motivo é que o KNN mede distância nas outras colunas numéricas,
e neste dataset elas não se correlacionam com nada — a maior correlação entre
preditoras é 0,06. Ele encontra vizinhos arbitrários e importa a variância deles.
Além disso é O(n²): nem roda em 663 mil linhas, precisamos de uma subamostra de
8 mil só para o benchmark.

**"O ganho de 0,6% não é muito pouco para todo esse trabalho?"**
É pouco em RMSE, e isso é esperado quando uma variável explica 84% do alvo. Mas o
ganho em RMSE não é a única entrega da fase. Sem a Fase 3, um episódio de 78
milhões de minutos entraria no `MinMaxScaler` e comprimiria todos os dados reais
contra o zero. E uma regra ingênua teria descartado 2.483 linhas válidas. A fase
vale pelo que ela impede.

**"Por que a tolerância é 5 minutos e não outro valor?"**
Ela vem da tabela de sensibilidade da Fase 2.6. Em 5 minutos, 96,7% das violações
já foram absorvidas como arredondamento, e o que sobra são 85 linhas com excesso
que chega a 115 minutos — casos qualitativamente diferentes. Entre 5 e 10 minutos a
tabela mal se mexe: 84 contra 62 linhas. Então 5 é o ponto onde a curva já achatou.

**"Vocês testaram remover as features de baixa importância?"**
Não removemos, e por uma decisão consciente. Modelos de árvore são indiferentes a
features inúteis — simplesmente não as escolhem para dividir. Como a Fase 4 vai
comparar árvores com modelos lineares, mantivemos tudo que tem sinal e removemos só
o que é redundante **por construção**, que é o que quebra o modelo linear.

**"O target encoding não vaza informação do alvo?"**
Vazaria, se fosse feito de forma ingênua, porque a média da categoria incluiria a
própria linha. O `TargetEncoder` do scikit-learn resolve com validação cruzada
interna: no `fit_transform`, cada linha recebe a média calculada **sem ela**. E o
`transform` do test usa as médias do train inteiro, sem reajuste.

**"Por que vocês mantiveram políticas de imputação que afetam zero linhas?"**
Por três motivos. Primeiro, a etapa 3.1 **cria** ausentes novos, e o mesmo código
roda de novo. Segundo, o test é comprovadamente mais sujo que o train, e nada
garante que uma versão futura mantenha as colunas limpas. Terceiro, uma política
que nunca dispara continua sendo a política correta.

**"O alvo pode ser previsto fora do intervalo [0, 120]?"**
Pode, e é uma coisa que registramos para a Fase 5. Nada impede um modelo de
regressão de prever valor negativo ou acima de 120. A recomendação é recortar as
previsões nesse intervalo — e é parte do motivo de termos escolhido o
`MinMaxScaler`, para manter o raciocínio de limites coerente de ponta a ponta.
