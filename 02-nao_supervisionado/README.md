# Aprendizado Não Supervisionado

Algoritmos que buscam estrutura em dados sem rótulo: agrupamento, regras de
associação e redução de dimensionalidade.

## Algoritmos

- [X] Clustering hierárquico (`hierarquico/hierarquico.py`)
- [X] K-means (`kmeans/kmeans.py`)
- [X] Regras de associação: Apriori (`apriori/apriori.py`)
- [X] Redução de dimensionalidade: PCA (`pca/pca.py`)

## Comparação de resultados

Tarefas diferentes (agrupar, achar regras, reduzir dimensão) usam métricas
diferentes, não uma tabela única de acurácia/precisão/recall/F1 como na
família supervisionada. Os dois algoritmos de clustering treinaram com
K=2 (fraude ou não) e são comparados contra a classe real só por
curiosidade pedagógica (o algoritmo nunca viu esse rótulo durante o
treino):

| Algoritmo | Amostra de treino | Silhueta | ARI (vs. classe real) |
|---|---|---|---|
| Clustering hierárquico (linkage=ward) | 1.000 linhas (492 fraudes + 508 normais) | 0,6322 | 0,0515 |
| K-means (K=2, k-means++) | 227.845 linhas (treino inteiro) | 0,0662 | -0,0000 |

PCA é redução de dimensionalidade, não agrupamento, então a métrica que
importa é quanta variância cada componente carrega, não silhueta nem ARI:

| Componentes | Variância acumulada (dataset de fraude, 30 atributos) |
|---|---|
| 1 | 6,5% |
| 2 | 12,1% |
| 27 | 95,0% (nº escolhido pelo limiar de 95%) |
| 30 (todos) | 100% |

Apriori não classifica nem agrupa, então a métrica não é acurácia nem
silhueta, é suporte/confiança/lift de cada regra de associação. O
dataset contínuo foi discretizado em faixas (valor da compra, período do
dia, e os 3 atributos V mais correlacionados com fraude) pra virar uma
"cesta de itens"; a tabela abaixo mostra só as regras cujo consequente é
`Classe_fraude`, ordenadas por lift:

| Antecedente | Suporte | Confiança | Lift |
|---|---|---|---|
| V12=baixo + V14=baixo + V17=baixo | 0,0013 | 0,0147 | 8,505 |
| V12=baixo + V17=baixo | 0,0014 | 0,0064 | 3,729 |
| V14=baixo + V17=baixo | 0,0013 | 0,0064 | 3,687 |
| V12=baixo + V14=baixo | 0,0016 | 0,0063 | 3,628 |
| V12=baixo | 0,0017 | 0,0033 | 1,915 |
| V14=baixo | 0,0017 | 0,0033 | 1,911 |
| V17=baixo | 0,0014 | 0,0027 | 1,589 |

## Notas e aprendizados

- Clustering hierárquico: comparando os quatro tipos de linkage na mesma
  amostra, `average` teve a MAIOR silhueta (0,7442, o cluster com a
  aparência geométrica mais "limpa") mas ARI perto de zero (0,0001,
  nenhuma relação com fraude de verdade); `ward` teve silhueta menor
  (0,6322) mas foi o único a separar uma fatia razoável de fraude num
  cluster à parte (106 das 492 fraudes da amostra caíram sozinhas num
  cluster de 106 linhas, as outras 386 ficaram misturadas com as 508
  normais). Prova concreta de que uma métrica interna (silhueta, que não
  usa rótulo nenhum) não garante nada sobre bater com uma estrutura
  externa que por acaso você já conhece: os dois medem coisas diferentes
  (ver `hierarquico/notes/anotacoes.md`). Precisou treinar numa amostra de
  1.000 linhas, não no dataset inteiro, porque o custo é O(n²): rodar nas
  ~285 mil transações inteiras estouraria memória.
- K-means: rodou no conjunto de treino inteiro (K-means escala bem,
  O(n·K·iterações), diferente do hierárquico) e a curva do método do
  cotovelo (`kmeans/images/curva_cotovelo.png`) desce suave, sem cotovelo
  nítido nenhum, sinal de que os 30 componentes de PCA do dataset não têm
  uma estrutura de clusters óbvia esperando ser encontrada. Com K=2, o
  resultado confirma isso: ARI praticamente zero (-0,0000) e silhueta
  baixa (0,0662), o k-means dividiu o espaço ao meio por alguma direção
  de maior variância, sem relação nenhuma com fraude (a tabela cruzada
  mostra as 394 fraudes do treino espalhadas quase 50/50 entre os dois
  clusters). É um resultado honesto, não um bug: clustering não
  supervisionado agrupa pelo que domina a VARIÂNCIA dos dados, e nada
  garante que "ser fraude" seja o eixo de maior variância; ao contrário
  de k-NN ou árvore de decisão, que aprendem a fronteira certa porque
  VEEM o rótulo fraude/normal durante o treino, o k-means nunca teve
  acesso a essa informação.
- PCA: a curva de variância explicada no dataset de fraude fica quase
  reta, sem cotovelo nítido nenhum (`pca/images/variancia_explicada_fraude.png`):
  são precisos 27 dos 30 componentes pra reter 95% da variância. Não é
  falha do PCA, é esperado: as colunas V1 a V28 desse dataset já são o
  resultado de um PCA que o próprio Kaggle aplicou antes de publicar os
  dados (anonimização), então já chegam razoavelmente decorrelacionadas
  entre si, sobra pouca redundância pra um segundo PCA explorar (o mesmo
  motivo, aliás, que fez `max_features` não ajudar o Random Forest nesse
  dataset, ver `01-supervisionado/README.md`). No exemplo de brinquedo
  (`pca/images/variancia_explicada_rpg.png`), com dois atributos
  propositalmente correlacionados (Força e Resistência de 8 personagens
  de RPG), o cotovelo aparece nítido: PC1 sozinho já explica 97,4% da
  variância, e a ordem dos personagens nessa única coordenada já separa
  sozinha quem é "tanque" de quem não é.
- Apriori: os 3 atributos escolhidos automaticamente por correlação com
  `Class` (V17, V14, V12) são exatamente os três que o Random Forest
  apontou como mais importantes em `01-supervisionado/README.md`, dois
  algoritmos bem diferentes concordando em quais componentes de PCA
  carregam o sinal de fraude. A confiança absoluta de toda regra
  encontrada é baixíssima (a maior é 1,47%), esperado quando o
  consequente é um evento raro (fraude é só 0,17% das transações): nunca
  vai existir uma regra "confiante" em valor absoluto pra prever algo
  raro assim. O que importa é o lift, e ali aparece um resultado nítido
  (`apriori/images/lift_regras_fraude.png`): as três faixas sozinhas
  (V12, V14 ou V17 baixos) já multiplicam a chance de fraude por cerca de
  1,6 a 1,9 vezes, mas juntar as três num único itemset multiplica por
  8,5 vezes, bem mais que a soma das partes. É a mesma lógica por trás de
  por que uma árvore de decisão ganha ao combinar várias perguntas em
  sequência: nenhum atributo sozinho separa bem a fraude, mas a
  interseção dos três aperta o cerco.
