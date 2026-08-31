# Aprendizado Supervisionado

Revisão de algoritmos clássicos de classificação, comparados sobre o mesmo
dataset e o mesmo split de treino/teste (ver `utils/data_utils.py`).

## Algoritmos

- [X] Árvore de decisão (`decision_tree/decision_tree.py`)
- [X] k-NN (`knn/knn.py`)
- [X] SVM (`svm/svm.py`)
- [X] Naive Bayes (`naive_bayes/naive_bayes.py`)
- [X] Bagging (`bagging/bagging.py`)
- [X] Boosting (`boosting/boosting.py`)
- [X] Random Forest (`random_forest/random_forest.py`)

## Comparação de resultados

Precisão, recall e F1 são da classe "fraude" (a rara), não da média geral:
num dataset com 0,17% de fraude, acurácia sozinha não diz muita coisa.

| Algoritmo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Árvore de decisão | | | | |
| k-NN (K=3, uniform) | 0,9996 | 0,9101 | 0,8265 | 0,8663 |
| SVM (LinearSVC, C=1) | 0,9991 | 0,8286 | 0,5918 | 0,6905 |
| Naive Bayes (GaussianNB, var_smoothing=1e-9) | 0,9764 | 0,0588 | 0,8469 | 0,1099 |
| Bagging (BaggingClassifier, 100 árvores) | 0,9996 | 0,9213 | 0,8367 | 0,8770 |
| Boosting (AdaBoostClassifier, stumps, 150 rodadas) | 0,9991 | 0,7347 | 0,7347 | 0,7347 |
| Random Forest (RandomForestClassifier, max_features='sqrt', 100 árvores) | 0,9996 | 0,9412 | 0,8163 | 0,8743 |

## Notas e aprendizados

- k-NN: testado com K=3 e K=11 (voto simples) e K=11 com voto ponderado por
  distância. K=3 deu o melhor F1 na classe fraude (0,8663); K=11 uniform foi
  o pior (recall caiu pra 0,7449), confirmando na prática o que a teoria
  prevê: K grande demais dilui o voto da classe rara. O voto ponderado por
  distância recuperou parte da perda do K=11 (F1 subiu de 0,8156 pra
  0,8380), mas não superou o K=3 puro neste dataset.
- Naive Bayes (GaussianNB): recall bem alto (0,8469, na mesma faixa do
  k-NN) mas precisão baixíssima (0,0588), o pior F1 da tabela até agora.
  Faz sentido pela suposição que o algoritmo assume: os atributos do
  dataset de fraude (componentes de PCA) não seguem uma curva de sino
  limpa dentro de cada classe, a classe fraude em particular fica bem mais
  espalhada e "torta" que uma Gaussiana (visível em
  `naive_bayes/images/densidades_gaussianas.png`), então o modelo classifica
  fraude demais como positivo, incluindo muito falso positivo, só pra não
  deixar passar as fraudes de verdade. Variar `var_smoothing` de 1e-9 pra
  1e-2 (o equivalente contínuo da suavização de Laplace) quase não mudou o
  resultado (F1 de 0,1099 pra 0,1109), sinal de que o problema aqui não é
  divisão por variância zero, é a suposição Gaussiana em si não encaixar
  nos dados.
- SVM: o resultado da tabela é do `LinearSVC` (kernel linear, treinado no
  conjunto completo). Precisão alta (0,8286) mas recall mediano (0,5918),
  o oposto do padrão de k-NN e Naive Bayes nesta tabela: quando o SVM
  aponta fraude, geralmente acerta, mas deixa passar bastante fraude de
  verdade. Faz sentido pela forma de treinar: `LinearSVC` maximiza a
  margem no espaço original, sem nenhum ajuste específico pra classe rara
  ficar mais fácil de acertar, então a fronteira linear tende a ficar do
  lado "seguro" (a favor da classe majoritária). Testei também um `SVC`
  com kernel RBF, mas só numa amostra estratificada (todas as fraudes do
  treino mais algumas milhares de transações normais, não o treino
  inteiro): nessa amostra o RBF teve recall bem mais alto (0,8673) e F1
  um pouco melhor (0,7658), mas não é um resultado comparável de verdade
  com o resto da tabela, porque treinou em muito menos dado normal. Serve
  mais pra mostrar a ideia de kernel não linear e o custo computacional
  do RBF (ver `svm/svm.py`) do que como número final da classe SVM neste
  dataset.
- Bagging: o F1 da classe fraude (0,8770) já supera todos os algoritmos
  individuais da tabela, esperado, já que o comitê inteiro decide, não uma
  única árvore. O ganho mais interessante não é esse número final, é a
  redução de variância: treinando uma única árvore em cada amostra bootstrap
  do treino (o equivalente a "um investigador sozinho") e comparando com o
  comitê inteiro, em 5 sementes diferentes, o F1 do investigador sozinho
  variou bastante entre sementes (desvio-padrão 0,0152) enquanto o do comitê
  quase não se mexeu (desvio-padrão 0,0051), quase 3x menos, o efeito de
  reduzir variância sem reduzir viés acontecendo com dados de verdade (ver
  `bagging/bagging.py` e `bagging/images/comparacao_variancia.png`). Pegadinha
  que caiu na hora de montar essa comparação: trocar só o `random_state` de
  uma árvore treinada sempre no MESMO conjunto de treino não mede variância
  nenhuma, porque com atributos contínuos a árvore quase nunca tem empate pra
  desempatar; a instabilidade de verdade só aparece reamostrando o próprio
  conjunto de treino. A acurácia estimada por out-of-bag (0,9995) também ficou
  bem colada na acurácia medida de fato no teste (0,9996), confirmando que o
  erro OOB funciona como uma prévia gratuita do desempenho em dados novos.
- Boosting: F1 da classe fraude (0,7347) ficou abaixo do bagging (0,8770) e
  do k-NN (0,8663) nesta tabela, apesar do AdaBoost ter 150 rodadas contra
  as 100 árvores do bagging. Não é falha de implementação, é o próprio
  `staged_predict` rodada a rodada mostrando o motivo: o F1 de treino sobe
  quase sem parar até a última rodada (0,7229 -> 0,7622), enquanto o F1 de
  teste sobe rápido nas primeiras ~40 rodadas e depois oscila estacionado
  em torno de 0,73-0,745, sem acompanhar o treino (ver
  `boosting/images/curva_f1_rodada.png`), a assinatura visual do overfitting
  que a teoria prevê pro boosting com rodadas demais. No exemplo de
  brinquedo (torneio de lutadores) o padrão contrário e complementar
  aparece: cada especialista é fraco sozinho (stump de profundidade 1,
  errando exatamente 1 lutador cada), mas o comitê ponderado das 3 rodadas
  chega a 0% de erro no próprio torneio, reduzindo viés de verdade. As duas
  pontas (viés caindo rápido no brinquedo, variância crescendo devagar no
  dataset de fraude) são a mesma moeda: por isso XGBoost e afins existem,
  pra domar essa variância crescente sem abrir mão do ganho de viés (ver
  `boosting/notes/anotacoes.md`).
- Random Forest: no exemplo de brinquedo (com um atributo campeão
  disparado de propósito), a decorrelação apareceu bem nítida: bagging
  puro cravou o mesmo atributo na raiz das 9 árvores (concordância 100%
  entre pares), enquanto o random forest, escondendo esse atributo em
  1/3 das divisões, variou a raiz (6 armadura, 2 agilidade, 1 sorte) e a
  concordância caiu pra 93,1% (ver `random_forest/images/concordancia_bagging_vs_rf.png`).
  Curiosamente, esse ganho de decorrelação NÃO se traduziu em F1 melhor
  no dataset de fraude de verdade: `max_features=None` (0,8877) bateu
  tanto `max_features='sqrt'` (0,8743, o valor escolhido como final da
  tabela) quanto `max_features=3` (0,8729). Não é bug, é o esperado
  quando não existe um atributo "campeão disparado" pra decorrelacionar:
  os 30 componentes de PCA do dataset já vêm razoavelmente decorrelacionados
  entre si, então restringir `max_features` só custa viés (cada árvore fica
  um pouco pior) sem comprar decorrelação suficiente pra compensar, o
  contraponto real e honesto ao exemplo de brinquedo (ver
  `random_forest/notes/anotacoes.md`, seção do trade-off de `max_features`).
  A importância de atributos do modelo final apontou V17, V14 e V12 como
  os três mais usados pelas árvores pra reduzir impureza, batendo com o
  que é comumente citado sobre esse dataset na literatura.
