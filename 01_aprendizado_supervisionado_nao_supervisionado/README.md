# Aprendizado Supervisionado e Não Supervisionado

Revisão de algoritmos clássicos de classificação, comparados sobre o mesmo
dataset e o mesmo split de treino/teste (ver `utils/data_utils.py`).

## Algoritmos

- [X] Árvore de decisão (`decision_tree.py`)
- [X] k-NN (`knn.py`)
- [ ] SVM (`svm.py`)
- [X] Naive Bayes (`naive_bayes.py`)

## Comparação de resultados

Precisão, recall e F1 são da classe "fraude" (a rara), não da média geral:
num dataset com 0,17% de fraude, acurácia sozinha não diz muita coisa.

| Algoritmo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Árvore de decisão | | | | |
| k-NN (K=3, uniform) | 0,9996 | 0,9101 | 0,8265 | 0,8663 |
| SVM | | | | |
| Naive Bayes (GaussianNB, var_smoothing=1e-9) | 0,9764 | 0,0588 | 0,8469 | 0,1099 |

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
  `bayes/images/densidades_gaussianas.png`), então o modelo classifica
  fraude demais como positivo, incluindo muito falso positivo, só pra não
  deixar passar as fraudes de verdade. Variar `var_smoothing` de 1e-9 pra
  1e-2 (o equivalente contínuo da suavização de Laplace) quase não mudou o
  resultado (F1 de 0,1099 pra 0,1109), sinal de que o problema aqui não é
  divisão por variância zero, é a suposição Gaussiana em si não encaixar
  nos dados.
