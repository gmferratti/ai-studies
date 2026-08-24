# Aprendizado Supervisionado e Não Supervisionado

Revisão de algoritmos clássicos de classificação, comparados sobre o mesmo
dataset e o mesmo split de treino/teste (ver `utils/data_utils.py`).

## Algoritmos

- [X] Árvore de decisão (`decision_tree.py`)
- [X] k-NN (`knn.py`)
- [ ] SVM (`svm.py`)
- [ ] Naive Bayes (`naive_bayes.py`)

## Comparação de resultados

Precisão, recall e F1 são da classe "fraude" (a rara), não da média geral:
num dataset com 0,17% de fraude, acurácia sozinha não diz muita coisa.

| Algoritmo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Árvore de decisão | | | | |
| k-NN (K=3, uniform) | 0,9996 | 0,9101 | 0,8265 | 0,8663 |
| SVM | | | | |
| Naive Bayes | | | | |

## Notas e aprendizados

- k-NN: testado com K=3 e K=11 (voto simples) e K=11 com voto ponderado por
  distância. K=3 deu o melhor F1 na classe fraude (0,8663); K=11 uniform foi
  o pior (recall caiu pra 0,7449), confirmando na prática o que a teoria
  prevê: K grande demais dilui o voto da classe rara. O voto ponderado por
  distância recuperou parte da perda do K=11 (F1 subiu de 0,8156 pra
  0,8380), mas não superou o K=3 puro neste dataset.
