# Árvores

- Árvores de decisão (para classificação)
- Árvores de regressão (para regressão)
- Dividir para conquistar. Problemas complexos fragmentados em problemas mais simples resolvidos recursivamente.

## Vocabulário básico

- Nó interno: uma pergunta sobre um atributo.
- Ramo: uma resposta possível daquela pergunta.
- Nó folha: a decisão final (classe prevista).
- Indução top-down (TDIDT = Top-Down Induction of Decision Trees): a árvore nasce da raiz e cresce pras folhas, escolhendo em cada nó a pergunta que mais separa o grupo.
  - "Top-down": começa do topo (raiz, com todos os exemplos juntos) e desce, nunca o contrário.
  - "Indução": aprende uma regra geral a partir de exemplos particulares de treino (o oposto de dedução, que parte de regras já dadas).
- Algoritmo guloso (greedy): escolhe a melhor pergunta NAQUELE nó, sem reconsiderar depois. Não garante a árvore globalmente ótima, pode convergir para mínimos locais.

## Critérios de divisão (decorar as fórmulas)

- Entropia: `H(S) = - Σ p_i * log2(p_i)`. Mede a "dúvida" do grupo. 0 = grupo puro. 1 = 50/50 no caso binário (máxima incerteza).
- Ganho de informação: `Ganho(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v)`. A árvore escolhe o atributo com maior ganho. Usado no ID3 e no C4.5.
  - `S_v` = o pedaço de S onde o atributo A deu o valor v (ex.: A = "Tipo", v = "Fogo" -> S_v é só os Pokémons do Tipo Fogo). É o mesmo "subgrupo" ou "galho" que aparece nas contas do `decision_tree.py`.
  - `|S_v|/|S|` = quantos exemplos caíram nesse galho, em proporção do total. É o peso usado pra fazer a média ponderada das entropias dos filhos.
- Índice Gini: `Gini(S) = 1 - Σ p_i²`. Também 0 no grupo puro, mas o máximo no caso binário é 0,5 (não 1, como a entropia). Usado no CART, é o padrão do scikit-learn.
- Split Info: `SplitInfo(S, A) = - Σ (|S_v|/|S|) * log2(|S_v|/|S|)`. Repara que é a MESMA fórmula da entropia, só que aplicada aos TAMANHOS dos galhos em vez de às classes. Ou seja: mede o quão "espalhado" o atributo A divide o grupo. Atributo com muitos valores distintos (tipo um "ID do cliente", quase único por linha) gera Split Info alto.
- Gain Ratio: `GainRatio(S, A) = Ganho(S, A) / SplitInfo(S, A)`. Correção do C4.5 pro viés do Ganho de Informação, que sozinho favorece atributos com muitos valores distintos (um atributo que separa tudo em grupos minúsculos parece ótimo pelo Ganho, mas generaliza mal). Dividir pelo Split Info penaliza esse excesso de fragmentação.
- Pegadinha de prova: entropia e Gini concordam quase sempre sobre qual atributo escolher, mas têm escalas diferentes (1 vs 0,5 no máximo binário).

## Os três algoritmos clássicos

| Algoritmo | Ano | Critério | Atributos | Poda | Valores faltantes |
|---|---|---|---|---|---|
| ID3 | 1986 (Quinlan) | Ganho de informação | só categóricos | não | não trata |
| C4.5 | 1993 (Quinlan) | Gain Ratio | categóricos e contínuos | pós-poda | trata |
| CART | 1984 (Breiman et al.) | Gini (classif.) / erro quadrático (regr.) | categóricos e contínuos, sempre árvore binária | pós-poda (cost-complexity) | trata |

- CART é o que o scikit-learn implementa (`DecisionTreeClassifier`, `DecisionTreeRegressor`).

## Overfitting e poda

- Árvore sem limite tende a decorar o treino (inclusive ruído): overfitting, alta variância.
- Pré-poda (early stopping): limita o crescimento durante a construção. Parâmetros no scikit-learn: `max_depth`, `min_samples_leaf`, `min_samples_split`.
- Pós-poda: deixa crescer tudo e depois corta os galhos que não compensam. No scikit-learn: `ccp_alpha` (cost-complexity pruning), calculável via `cost_complexity_pruning_path`.

## Aspectos Positivos

- Fácil de interpretar (modelo caixa branca, dá pra desenhar o fluxograma).
- Não exige normalização/escalonamento dos atributos.
- Lida naturalmente com atributos numéricos e categóricos juntos.

## Aspectos Negativos

- Instável: pequena mudança nos dados de treino pode gerar árvore bem diferente (alta variância).
- Propensa a overfitting se não for limitada ou podada.
- Gulosa, não garante a árvore ótima globalmente.
- Em dataset desbalanceado, tende a favorecer a classe majoritária (relevante no nosso dataset de fraude).
- Random Forest e Gradient Boosting existem justamente pra compensar essa instabilidade, combinando várias árvores.

## Ver também

- `decision_tree/decision_tree.py`: contas feitas na mão (entropia, Gini, ganho de informação) com o exemplo dos Pokémons, e o treino de verdade no dataset de fraude.
