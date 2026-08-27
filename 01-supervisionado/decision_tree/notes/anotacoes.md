# Árvores

- Árvores de decisão (para classificação)
- Árvores de regressão (para regressão)
- Dividir para conquistar. Problemas complexos fragmentados em problemas mais simples resolvidos recursivamente.

## Analogia central: o Chapéu Seletor de Hogwarts

Ele não escaneia a alma inteira de uma vez e já cospe "Grifinória!". Vai sondando aos poucos: "você valoriza mais coragem ou esperteza?", "prefere agir ou planejar?". Cada resposta elimina possibilidades, até sobrar só uma casa. Uma árvore de decisão é exatamente isso, só que aplicada a dados: uma sequência de perguntas organizadas de um jeito que chega numa resposta o mais rápido possível.

Pra classificar um caso novo, começa-se no topo (raiz) e vai-se respondendo pergunta por pergunta até cair numa folha. Esse caminho é literalmente um fluxograma: sim leva pra um lado, não leva pro outro.

## Vocabulário básico

- Nó interno: uma pergunta sobre um atributo.
- Ramo: uma resposta possível daquela pergunta.
- Nó folha: a decisão final (classe prevista).
- Indução top-down (TDIDT = Top-Down Induction of Decision Trees): a árvore nasce da raiz e cresce pras folhas, escolhendo em cada nó a pergunta que mais separa o grupo.
  - "Top-down": começa do topo (raiz, com todos os exemplos juntos) e desce, nunca o contrário.
  - "Indução": aprende uma regra geral a partir de exemplos particulares de treino (o oposto de dedução, que parte de regras já dadas).
- Algoritmo guloso (greedy): escolhe a melhor pergunta NAQUELE nó, sem reconsiderar depois. Não garante a árvore globalmente ótima, pode convergir para mínimos locais. É como alguém jogando um jogo de perguntas e respostas sem poder desfazer uma pergunta já feita.

## Como a árvore é construída

O "Chapéu Seletor" da árvore não nasce sabendo quais perguntas fazer. Ele aprende, olhando pros exemplos de treino, qual é a melhor pergunta pra começar, depois a melhor pra cada ramificação, e assim por diante:

1. Olha pro grupo de exemplos que chegou até aqui. Todos são da mesma classe? Vira uma folha (decisão pronta, não precisa perguntar mais nada).
2. Se não, testa todas as perguntas possíveis e escolhe a que mais separa o grupo em subgrupos mais parecidos entre si (menos misturados).
3. Divide o grupo pelas respostas dessa pergunta e repete o processo dentro de cada subgrupo, do zero.

Ver esse processo acontecendo de verdade, com duas perguntas em sequência (Tipo e Já evoluiu?), no exemplo dos Pokémons em `decision_tree.py`.

## Critérios de divisão (decorar as fórmulas)

Ideia central, numa frase: a árvore prefere a pergunta que deixa os subgrupos resultantes mais "arrumados" (menos misturados). Imagine dois potinhos de bala: um com 20 balas todas de morango (bem arrumado) e outro com 20 balas de 10 sabores diferentes (uma bagunça). Impureza é justamente essa medida de bagunça de um grupo: zero num grupo de uma classe só, máxima num grupo 50/50. Entropia e Gini, abaixo, são só duas réguas diferentes pra medir essa bagunça, e no fim quase sempre concordam sobre qual pergunta é a melhor.

- Entropia: `H(S) = - Σ p_i * log2(p_i)`. Mede a "dúvida" do grupo. 0 = grupo puro. 1 = 50/50 no caso binário (máxima incerteza).

  Pensa assim: já jogou Akinator, aquele gênio de site que adivinha em quem você está pensando fazendo perguntas de sim ou não? Entropia é basicamente "quantas perguntas de sim/não o Akinator precisaria, em média, pra descobrir a classe certa dentro desse grupo, jogando do jeito mais esperto possível". Cada termo `-log2(p_i)` mede a "surpresa" de ver a classe i: quanto mais raro um evento, mais surpreendente ele é quando acontece (ninguém se surpreende com "o sol nasceu hoje"; todo mundo se surpreende com "nevou no Rio de Janeiro"). Em bits, a surpresa sobe exatamente 1 bit cada vez que a chance do evento cai pela metade:

  | chance | surpresa |
  |---|---|
  | 100% | 0 bit |
  | 50% | 1 bit |
  | 25% | 2 bits |
  | 12,5% | 3 bits |

  A entropia é a MÉDIA dessa surpresa dentro do grupo, ponderada pela própria chance de cada classe aparecer.

- Ganho de informação: `Ganho(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v)`. A árvore escolhe o atributo com maior ganho. Usado no ID3 e no C4.5. É simplesmente "quanto a dúvida (entropia) diminuiu depois que eu fiz essa pergunta": a árvore testa essa conta pra cada pergunta candidata e escolhe a que derruba mais a dúvida.
  - `S_v` = o pedaço de S onde o atributo A deu o valor v (ex.: A = "Tipo", v = "Fogo" -> S_v é só os Pokémons do Tipo Fogo). É o mesmo "subgrupo" ou "galho" que aparece nas contas do `decision_tree.py`.
  - `|S_v|/|S|` = quantos exemplos caíram nesse galho, em proporção do total. É o peso usado pra fazer a média ponderada das entropias dos filhos.
- Índice Gini: `Gini(S) = 1 - Σ p_i²`. Também 0 no grupo puro, mas o máximo no caso binário é 0,5 (não 1, como a entropia). Usado no CART, é o padrão do scikit-learn. Outra régua de bagunça, com leitura diferente: é a chance de errar a classe se alguém pegasse uma bala aleatória do potinho e apostasse um rótulo sorteado seguindo a mesma mistura do potinho. É mais rápido de calcular que a entropia (não usa logaritmo) e, na prática, quase sempre concorda com ela.
- Split Info: `SplitInfo(S, A) = - Σ (|S_v|/|S|) * log2(|S_v|/|S|)`. Repara que é a MESMA fórmula da entropia, só que aplicada aos TAMANHOS dos galhos em vez de às classes. Ou seja: mede o quão "espalhado" o atributo A divide o grupo. Atributo com muitos valores distintos (tipo um "ID do cliente", quase único por linha) gera Split Info alto.
- Gain Ratio: `GainRatio(S, A) = Ganho(S, A) / SplitInfo(S, A)`. Correção do C4.5 pro viés do Ganho de Informação, que sozinho favorece atributos com muitos valores distintos (um atributo que separa tudo em grupos minúsculos parece ótimo pelo Ganho, mas generaliza mal). Dividir pelo Split Info penaliza esse excesso de fragmentação.
- Pegadinha de prova: entropia e Gini concordam quase sempre sobre qual atributo escolher, mas têm escalas diferentes (1 vs 0,5 no máximo binário). As duas valem 0 num grupo puro e são máximas no caso 50/50, só que em escalas diferentes.
- Gini é mais barato de calcular que entropia (não usa logaritmo), por isso é o padrão em implementações rápidas como o CART e o scikit-learn.

## Os três algoritmos clássicos

| Algoritmo | Ano | Critério | Atributos | Poda | Valores faltantes |
|---|---|---|---|---|---|
| ID3 | 1986 (Quinlan) | Ganho de informação | só categóricos | não | não trata |
| C4.5 | 1993 (Quinlan) | Gain Ratio | categóricos e contínuos | pós-poda | trata |
| CART | 1984 (Breiman et al.) | Gini (classif.) / erro quadrático (regr.) | categóricos e contínuos, sempre árvore binária | pós-poda (cost-complexity) | trata |

- CART é o que o scikit-learn implementa (`DecisionTreeClassifier`, `DecisionTreeRegressor`).
- Diferença que cai em prova: ID3 e C4.5 fazem divisão MULTIWAY (um ramo pra cada valor distinto do atributo categórico, ex.: "Tipo" vira 3 ramos: Fogo, Água, Grama). CART só faz perguntas de SIM/NÃO, sempre binário (mesmo um atributo categórico vira uma pergunta tipo "Tipo é Fogo? Sim/Não").
- Atributo contínuo (número real, tipo "idade"): ID3 puro não sabe lidar. C4.5 e CART tratam do mesmo jeito, testando limiares (ex.: "idade <= 30?") e escolhendo o corte que mais reduz a impureza, como se o número contínuo virasse uma pergunta binária na hora.

## Árvore de regressão

Mesma lógica de divisão recursiva, mas pra prever um NÚMERO em vez de uma classe (ex.: prever o preço de uma casa, não "é fraude ou não").

- Critério de divisão: em vez de entropia/Gini, usa erro quadrático médio (MSE) ou, equivalentemente, redução de variância. Pra cada divisão candidata, calcula o MSE dentro de cada subgrupo (a média de (y_i - ȳ)², onde ȳ é a média dos valores de y naquele subgrupo) e escolhe a divisão que mais reduz o MSE médio ponderado dos filhos, comparado ao MSE do pai.
- Previsão na folha: em vez de "classe mais comum", a folha prevê a MÉDIA dos valores de y dos exemplos de treino que caíram ali.
- Mesmos problemas de overfitting/instabilidade da árvore de classificação, e as mesmas soluções (pré-poda, pós-poda).

## Overfitting e poda

Se a árvore puder crescer sem limite, ela vai fazendo perguntas cada vez mais específicas até "decorar" cada exemplo de treino de cor, inclusive os erros e ruídos dos dados. Na hora de prever um caso novo, ela vai mal, porque aprendeu detalhe demais em vez de padrão de verdade. Isso é overfitting, alta variância.

- Pré-poda (early stopping): limita o crescimento durante a construção. Parâmetros no scikit-learn: `max_depth`, `min_samples_leaf`, `min_samples_split`.
- Pós-poda: deixa crescer tudo e depois corta os galhos que não compensam. No scikit-learn: `ccp_alpha` (cost-complexity pruning), calculável via `cost_complexity_pruning_path`.
- Fórmula do cost-complexity pruning (a "poda por custo-complexidade" do CART): `R_alpha(T) = R(T) + alpha * |T|`, onde `R(T)` é o erro da árvore T somado sobre as folhas (ex.: taxa de erro ou impureza total) e `|T|` é o número de folhas. O `alpha` é uma penalidade por folha: quanto maior o alpha, mais caro fica ter muitas folhas, então a árvore ótima fica menor. `alpha=0` não penaliza nada (árvore cresce inteira); o scikit-learn testa vários alphas ao longo do caminho de poda e escolhe o melhor por validação cruzada.
- Isso tudo é o clássico trade-off VIÉS x VARIÂNCIA: árvore profunda e sem poda tem viés baixo (encaixa bem nos dados de treino, inclusive no ruído) mas variância alta (muda muito se o treino mudar um pouco). Árvore rasa ou bem podada tem viés mais alto (simplifica demais) mas variância baixa (mais estável entre treinos diferentes). Podar é, de propósito, trocar um pouco de viés a mais por bem menos variância.

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

## Importância de atributos (feature importance)

- Pra cada atributo, soma-se, em TODOS os nós da árvore que usaram esse atributo pra dividir, o quanto a impureza caiu naquele nó (Ganho de informação ou redução de Gini), ponderado pela fração de exemplos que passaram por aquele nó. Depois normaliza-se pra tudo somar 1.
- É o que o scikit-learn devolve em `modelo.feature_importances_`.
- Serve pra duas coisas: explicar o modelo (quais atributos mais pesaram nas decisões) e seleção de atributos (descartar os que a árvore não usou ou usou muito pouco).
- Cuidado clássico de prova: atributo com muitos valores distintos tende a parecer mais "importante" do que realmente é, pelo mesmo viés do Ganho de Informação (ver Split Info/Gain Ratio acima).

## Ver também

- `decision_tree/decision_tree.py`: contas feitas na mão (entropia, Gini, ganho de informação) com o exemplo dos Pokémons, e o treino de verdade no dataset de fraude.
