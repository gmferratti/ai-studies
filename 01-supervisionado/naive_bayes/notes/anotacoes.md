# Naive Bayes

## Analogia central: o sistema anti-cheat de um jogo online

Ele não segue um fluxograma de perguntas tipo "jogou mais de 20 horas seguidas? então é bot" (isso seria mais parecido com uma árvore de decisão). Ele faz outra coisa: junta várias pistas de comportamento (tempo de reação, horário que joga, tipo de mensagem no chat) e calcula uma probabilidade de a conta ser bot, combinando cada pista como se ela não tivesse nenhuma relação com as outras. No fim, escolhe o rótulo (bot ou humano) que ficou com a maior probabilidade. É esse "combinar pistas como se fossem independentes" que dá nome ao algoritmo: ele é "ingênuo" (naive) de propósito.

## O que é

- Classificador probabilístico: em vez de desenhar uma fronteira geométrica entre as classes (como o k-NN faz com distância, ou a árvore de decisão com perguntas sim/não), o Naive Bayes calcula a probabilidade de cada classe dado os atributos observados, e escolhe a mais provável.

- Existem três famílias de classificadores, e cada uma decide o rótulo de um jeito diferente: geométricos ou baseados em distância (k-NN, que olha quem está mais perto no espaço de atributos), baseados em regras ou perguntas (árvore de decisão, que divide o espaço com cortes sucessivos) e probabilísticos (Naive Bayes). A diferença central é que um classificador probabilístico devolve uma probabilidade pra cada classe (por exemplo, 80% fraude, 20% não fraude) e escolhe a maior; os outros dois tipos chegam na decisão por geometria ou divisão de espaço, sem necessariamente calcular uma probabilidade de verdade no meio do caminho.

- O nome vem do Teorema de Bayes, que relaciona a probabilidade de A dado B com a probabilidade de B dado A. A fórmula e cada termo dela estão detalhados na seção seguinte.

## Por que "ingênuo" (naive)

É "ingênuo" porque assume que os atributos são condicionalmente independentes entre si, dado a classe. Ou seja, sabendo a classe, um atributo não diz nada sobre o valor de outro atributo. Essa suposição quase nunca é verdadeira na prática (altura e peso, por exemplo, claramente se relacionam), mas o algoritmo funciona bem mesmo assim, porque pra classificar só importa qual classe fica com a maior probabilidade, não o valor exato dela.

Essa suposição lembra a ideia de colinearidade usada em regressão, mas não é a mesma coisa, e vale separar bem os dois conceitos:

- Colinearidade linear é medida olhando o dataset inteiro, todas as classes misturadas, sem separar nada. Ela responde: será que dá pra calcular o valor de uma variável a partir da outra usando só uma conta de multiplicar e somar (é isso que "combinação linear" quer dizer, uma conta simples desse tipo)? O caso mais extremo é altura em centímetros e altura em polegadas: multiplica um valor por uma constante fixa e chega exatamente no outro, sempre, sem erro nenhum. Por isso elas são perfeitamente colineares.

- Independência condicional, a suposição que o Naive Bayes faz, é medida de um jeito diferente: dentro de cada classe, separadamente, depois de já saber o rótulo daquele exemplo. A pergunta muda: sabendo de antemão que o exemplo pertence à classe X, o valor de um atributo ainda dá alguma pista sobre o valor de outro atributo, ou saber um não ajuda em nada a adivinhar o outro?

Um exemplo concreto separa bem as duas ideias. Imagine as classes "cachorro" e "gato", com os atributos peso e altura.

Primeiro, olhando todos os animais juntos, cachorros e gatos misturados: animal maior tende a pesar mais, então peso e altura sobem e descem juntos com bastante regularidade. Existe colinearidade linear forte no dataset inteiro.

Agora separa só os cachorros e pergunta de novo: dentro desse grupo, sabendo que o bicho já é cachorro, o peso dele ainda ajuda a adivinhar a altura, além do que a média da raça já entregaria sozinha? Se essa relação praticamente desaparecer quando as classes são olhadas uma de cada vez, a suposição de independência condicional se sustenta bem, mesmo tendo aparecido colinearidade forte lá no dataset misturado.

Ou seja, são duas medidas em "cortes" diferentes dos mesmos dados: colinearidade linear olha o bolo inteiro, independência condicional olha fatia por fatia, uma classe de cada vez. As duas conclusões podem apontar em direções opostas ao mesmo tempo, e não tem contradição nisso.

Por que só importa a classe com maior probabilidade: o objetivo do classificador é decidir um rótulo, não estimar a probabilidade exata e calibrada de cada classe. Mesmo que o valor numérico saia completamente distorcido pela suposição de independência (o modelo calcular 95% quando o valor real seria 70%, por exemplo), a decisão final continua certa desde que a classe correta continue sendo a de maior valor entre as opções. É essa tolerância que faz o algoritmo funcionar bem na prática, apesar da suposição irreal.

## Teorema de Bayes

O teorema pressupõe sempre uma relação classe-atributo, e vale fixar o que é cada coisa antes da fórmula: classe é o que se quer prever, o rótulo, a resposta que só se saberia de verdade depois (como "essa transação era fraude?"). Atributo é a informação disponível de antemão, usada pra tentar prever a classe (valor da compra, horário, localização). 

Essa divisão não é uma propriedade fixa da variável, é uma escolha de qual pergunta está sendo feita: no dataset de fraude, "é fraude" é a classe porque é a resposta que se quer descobrir, e "valor da transação" é atributo porque já está disponível no momento da decisão. 

Em outro problema os papéis poderiam se inverter (por exemplo, prever a faixa de valor mais provável de uma transação já sabendo que ela foi confirmada como fraude, aí "faixa de valor" vira a classe e "é fraude" vira informação de entrada), mas o Teorema de Bayes em si não muda: ele sempre relaciona "probabilidade do que quero prever, dado o que já observei".

`P(A|B) = P(B|A) * P(A) / P(B)`

- `P(A|B)` (posterior): probabilidade da classe A ser a certa, depois de observar o atributo B. É o que se quer calcular.
- `P(B|A)` (verossimilhança, likelihood): probabilidade de observar o atributo B, se a classe já fosse A. Vem das frequências contadas no treino.
- `P(A)` (prior, probabilidade a priori): probabilidade da classe A sem olhar pra nenhum atributo ainda. Ex.: se 2% das transações do dataset são fraude, `P(fraude) = 0,02`, independente de qualquer outra informação.
- `P(B)` (evidência): probabilidade de observar o atributo B, somando todas as classes possíveis. Funciona como constante de normalização, pra a soma das probabilidades posteriores de todas as classes dar 1.

A ideia central é que a probabilidade de A dado B não depende só da relação entre A e B (`P(B|A)`), depende também de quão provável A já era antes de qualquer evidência (`P(A)`, o prior). Um evento raro continua raro mesmo com uma evidência que aponta pra ele, se o prior for baixo o bastante: é a mesma lógica por trás de "exame raro dá falso positivo raro, mesmo sendo um bom exame".

Um exemplo concreto, usando o dataset de fraude: imagine que 2% das transações são fraude (`P(fraude) = 0,02`, o prior) e que, entre as fraudes, 60% têm valor de compra acima de mil reais (`P(valor alto | fraude) = 0,6`, a verossimilhança). Se, olhando todas as transações juntas, fraude e não fraude, 5% têm valor acima de mil reais (`P(valor alto) = 0,05`, a evidência), então a probabilidade de ser fraude sabendo que o valor foi alto é:

`P(fraude | valor alto) = 0,6 * 0,02 / 0,05 = 0,24`

Ou seja, 24%. Repare que mesmo com uma evidência que aponta pra fraude, o resultado final não passa de 24%, porque o prior de fraude já era baixo pra começar: é o mesmo raciocínio do exame raro citado acima.

## Extensão pra vários atributos e o critério MAP

Na prática um exemplo tem vários atributos (`X1, X2, ..., Xn`), não só um. Com a suposição de independência condicional, a verossimilhança conjunta vira o produtório das verossimilhanças individuais:

`P(Classe | X1, ..., Xn) ∝ P(Classe) * P(X1|Classe) * P(X2|Classe) * ... * P(Xn|Classe)`

O símbolo `∝` quer dizer "proporcional a": dá pra ignorar o denominador `P(B)` do Teorema de Bayes, porque ele é igual pra todas as classes (não muda qual classe vence a comparação), então só interessa comparar os numeradores.

A classificação final calcula esse produto pra cada classe possível e escolhe a classe com maior valor. Esse critério tem nome, MAP (Maximum A Posteriori), sem relação nenhuma com o MapReduce, o modelo de processamento distribuído do Hadoop/Spark: é coincidência de sigla, cada um usa "MAP" pra uma coisa completamente diferente. É "máximo a posteriori" porque maximiza a probabilidade calculada depois (posteriori) de observar a evidência, em contraste com outro critério comum em estatística, o MLE (Maximum Likelihood Estimation), que maximiza só a verossimilhança e ignora o prior. No Naive Bayes, usar MAP é o que faz o prior (`P(Classe)`) entrar na conta: se o prior de fraude é baixíssimo, precisa de uma verossimilhança bem forte pra compensar e fazer a classe fraude vencer a comparação.

## Problema do produto de zero

Se algum atributo nunca apareceu junto com uma classe no treino, `P(Xi|Classe) = 0`, e como é um produtório, isso zera a probabilidade inteira daquela classe, não importa quão fortes sejam as outras evidências. É um dos pontos fracos clássicos do método: uma única combinação atributo-classe nunca vista derruba a previsão inteira daquela classe pra zero.

Correção: suavização de Laplace (Laplace smoothing, também chamada add-one smoothing). Em vez de contar direto, soma 1 em cada contagem antes de dividir: `P(Xi|Classe) = (contagem(Xi, Classe) + 1) / (contagem(Classe) + k)`, onde `k` é o número de valores possíveis daquele atributo. Isso garante que nenhuma probabilidade fica exatamente zero, e é a técnica mais usada na prática, principalmente em classificação de texto.

Existem variações: a suavização de Lidstone generaliza a ideia trocando o "+1" por um parâmetro `α` ajustável (entre 0 e 1, sendo Laplace o caso particular `α = 1`), controlando o quanto se "empresta" de probabilidade pras combinações não vistas. Fora da família Naive Bayes, problemas parecidos de contagem zero aparecem em modelos de linguagem, resolvidos com técnicas mais sofisticadas, como o Kneser-Ney smoothing, mas pro escopo do Naive Bayes o Laplace já resolve bem.

## Variantes

O que muda entre elas é só como `P(Xi|Classe)` é calculado:

| Variante | Tipo de atributo | Como calcula a verossimilhança |
|---|---|---|
| Gaussian NB | contínuo (número real) | assume que os valores seguem uma distribuição normal dentro de cada classe, usa média e desvio padrão da classe pra calcular a densidade |
| Multinomial NB | contagem (ex.: quantas vezes uma palavra aparece) | frequência relativa do valor dentro da classe, clássico em classificação de texto (bag of words) |
| Bernoulli NB | binário (presença/ausência) | trata cada atributo como "aconteceu ou não", penaliza inclusive a ausência de um atributo esperado, diferente do Multinomial que só olha o que apareceu |

`GaussianNB` é a variante mais usada quando os atributos são numéricos contínuos, caso do dataset de fraude do módulo 01. Pra cada atributo contínuo, calcula a média e o desvio padrão dos valores observados dentro de cada classe, no treino. Na hora de prever um exemplo novo, substitui o valor do atributo na fórmula da densidade normal (a mesma curva de sino da distribuição gaussiana) usando a média e o desvio padrão daquela classe, e isso devolve um número que funciona como `P(Xi|Classe)` na equação do produtório. Como a curva de sino é mais alta perto da média, um valor de atributo próximo da média da classe puxa a probabilidade pra cima, e um valor muito distante puxa pra baixo.

Um detalhe de implementação relevante: como o produtório de muitos números pequenos, entre 0 e 1, pode ficar tão perto de zero que o computador arredonda pra zero (erro de underflow), na prática os algoritmos trabalham com o logaritmo das probabilidades, transformando o produtório numa soma, que é numericamente mais estável e dá o mesmo resultado na comparação entre classes.

## Aspectos positivos

- Rápido de treinar e de prever: só calcula frequências e médias, não precisa de otimização iterativa.
- Funciona bem com poucos dados de treino, comparado a modelos mais complexos.
- Lida naturalmente com muitos atributos, como milhares de palavras num filtro de spam, justamente porque a suposição de independência evita calcular probabilidades conjuntas complexas. Sem essa suposição, calcular a probabilidade de milhares de palavras aparecerem juntas exigiria uma tabela de frequências pra cada combinação possível delas, um número que explode exponencialmente e que nenhum dataset teria exemplos suficientes pra cobrir. Com a suposição de independência, o problema vira contar cada palavra separadamente dentro de cada classe, o que precisa de muito menos dados. Texto também gera atributos esparsos, a maioria das palavras do vocabulário não aparece na maioria dos textos, e o Multinomial NB lida bem com isso porque só olha as palavras que de fato apareceram.
- Boa baseline: mesmo com a suposição "ingênua" claramente errada em muitos casos reais, costuma ter desempenho competitivo.

## Aspectos negativos

- Suposição de independência condicional raramente é verdadeira, prejudica a qualidade das probabilidades estimadas (mesmo que a classificação final ainda saia certa na maioria das vezes).
- Sensível a atributos redundantes: dois atributos correlacionados são contados como se fossem duas evidências independentes, o que pode enviesar a decisão numa direção sem motivo real.
- GaussianNB assume distribuição normal dos atributos contínuos, que pode não bater com a distribuição real dos dados.

## Aplicações clássicas

- Filtro de spam, o exemplo mais citado, com Multinomial NB sobre contagem de palavras.
- Classificação de sentimento e outras tarefas de texto.
- Diagnóstico médico simples, sintomas como atributos, doença como classe.

## Ver também

- `naive_bayes/naive_bayes.py`: contas feitas na mão (prior, verossimilhança, produtório, critério MAP, problema do produto de zero) com o exemplo do detector de bot, e o treino de verdade no dataset de fraude.
