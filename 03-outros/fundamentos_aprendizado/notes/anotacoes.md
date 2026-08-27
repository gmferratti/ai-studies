# Formalização do Problema de Aprendizado e Viés Indutivo

- Antes de entrar em algoritmo específico, vale formalizar o que "aprender com exemplos" quer dizer de verdade, com nome pra cada peça do quebra-cabeça.
- E entender por que nenhum algoritmo aprende só "olhando os dados": todo mundo precisa de uma suposição extra, chamada viés indutivo, pra conseguir generalizar.
- Esses dois conceitos são pré-requisito pra entender o PAC-learning (`03-outros/pac_learning/notes/anotacoes.md`), que formaliza com números exatos quantos exemplos são necessários pra generalizar com uma certa confiança.

## Analogia central: o detetive que só viu parte das pistas

Imagine um detetive chamado pra resolver um caso. Ele não viu o crime acontecer, só tem acesso a um punhado de pistas: uma pegada, um horário, um álibi furado. A partir desse punhado de pistas, ele precisa construir uma teoria de quem é o culpado, uma teoria que não só explique as pistas já vistas, mas que também sirva pra prever o comportamento do suspeito daqui pra frente (se ele vai fugir, se vai confessar, se existe cúmplice).

Só que aqui está o problema: praticamente qualquer punhado pequeno de pistas pode ser explicado por várias teorias diferentes, algumas óbvias, outras mirabolantes. "O mordomo fez isso" e "foi uma conspiração de três pessoas encobrindo o mordomo" podem ambas explicar as mesmas três pistas. Um bom detetive não escolhe entre essas teorias só olhando as pistas de novo, ele aplica um princípio extra, como a navalha de Occam ("a explicação mais simples que encaixa nos fatos costuma ser a certa"), pra decidir em qual teoria apostar. Esse princípio extra, que não vem dos dados e sim de uma suposição sobre como o mundo costuma funcionar, é exatamente o que chamamos de viés indutivo.

## Formalização do problema de aprendizado

Trocando a metáfora do detetive pelos nomes "oficiais" que aparecem em qualquer livro de aprendizado de máquina:

- Espaço de instâncias (`X`): o conjunto de todos os exemplos possíveis, descritos pelos seus atributos. No dataset de fraude deste repositório, cada transação (com seu valor, horário, componentes de PCA) é um ponto de `X`.
- Espaço de saída (`Y`): o conjunto de respostas possíveis. Em classificação, um conjunto de classes (`Y = {fraude, normal}`); em regressão, os números reais; no aprendizado não supervisionado, geralmente nem existe um `Y` de verdade, porque não há rótulo pra aprender.
- Conceito-alvo ou função-alvo (`c`, às vezes chamada de `f`): a regra verdadeira, desconhecida, que de fato decide a resposta certa pra qualquer exemplo (`c: X -> Y`). Ninguém tem acesso direto a `c`; ela é a "verdade lá fora" que o aprendizado tenta se aproximar. No exemplo do detetive, `c` é "o que realmente aconteceu"; no dataset de fraude, é a regra real (bem mais complicada do que qualquer modelo consegue capturar por completo) que separa fraude de transação legítima.
- Conjunto de treinamento (`D`): uma amostra finita de exemplos já rotulados, `D = {(x_1,y_1), ..., (x_n,y_n)}`, onde `y_i = c(x_i)`. É tudo que o algoritmo tem pra trabalhar, o equivalente às pistas que o detetive já levantou.
- Espaço de hipóteses (`H`): o conjunto de regras que o algoritmo escolhido é CAPAZ de considerar. Cada família de algoritmo enxerga um `H` diferente: pra uma árvore de decisão, `H` é o conjunto de todas as árvores possíveis com aquela profundidade máxima; pra um SVM linear, `H` é o conjunto de todos os hiperplanos; pra Naive Bayes, `H` é o conjunto de classificadores que respeitam a suposição de independência condicional. Ver o vocabulário completo em `03-outros/pac_learning/notes/anotacoes.md`, que usa exatamente esses mesmos termos (`c`, `H`, `h`) pra derivar quantos exemplos são necessários.
- Hipótese aprendida (`h`): a regra específica, escolhida de dentro de `H`, depois de olhar pra `D`. É a teoria final do detetive, o modelo treinado (a árvore podada, o hiperplano encontrado, as probabilidades estimadas).

Aprender, nessa formalização, é o processo de escolher uma hipótese `h` dentro de `H` a partir do conjunto de treino `D`, na esperança de que `h` se pareça o bastante com `c` pra acertar também exemplos que não estavam em `D`. Esse processo de generalizar do particular (os exemplos vistos) pro geral (uma regra que vale pra qualquer exemplo futuro) chama-se indução, o oposto de dedução, que parte de regras já dadas e aplica pra casos específicos. Um algoritmo de aprendizado, no fim das contas, é um procedimento de indução.

Duas medidas de erro que não podem ser confundidas:

- Erro empírico (ou erro de treino): a taxa de erro de `h` só nos exemplos de `D`, os que já foram vistos. É fácil de calcular, mas engana: uma hipótese pode ter erro empírico zero e ainda assim ser péssima em exemplos novos (é a raiz do overfitting).
- Erro de generalização (ou erro verdadeiro): a taxa de erro de `h` sobre TODOS os exemplos possíveis de `X`, incluindo os que nunca apareceram no treino. É o que realmente importa, e é o que a teoria PAC-learning tenta colocar um número em cima (`03-outros/pac_learning/notes/anotacoes.md`), já que na prática nunca dá pra medir esse erro diretamente (equivaleria a testar contra `c`, que é desconhecida).

Uma hipótese `h` é chamada de consistente com `D` quando acerta 100% dos exemplos de treino (`h(x_i) = y_i` pra todo `i`). Achar uma hipótese consistente é sempre possível se `H` for flexível o bastante (por exemplo, uma árvore de decisão sem limite de profundidade quase sempre consegue decorar o treino inteiro), mas consistência com o treino não é garantia nenhuma de bom erro de generalização: é só decorar as pistas já vistas, sem necessariamente ter entendido o crime.

## Viés indutivo: por que só olhar os dados nunca é suficiente

Aqui mora o ponto mais sutil da formalização acima: pra qualquer conjunto de treino `D` finito, existe mais de uma hipótese `h` consistente com ele, geralmente infinitas. Volte pro detetive: com três pistas, sempre dá pra inventar uma segunda teoria (mais complicada, mais forçada) que também explica as mesmas três pistas, mas que discorda completamente da primeira teoria sobre o que vai acontecer depois. Sem nenhum critério além das pistas, não tem como escolher racionalmente entre elas.

Viés indutivo é justamente esse critério extra: o conjunto de suposições que um algoritmo de aprendizado carrega, além dos dados observados, que o permite escolher UMA hipótese entre as várias consistentes com o treino, e assim generalizar pra exemplos novos. Sem viés indutivo, nenhum algoritmo consegue prever nada sobre um exemplo que não esteja literalmente já no treino: "aprender sem nenhuma suposição" e "decorar" são a mesma coisa.

Existem dois tipos, que vale diferenciar porque costumam confundir em prova:

- Viés de restrição (ou viés de linguagem): limita de antemão QUAIS hipóteses o algoritmo sequer consegue representar, reduzindo `H` pra um subconjunto menor que "todas as funções imagináveis". Um SVM com kernel linear só considera hiperplanos: mesmo que a fronteira ideal fosse uma curva maluca, o SVM linear nunca vai propor isso, porque essa hipótese nem existe dentro do `H` dele. O Naive Bayes só considera classificadores que respeitam a independência condicional entre atributos: essa suposição está embutida antes mesmo de olhar qualquer dado.
- Viés de preferência (ou viés de busca): `H` pode até ser grande, às vezes até conter a hipótese "certa", mas o jeito como o algoritmo busca dentro de `H` prefere sistematicamente certas hipóteses sobre outras. Uma árvore de decisão (ID3/CART) não testa toda árvore possível, ela constrói gulosamente escolhendo a melhor pergunta a cada passo, o que na prática empurra a busca em direção a árvores mais compactas quando isso resolve o problema, uma versão prática da navalha de Occam ("entre hipóteses que explicam os dados igualmente bem, prefira a mais simples").

Os algoritmos já revisados neste repositório servem de exemplo concreto de viés indutivo em ação, cada um apostando numa suposição diferente sobre como o mundo real costuma se comportar:

| Algoritmo | Viés indutivo principal | Tipo |
|---|---|---|
| Árvore de decisão | Prefere árvores mais simples e rasas (indução gulosa, escolhe a melhor pergunta a cada nó sem reconsiderar depois) | Preferência |
| k-NN | Exemplos parecidos (próximos no espaço de atributos) tendem a ter o mesmo rótulo | Nenhuma restrição forte de forma, mas assume "suavidade" local |
| SVM | Entre as fronteiras que separam as classes, a de margem máxima generaliza melhor; com kernel linear, só considera hiperplanos | Restrição (com kernel linear) + preferência (margem máxima) |
| Naive Bayes | Os atributos são condicionalmente independentes dado a classe | Restrição |

Nenhum desses vieses é "o certo" de forma absoluta. Cada um funciona bem quando a suposição embutida realmente combina com a estrutura do problema (o Naive Bayes vai mal quando os atributos são fortemente dependentes entre si; o SVM linear vai mal quando a fronteira de verdade é bem torta) e vai mal quando não combina. Essa é a intuição por trás do teorema do "não existe almoço grátis" (No Free Lunch Theorem, Wolpert e Macready, 1997): em média, sobre TODOS os problemas possíveis, nenhum algoritmo de aprendizado é melhor que outro. Um algoritmo só se sai melhor que outro nos problemas onde o viés indutivo dele calha de combinar com a estrutura real daqueles dados. É exatamente por isso que existem vários algoritmos diferentes, cada um carregando um viés diferente, em vez de um único "algoritmo definitivo".

Viés indutivo forte demais (H muito restrito) corre o risco de nem conseguir representar o conceito-alvo de jeito nenhum, ficando com erro alto mesmo no treino: viés alto no sentido estatístico do trade-off viés-variância. Viés indutivo fraco demais (H muito flexível, poucas restrições) consegue decorar qualquer treino, inclusive ruído, mas generaliza mal: variância alta. É o mesmo trade-off que aparece na poda de árvore de decisão e na escolha de K no k-NN, só que aqui descrito na origem: a causa raiz da variância alta é justamente um viés indutivo fraco demais pra aquele tamanho de treino.

## Ver também

- `03-outros/pac_learning/notes/anotacoes.md`: usa esse mesmo vocabulário (`c`, `H`, `h`, erro verdadeiro) pra derivar quantos exemplos de treino são necessários pra generalizar com uma confiança escolhida.
- `01-supervisionado/decision_tree/notes/anotacoes.md`, `01-supervisionado/knn/notes/anotacoes.md`, `01-supervisionado/svm/notes/anotacoes.md`, `01-supervisionado/naive_bayes/notes/anotacoes.md`: cada um detalha o viés indutivo específico do seu algoritmo em ação.
