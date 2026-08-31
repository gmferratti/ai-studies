# Random Forest

- Random Forest é bagging de árvores de decisão, com um ingrediente extra: além de sortear os EXEMPLOS de cada árvore (o bootstrap do bagging), também sorteia, em CADA divisão de CADA árvore, um subconjunto aleatório dos ATRIBUTOS disponíveis pra considerar naquela pergunta.
- O nome não é só estética: "floresta" porque é um bagging de várias árvores, "aleatória" porque tem essa segunda camada de sorteio (nos atributos) que o bagging puro não tem. Sem essa segunda camada, random forest é exatamente bagging de árvores.
- É de longe a variação de bagging mais usada na prática, porque essa segunda camada de aleatoriedade resolve um problema que o bagging sozinho não resolve bem (ver a próxima seção).

## Analogia central: batalha royale numa floresta de verdade

Imagine um jogo de batalha royale (tipo Fortnite): vários grupos de sobreviventes caem numa floresta cheia de baús de loot, cada grupo numa parte diferente do mapa (o bootstrap, cada grupo vendo uma amostra diferente da floresta). Só que existe uma arma claramente mais forte que qualquer outra no jogo, tipo um lança-foguetes: se todo mundo pudesse escolher livremente qual arma pegar em cada baú, praticamente todo grupo ia correr atrás do lança-foguetes primeiro, e a estratégia de quase todo grupo ia acabar sendo idêntica: "acha o lança-foguetes, vence o jogo". Isso é o bagging puro quando existe um atributo campeão disparado: quase toda árvore do comitê aprende a mesma primeira pergunta.

A correção do random forest: em cada baú, o jogo só oferece escolher entre um sorteio aleatório de ARMAS DISPONÍVEIS NAQUELE BAÚ (tipo só 2 das 3 armas do jogo), nunca o catálogo inteiro. Às vezes o lança-foguetes está entre as opções daquele baú, e o grupo pega ele mesmo assim. Mas às vezes ele NÃO está disponível ali, e o grupo é forçado a se virar com a segunda ou terceira melhor arma, desenvolvendo uma estratégia genuinamente diferente dos grupos que pegaram o lança-foguetes. No fim, os grupos da floresta inteira têm estratégias mais variadas entre si do que teriam se todo mundo pudesse sempre escolher livremente, e um comitê de estratégias variadas vota melhor que um comitê de estratégias todas iguais. Ver essa "escassez forçada de loot" acontecendo com números de verdade, num torneio de sobrevivência de brinquedo com um atributo campeão disparado, em `random_forest.py`.

## Vocabulário básico

- `max_features`: quantos atributos sortear como candidatos em CADA divisão de CADA árvore (não é sorteado uma vez só por árvore, é sorteado de novo em cada pergunta). Valor clássico pra classificação é `sqrt(número total de atributos)`; pra regressão costuma ser `número total de atributos / 3`.
- Correlação entre árvores: o quanto as previsões de duas árvores do comitê se parecem entre si. Duas árvores muito parecidas (alta correlação) basicamente votam a mesma coisa sempre, então juntar as duas no comitê rende pouco ganho a mais do que confiar numa só.
- Decorrelação: o efeito de tornar as árvores do comitê menos parecidas entre si. É o que `max_features` menor que o total produz, forçando árvores a usar atributos diferentes.
- Importância de atributo (feature importance): uma pontuação, calculada a partir do comitê inteiro, de quanto cada atributo contribuiu pra reduzir a impureza (entropia ou Gini) nas divisões onde ele foi usado, somado por todas as árvores do comitê. Serve pra responder "quais atributos o modelo mais usa pra decidir", mesmo sem conseguir ler uma árvore individual inteira.

## Por que bagging sozinho não decorrelaciona o suficiente

A redução de variância do bagging (ver `01-supervisionado/bagging/notes/anotacoes.md`) depende de quão CORRELACIONADAS as previsões das árvores do comitê são entre si. A variância da MÉDIA de `B` variáveis aleatórias, cada uma com variância `σ²` e correlação par a par `ρ`, tem essa cara:

```
Var(média de B árvores) = ρσ² + (1 - ρ)σ²/B
```

Repare no que acontece quando `B` cresce (mais árvores no comitê): o segundo termo, `(1-ρ)σ²/B`, encolhe até quase sumir, mas o primeiro termo, `ρσ²`, NÃO depende de `B` nenhum, ele fica parado lá independente de quantas árvores você somar. Ou seja, existe um piso de variância que nenhuma quantidade de árvores extras consegue quebrar, e a altura desse piso é ditada pela correlação `ρ` entre as árvores. Quando existe um atributo campeão disparado (tipo o "lança-foguetes" da analogia), quase toda árvore do bagging aprende a mesma primeira pergunta, e a correlação `ρ` entre as árvores fica alta, travando esse piso bem acima de zero, mesmo com centenas de árvores no comitê.

Reduzir `max_features` ataca exatamente esse `ρ`: forçando algumas árvores a nem sequer considerar o atributo campeão em certas divisões, as árvores passam a discordar mais entre si (correlação mais baixa), o que baixa o piso de variância e deixa o "mais árvores ajuda" continuar valendo por mais tempo. É por isso que random forest costuma superar bagging puro em datasets com atributos bem desbalanceados em poder preditivo, e empata (ou quase) com bagging quando os atributos já têm poder preditivo parecido entre si (nesse caso não existe "lança-foguetes" pra decorrelacionar, a correlação entre árvores já era baixa mesmo sem restringir `max_features`).

## O trade-off de `max_features`

Restringir demais os atributos disponíveis por divisão também tem custo: com `max_features` muito baixo, cada árvore individual fica pior (às vezes é forçada a usar um atributo bem fraco, porque o campeão simplesmente não estava disponível naquele sorteio), então o viés de cada árvore sobe. O ajuste certo de `max_features` é um equilíbrio: baixo o suficiente pra decorrelacionar de verdade as árvores, alto o suficiente pra cada árvore individual continuar sendo um estimador decente. Na prática, `sqrt(n_atributos)` (o padrão do scikit-learn pra classificação) costuma ser um bom ponto de partida, mas vale testar valores vizinhos. Ver essa comparação de verdade, com `max_features` variando, no dataset de fraude, em `random_forest.py`.

## Erro out-of-bag também vale aqui

Como random forest ainda é bagging por baixo (cada árvore treina numa amostra bootstrap), a técnica de erro out-of-bag (ver a nota do bagging) se aplica exatamente igual, sem nenhuma adaptação: cada árvore ainda deixa de fora, em média, uns 36,8% dos exemplos originais, e dá pra estimar o erro de generalização do comitê sem separar um conjunto de validação à parte. É um dos motivos práticos de random forest ser tão popular: ganha a estimativa de erro OOB de graça, junto com a redução de variância extra da decorrelação.

## Importância de atributos: o que sobra depois do comitê

Uma árvore de decisão sozinha já carrega uma medida natural de "quanto cada divisão ajudou": a redução de impureza (entropia ou Gini) que aquela pergunta específica trouxe. Random forest soma essa redução, ponderada pela fração de exemplos que passa por cada divisão, por TODAS as divisões que usam aquele atributo, em TODAS as árvores do comitê, e normaliza pra somar 1 no final. O resultado é um ranking de "quais atributos o modelo mais se apoiou pra decidir", uma forma de espiar dentro de um comitê de centenas de árvores sem conseguir ler nenhuma delas individualmente. É importante notar que essa importância mede o quanto o atributo foi ÚTIL PRA REDUZIR IMPUREZA dentro do próprio modelo treinado, não necessariamente uma relação causal com o problema real, e atributos correlacionados entre si podem "dividir o crédito", cada um aparecendo com importância menor do que teria sozinho. Ver o ranking de atributos de verdade, no dataset de fraude, em `random_forest.py`.

## Aspectos positivos e negativos

- (+) Herda todos os pontos fortes do bagging (redução de variância, erro OOB de graça, treino paralelizável) e melhora a redução de variância em cima disso, decorrelacionando as árvores.
- (+) Costuma exigir pouquíssimo ajuste de hiperparâmetro pra já ter um desempenho bom, o padrão `max_features='sqrt'` funciona bem numa faixa larga de problemas.
- (+) A importância de atributos dá uma forma prática (mesmo que aproximada) de interpretar o que o modelo está usando, mitigando parte da perda de interpretabilidade do bagging puro.
- (-) Ainda mais caro computacionalmente que uma árvore única ou até que o bagging puro, porque cada divisão de cada árvore também precisa sortear e avaliar um subconjunto de atributos.
- (-) `max_features` baixo demais aumenta o viés de cada árvore individual, existe um ponto de piora se restringir demais.
- (-) A importância de atributos pode enganar quando há atributos fortemente correlacionados entre si, dividindo o crédito de forma que nenhum deles apareça como claramente importante sozinho.

## Ver também

- `random_forest/random_forest.py`: comparação de bagging puro x random forest num torneio de sobrevivência de brinquedo com um atributo campeão disparado (mostrando a raiz da árvore variando ou não, e a concordância entre árvores caindo com a decorrelação), e o treino de verdade no dataset de fraude, incluindo a comparação de `max_features` e o ranking de importância dos atributos.
- `01-supervisionado/bagging/notes/anotacoes.md`: teoria do bagging, a base sobre a qual o random forest constrói a camada extra de aleatoriedade.
- `03-outros/comites_classificadores/notes/anotacoes.md`: teoria geral de comitês (requisito de diversidade, a mesma fórmula de correlação usada aqui), o guarda-chuva teórico por cima desta nota.
