# Bagging (Bootstrap Aggregating)

- Bagging não é um algoritmo de classificação novo, é uma receita pra combinar vários modelos (em geral do mesmo tipo) treinados em versões levemente diferentes do mesmo dataset, e depois juntar as opiniões deles numa única previsão.
- O nome já entrega a receita: **B**ootstrap (a forma de gerar essas versões diferentes do dataset) + **Agg**regat**ing** (a forma de juntar as opiniões no final).
- É um tipo de aprendizado de comitê (ensemble): a aposta é que um grupo de modelos "medianos", cada um enxergando uma fatia levemente diferente dos dados, erra menos no conjunto do que um único modelo enxergando tudo de uma vez.

## Analogia central: o julgamento de classe

Imagine um anime de mistério escolar, tipo Danganronpa, onde a turma inteira precisa votar quem é o culpado num julgamento. Só que ninguém na sala viu todas as evidências: cada investigador recebeu, por sorteio, um monte de pistas tiradas ao acaso do baralho de evidências, podendo até pegar a mesma pista repetida (tipo tirar uma carta, anotar, devolver ao baralho e sortear de novo). Alguns investigadores, por puro azar do sorteio, receberam pistas enganosas ou incompletas, e vão chegar numa conclusão errada. Mas a turma inteira não decide por um investigador só: todo mundo vota, e vence a opinião da maioria.

É exatamente essa a ideia do bagging. Cada "investigador" é um modelo (quase sempre uma árvore de decisão) treinado numa amostra sorteada com reposição do dataset original (o bootstrap). Um investigador isolado pode ter visto uma amostra tendenciosa e errar feio, mas o erro de cada investigador tende a ir num sentido diferente do erro dos outros, porque cada um viu um sorteio diferente. Quando a turma inteira vota (a agregação), os erros individuais tendem a se cancelar, e sobra uma decisão mais estável e confiável do que qualquer investigador sozinho conseguiria dar. Ver esse "julgamento" acontecendo com números de verdade, incluindo o voto de cada investigador divergindo, em `bagging.py`.

## Vocabulário básico

- Bootstrap: técnica estatística de gerar uma nova amostra sorteando, **com reposição**, o mesmo número de exemplos que o dataset original tem. "Com reposição" quer dizer que, depois de sortear um exemplo, ele volta pro baralho e pode ser sorteado de novo: então a amostra final costuma repetir alguns exemplos originais e deixar outros de fora completamente.
- Estimador base (base learner): o tipo de modelo que cada "investigador" usa. No bagging costuma ser uma árvore de decisão sem poda (ou pouco podada), mas em tese pode ser qualquer classificador.
- Agregação (aggregating): a forma de juntar as previsões dos vários estimadores numa única resposta. Em classificação, é voto majoritário (a moda): cada estimador vota numa classe, ganha a mais votada. Em regressão, é a média das previsões.
- Out-of-bag (OOB): os exemplos do dataset original que **não** entraram numa determinada amostra bootstrap (ficaram de fora do sorteio daquele investigador específico). Servem de "conjunto de validação de graça" pra aquele estimador, sem precisar separar dados extras (ver seção própria abaixo).
- Variância (de um modelo): o quanto as previsões do modelo mudariam se ele fosse retreinado com um dataset de treino levemente diferente. Um modelo de alta variância é "instável": pequenas mudanças nos dados de treino mudam bastante a fronteira aprendida.

## Por que o bootstrap deixa de fora cerca de 37% dos exemplos

Cada amostra bootstrap tem o mesmo tamanho `n` do dataset original, mas sorteada com reposição. A chance de um exemplo específico **não** ser sorteado numa única tentativa é `(1 - 1/n)`. Como são `n` sorteios independentes (um pra cada posição da amostra nova), a chance desse mesmo exemplo ficar de fora da amostra inteira é `(1 - 1/n)^n`.

Pra `n` grande, esse número converge pra uma constante conhecida:

```
lim (n -> infinito) (1 - 1/n)^n = 1/e ≈ 0,368
```

Ou seja, cerca de 36,8% dos exemplos originais ficam de fora de cada amostra bootstrap (esses são os exemplos out-of-bag daquele estimador), e os outros 63,2% entram, alguns deles repetidos mais de uma vez pra completar o tamanho `n`. Esse número (63,2% dentro, 36,8% fora) é uma conta clássica de prova: não depende do dataset, só do fato de sortear `n` vezes com reposição de um total de `n` itens. Ver a contagem batendo com esse valor teórico, numa simulação com o dataset de brinquedo, em `bagging.py`.

## Aggregating: por que juntar opiniões reduz o erro

A justificativa formal vem da decomposição do erro esperado de um modelo em três partes: viés (bias), variância e ruído irredutível.

```
erro esperado = viés² + variância + ruído irredutível
```

- Viés é o erro sistemático de um modelo simples demais pra capturar o padrão real (ex.: tentar separar duas classes entrelaçadas com uma única reta).
- Variância é o erro que vem da instabilidade: o modelo se ajusta demais às particularidades da amostra de treino específica que ele viu, e prevê diferente se treinado numa amostra levemente diferente.
- Ruído irredutível é o erro que nenhum modelo consegue eliminar, porque vem de aleatoriedade genuína ou de informação que falta nos próprios atributos.

Treinar vários estimadores em amostras bootstrap diferentes e depois tirar a média (ou o voto majoritário) das previsões deles reduz a parcela de **variância** do erro, sem piorar o viés: se cada estimador individual é aproximadamente sem viés (ou tem o mesmo viés), a média de vários estimadores continua com esse mesmo viés, mas com bem menos variância, porque os erros "aleatórios" de cada estimador (motivados pela amostra bootstrap específica que ele viu) tendem a apontar em direções diferentes e se cancelar parcialmente na agregação. É a mesma lógica estatística de por que a média de várias medições barulhentas é mais confiável que uma medição só.

Essa é a razão pela qual bagging combina tão bem com árvores de decisão sem poda: uma árvore bem profunda tem viés baixo (decora até padrões bem específicos do treino) mas variância altíssima (muda bastante de formato com pequenas mudanças no dataset), justamente o perfil de modelo em que reduzir variância rende mais ganho. Um classificador que já é estável por natureza, como k-NN com K grande ou uma regressão logística simples, tem pouca variância pra reduzir, então o bagging ajuda bem menos (às vezes quase nada) nesses casos.

Vale reparar de onde exatamente vem essa variância: é da amostra de TREINO mudar, não de qualquer fonte de aleatoriedade. Treinar duas árvores de decisão com o mesmo dataset de treino inteiro, só trocando o parâmetro de semente aleatória do algoritmo, quase não muda nada quando os atributos são contínuos: essa semente só serve pra desempatar entre cortes igualmente bons, e com atributos contínuos raramente existe empate exato pra desempatar. A instabilidade de verdade só aparece quando o CONJUNTO de treino em si muda, tipo entre uma amostra bootstrap e outra. Ver essa pegadinha acontecendo na prática (e a correção do experimento) em `bagging.py`.

## Out-of-bag (OOB): validação de graça

Como cada estimador do comitê só viu cerca de 63,2% dos exemplos originais no seu treino, sobra pra cada exemplo, em média, uns 36,8% dos estimadores que nunca o viram durante o treino. Isso permite estimar o erro de generalização do comitê inteiro sem separar um conjunto de validação à parte:

Importante não confundir: o "out-of-bag" não é um conjunto fixo de exemplos que sobra pra todo mundo, tipo um holdout único separado antes do treino. Cada estimador sorteia a própria amostra bootstrap de forma independente, então cada um deixa de fora um subconjunto diferente de exemplos. Olhando pela ótica de um exemplo específico, ele acaba sendo out-of-bag pra alguns estimadores do comitê (os que não o sortearam) e dentro da amostra de outros (os que sortearam), variando exemplo a exemplo. É por isso que o cálculo do erro OOB é feito individualmente: para cada exemplo, primeiro descobre-se quais estimadores especificamente não o viram, e só o voto desse subconjunto entra na conta daquele exemplo.

1. Para cada exemplo `x_i` do dataset original, identifica-se quais estimadores **não** usaram `x_i` no treino (os que o tinham out-of-bag).
2. Agrega-se só o voto desses estimadores pra prever a classe de `x_i`.
3. Compara-se essa previsão com o rótulo verdadeiro de `x_i`, repetindo pra todos os exemplos, e a taxa de erro final é o erro OOB.

O erro OOB funciona como um substituto razoável de validação cruzada, praticamente de graça (é só reaproveitar os mesmos treinos que já aconteceriam de qualquer jeito), o que é útil quando os dados são escassos e separar um conjunto de validação à parte custa caro em quantidade de exemplos disponíveis pro treino. Ver o cálculo do erro OOB acontecendo no exemplo de brinquedo, comparando com o erro medido de fato num conjunto de teste separado, em `bagging.py`.

## Paralelização: a diferença estrutural pro boosting

Cada estimador do bagging é treinado de forma **independente** dos outros: a amostra bootstrap do investigador 3 não depende em nada do resultado do investigador 2. Isso quer dizer que todos os estimadores do comitê podem ser treinados ao mesmo tempo, em paralelo, sem nenhuma dependência de ordem.

Essa independência é a principal diferença estrutural para o boosting (tema da próxima nota): no boosting, cada novo estimador é treinado **em sequência**, prestando atenção específica nos erros que os estimadores anteriores cometeram, então não dá pra paralelizar o treino da mesma forma. Random forest (também na próxima nota) é uma variação do bagging que soma mais uma fonte de aleatoriedade (sorteio de atributos a cada divisão da árvore, além do sorteio de exemplos), mantendo a mesma estrutura paralela e a mesma lógica de redução de variância.

## Aspectos positivos e negativos

- (+) Reduz variância de forma consistente, especialmente eficaz em cima de estimadores de alta variância (árvores profundas sem poda).
- (+) Treino inteiramente paralelizável: cada estimador não depende dos outros.
- (+) Erro out-of-bag dá uma estimativa de generalização sem gastar dados numa validação separada.
- (+) Reduz a chance de overfitting num único conjunto de treino específico, sem exigir poda manual cuidadosa do estimador base.
- (-) Não reduz viés: se o estimador base é sistematicamente ruim pro problema (viés alto, ex.: um estimador linear demais pra um padrão bem curvo), o comitê inteiro herda esse mesmo viés.
- (-) Ganho pequeno (às vezes nenhum) em cima de estimadores que já são estáveis por natureza, porque não sobra muita variância pra reduzir.
- (-) Perde a interpretabilidade fácil de um único modelo: uma árvore de decisão sozinha dá pra ler como uma sequência de perguntas, um comitê de centenas de árvores não.
- (-) Custo de treino e de memória multiplicado pelo número de estimadores do comitê, mesmo treinando em paralelo.

## Ver também

- `bagging/bagging.py`: bootstrap manual num júri de brinquedo com 16 suspeitos, votos individuais divergindo, cálculo do erro out-of-bag comparado ao erro de teste, e o treino de verdade no dataset de fraude (`BaggingClassifier` comparado a uma árvore única, em várias sementes aleatórias, pra ver a variância caindo na prática).
- `01-supervisionado/decision_tree/notes/anotacoes.md`: teoria da árvore de decisão, o estimador base mais comum do bagging.
