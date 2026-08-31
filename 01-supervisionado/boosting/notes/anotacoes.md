# Boosting (AdaBoost)

- Assim como o bagging, boosting não é um classificador novo, é uma receita pra combinar vários modelos fracos numa previsão só. A diferença é a receita inteira: onde o bagging treina o comitê em paralelo, todo mundo vendo uma amostra sorteada do mesmo jeito, o boosting treina em **sequência**, e cada novo integrante do comitê é escolhido de propósito pra corrigir o que os anteriores erraram.
- O algoritmo clássico (e o mais cobrado em prova) é o AdaBoost (Adaptive Boosting). A ideia se generaliza depois pro Gradient Boosting (XGBoost, LightGBM), mas o raciocínio central de "focar peso no que ainda está errado" é o mesmo.

## Analogia central: o torneio de lutadores contra o campeão

Imagine uma escola de artes marciais tentando montar a estratégia perfeita pra vencer o campeão invicto do torneio. Ninguém acerta sozinho: o primeiro lutador que a escola manda pra treinar contra o campeão é só um pouco melhor que chute aleatório, um "especialista fraco", que já acerta a estratégia certa contra alguns golpes do campeão, mas apanha feio dos outros. Depois desse primeiro treino, a escola não manda o próximo lutador aleatoriamente: ela olha exatamente ONDE o primeiro apanhou mais, e treina o segundo lutador focado bem naqueles golpes específicos, dando menos atenção ao que o primeiro já resolveu. O terceiro lutador entra focado no que sobrou de difícil depois dos dois primeiros, e assim por diante.

No fim do torneio, a escola não decide a estratégia final perguntando pra um lutador só: ela faz uma votação ponderada entre todos os que passaram pelo treino, mas o voto de cada um pesa proporcional a quão bem aquele lutador específico se saiu no PRÓPRIO treino. O lutador que resolveu golpes difíceis com precisão tem mais peso na decisão final do que aquele que mal ficou acima da sorte. Ver esse "torneio" acontecendo com números de verdade, peso crescendo nos golpes mais difíceis a cada rodada, em `boosting.py`.

## Vocabulário básico

- Estimador fraco (weak learner): um modelo só um pouco melhor que adivinhar aleatoriamente (mais de 50% de acerto numa classificação binária, mas não muito mais). No AdaBoost, o estimador fraco quase sempre é um "stump", uma árvore de decisão de profundidade 1 (uma pergunta só).
- Peso do exemplo (`w_i`): o quanto aquele exemplo específico "pesa" na hora de medir o erro e treinar o próximo estimador. Começa igual pra todo mundo e cresce pros exemplos que o comitê, até agora, ainda erra.
- Erro ponderado (`ε_t`, épsilon do round t): a fração de peso total que o estimador daquele round errou, não a contagem crua de erros. Um exemplo com peso alto que é errado pesa muito mais nessa conta do que um exemplo de peso baixo.
- Peso de voto do estimador (`α_t`, alfa do round t): o quanto a opinião daquele estimador específico conta na votação final. Quanto menor o erro ponderado do estimador, maior o `α_t`.
- Estimador forte (strong learner): o comitê inteiro, a combinação ponderada de todos os estimadores fracos. É a promessa central do boosting: uma sequência de estimadores fracos, cada um mal acima da sorte, pode virar coletivamente um classificador bem preciso.

## O algoritmo do AdaBoost, passo a passo

Considerando rótulos codificados como +1 e -1 (não 0 e 1, é a convenção que faz a matemática do AdaBoost funcionar de forma limpa):

1. **Inicializar os pesos**: todo exemplo começa com o mesmo peso, `w_i = 1/n`, onde `n` é o número de exemplos de treino. Ninguém é "difícil" ainda, é o primeiro round.
2. **Treinar o estimador fraco da rodada**, usando esses pesos: o algoritmo de treino do estimador (a busca da árvore de decisão pelo melhor corte, por exemplo) recebe os pesos como entrada, e busca a pergunta que minimiza o erro PONDERADO, não o erro cru. Exemplos de peso alto pesam mais nessa busca.
3. **Medir o erro ponderado** desse estimador:

```
ε_t = (soma dos pesos dos exemplos que o estimador t errou) / (soma de todos os pesos)
```

4. **Calcular o peso de voto do estimador**:

```
α_t = (1/2) * ln((1 - ε_t) / ε_t)
```

Repare no comportamento dessa fórmula: se `ε_t` é pequeno (o estimador acertou quase tudo, pesado pelos pesos), `α_t` fica grande e positivo, esse estimador vai pesar bastante na votação final. Se `ε_t` se aproxima de 0,5 (o estimador está tão bom quanto chute aleatório numa classificação binária), `α_t` se aproxima de 0, quase não conta na votação. Se por acaso `ε_t` passa de 0,5 (o estimador acerta menos que o chute aleatório), `α_t` fica negativo, o que na prática inverte o voto desse estimador, ele erra tão sistematicamente que "virar ele do avesso" ajuda mais do que ignorá-lo.

5. **Atualizar o peso de cada exemplo**, aumentando o peso de quem o estimador atual errou e diminuindo o peso de quem ele acertou:

```
w_i <- w_i * exp(-α_t * y_i * pred_t(x_i))
```

Como `y_i` e `pred_t(x_i)` são +1 ou -1, o produto `y_i * pred_t(x_i)` vale +1 quando o estimador acertou aquele exemplo, e -1 quando errou. Isso faz o expoente virar `-α_t` (acerto: peso multiplicado por um número menor que 1, cai) ou `+α_t` (erro: peso multiplicado por um número maior que 1, sobe). Depois dessa atualização, os pesos são normalizados de novo pra somar 1, senão eles só cresceriam sem parar.
6. **Repetir os passos 2 a 5** por `T` rodadas (um número escolhido antes de começar, o `n_estimators`), treinando um novo estimador a cada rodada, sempre em cima dos pesos mais recentes.
7. **Combinar tudo numa previsão final**, pela votação ponderada por `α_t` de cada estimador:

```
previsão final(x) = sinal( Σ α_t * pred_t(x) )
```

Ver essas sete contas acontecendo de verdade, rodada por rodada, com pesos crescendo nos lutadores mais difíceis, em `boosting.py`.

## Por que um estimador "fraco" basta

Essa é a virada de chave conceitual do boosting: não é preciso que cada estimador individual seja bom, só que seja um pouco melhor que chute aleatório (`ε_t < 0,5` na classificação binária). O teorema de boosting (uma das ideias fundadoras da teoria PAC de aprendizado) garante que, sob certas condições, uma sequência desses estimadores fracos, combinados dessa forma ponderada, converge pra um classificador com erro tão baixo quanto se queira, dado rodadas suficientes. É essa garantia formal que separa o boosting de "só treinar um monte de modelos ruins e torcer": tem prova matemática por trás.

Isso também explica por que o estimador base do boosting costuma ser bem mais simples que o do bagging: um stump de profundidade 1 seria um péssimo classificador sozinho, mas é exatamente esse tipo de estimador simples e de viés alto (underfitting sozinho) que o boosting sabe aproveitar, corrigindo o viés dele rodada a rodada.

## Boosting reduz viés, bagging reduz variância

Essa contraposição é o resumo mais cobrado em prova sobre a diferença entre os dois:

| | Bagging | Boosting (AdaBoost) |
|---|---|---|
| Treino dos estimadores | Paralelo, independente | Sequencial, cada um depende do anterior |
| Estimador base ideal | Forte e instável (alta variância), ex.: árvore profunda sem poda | Fraco e enviesado (alto viés), ex.: stump de profundidade 1 |
| Agregação final | Voto simples (todo estimador pesa igual) | Voto ponderado por `α_t` (estimador melhor pesa mais) |
| O que reduz | Variância | Viés (e também reduz variância, mas o ganho principal é no viés) |
| Risco de mais estimadores | Baixo: o desempenho tende a estabilizar, dificilmente piora | Existe risco real de overfitting com rodadas demais, principalmente em dados ruidosos |

O risco de overfitting do boosting merece atenção: como o algoritmo aumenta o peso de exemplos que continuam errados, um exemplo genuinamente ruidoso ou mal rotulado (que nenhum padrão real explica) vai acumulando peso cada vez maior a cada rodada, porque nenhum estimador consegue acertá-lo de forma consistente. Rodadas demais podem fazer o comitê inteiro se contorcer tentando decorar justamente esse ruído, o oposto do que acontece no bagging, onde adicionar mais estimadores raramente piora as coisas. Ver o comportamento do erro em função do número de rodadas, tanto no exemplo de brinquedo quanto no dataset de fraude, em `boosting.py`.

## Extensão: Gradient Boosting

O AdaBoost é um caso particular de uma ideia mais geral chamada Gradient Boosting. Pra entender a ideia geral, vale voltar num conceito básico de otimização: descida de gradiente.

### O que é descida de gradiente, rapidinho

Imagine que você está numa neblina total, em pé numa encosta, e quer chegar no ponto mais baixo do vale sem enxergar nada, só sentindo a inclinação do chão debaixo dos pés. A estratégia óbvia: a cada passo, sente pra que lado o chão desce mais rápido, e dá um passo curto nessa direção. Repete isso várias vezes, e mesmo sem nunca ver o vale inteiro, você geralmente acaba chegando perto do ponto mais baixo. É essa a ideia de descida de gradiente: otimizar uma função (achar o ponto onde ela é mínima) dando passos curtos e repetidos na direção que mais reduz o valor da função naquele ponto, medida pelo gradiente (a versão da derivada quando existe mais de uma variável envolvida).

Numa rede neural, por exemplo, a "encosta" é a função de perda (o quanto o modelo erra) e os "passos" ajustam os PESOS da rede, um número finito de parâmetros. O Gradient Boosting faz uma coisa parecida, só que dá o passo num espaço mais estranho: em vez de otimizar parâmetros de um modelo fixo, ele otimiza a PREVISÃO em si, tratando a previsão `F(x_i)` de cada exemplo de treino como se fosse "um parâmetro a ajustar" separadamente. Por isso a expressão "descida de gradiente no espaço de funções": o "passo" de cada rodada não é um número, é uma função nova inteira (uma árvore), somada ao comitê.

### Como isso vira uma árvore nova por rodada: o resíduo

Concretamente, pra cada rodada:

1. Calcula-se o gradiente da função de perda em relação à previsão atual do comitê, pra cada exemplo de treino. No caso mais simples (erro quadrático, comum em regressão), esse gradiente tem uma cara bem intuitiva: é exatamente `-(y_i - F(x_i))`, o negativo do resíduo, a diferença entre o valor real e o que o comitê já está prevendo.
2. Treina-se uma árvore nova pra prever esses "pseudo-resíduos" (o gradiente calculado no passo 1) a partir de `x`, não pra prever `y` diretamente.
3. Soma-se essa árvore nova ao comitê, multiplicada por uma taxa de aprendizado pequena (o "tamanho do passo" da descida de gradiente, ver a seção de regularização logo abaixo).

Repetindo isso rodada após rodada, cada árvore nova está literalmente aprendendo a prever "o que ainda falta" pra acertar, o pedaço do erro que o comitê atual ainda não explica. É uma forma diferente de "focar no que sobrou difícil" do que o reponderamento explícito do AdaBoost (pesos `w_i` subindo pros exemplos errados), mas a intenção é a mesma: cada nova árvore corrige especificamente o que o comitê atual erra. Pra classificação (usando log-loss, a função de perda mais comum na prática, ou até a própria perda exponencial do AdaBoost) a matemática do gradiente fica mais elaborada, mas a lógica de "treinar em cima do gradiente da perda atual" continua igual. Essa generalização pra qualquer função de perda diferenciável é o que torna o Gradient Boosting mais flexível que o AdaBoost, e é a base de implementações modernas muito usadas na prática como XGBoost, LightGBM e CatBoost.

### Por que XGBoost e afins dominam a prática, apesar do risco de overfitting

Vale separar duas coisas que às vezes se confundem. O mecanismo central do boosting (focar rodada a rodada no que ainda está errado) reduz principalmente **viés**, isso é uma força, não um defeito. O risco de overfitting mencionado na seção anterior não vem desse mecanismo em si, vem de rodadas DEMAIS: com rodadas suficientes, o comitê acumula complexidade (variância) suficiente pra decorar ruído específico do treino, o sintoma clássico de excesso de variância, mesmo tendo nascido de um mecanismo redutor de viés. Ou seja, `n_estimators` percorre a curva inteira de viés-variância: poucas rodadas, muito viés (o comitê ainda mal generaliza o padrão); rodadas demais, muita variância (o comitê já decorou ruído); o ponto ótimo fica no meio, e é justamente esse ponto que a curva de F1 treino x teste em `boosting.py` deixa visível.

XGBoost, LightGBM e CatBoost não mudam esse mecanismo central, ainda são Gradient Boosting de árvores por baixo dos panos. A diferença é que eles empacotam, de fábrica, um arsenal de ferramentas específicas pra controlar exatamente esse risco de variância crescente, tornando o ponto ótimo mais fácil de achar e mais estável de manter:

- **Taxa de aprendizado pequena (`learning_rate`, também chamada de shrinkage)**: cada árvore nova entra multiplicada por um fator pequeno (tipo 0,01 a 0,3), não em peso cheio. Isso desacelera o aprendizado de propósito: precisa de mais rodadas pra convergir, mas cada rodada individual "empurra menos" o comitê, deixando a trajetória de treino mais suave e menos propensa a se contorcer atrás de ruído específico.
- **Regularização explícita na função objetivo**: XGBoost soma, na própria conta que otimiza, uma penalidade pelo tamanho e pela complexidade de cada árvore nova (número de folhas, magnitude dos valores nas folhas, ao estilo L1/L2 de uma regressão regularizada). Uma árvore nova só entra no comitê se o ganho de perda realmente compensar essa penalidade, o que já desestimula boa parte do overfitting antes mesmo de ele acontecer.
- **Subamostragem de linhas e colunas** (`subsample`, `colsample_bytree`): cada árvore nova treina só numa fração aleatória dos exemplos e dos atributos, exatamente o mesmo truque do bagging (linhas) e do random forest (colunas), emprestado pra dentro do boosting. Isso injeta a mesma decorrelação que reduz variância no bagging, só que agora combinada com o mecanismo de correção sequencial de viés do boosting, o melhor dos dois mundos.
- **Parada antecipada (early stopping)**: monitorar o desempenho num conjunto de validação a cada rodada e parar assim que ele para de melhorar (ou piora), na prática automatizando exatamente a leitura do gráfico `curva_f1_rodada.png` deste script: parar de somar rodadas no ponto em que a curva de teste estaciona.
- **Árvores rasas como estimador fraco** (`max_depth` tipicamente entre 3 e 8): cada árvore individual já nasce limitada, então mesmo sem os outros controles, nenhuma árvore sozinha consegue decorar padrões complexos demais do treino.

Some a isso um motivo bem menos teórico e bem mais prático: dados tabulares (linhas e colunas, tipo o próprio dataset de fraude usado aqui) são o tipo de dado onde árvores de decisão boostadas consistentemente superam outras famílias de modelo, incluindo redes neurais, em benchmarks e competições de dados tabulares. Some ainda otimizações de engenharia (busca de corte por histograma, crescimento das árvores folha-a-folha em vez de nível-a-nível no LightGBM, suporte nativo a valores faltantes, paralelismo e uso de GPU) que tornam essas bibliotecas rápidas o bastante pra rodar em datasets grandes de verdade. A base teórica continua sendo a mesma deste capítulo, boosting de estimadores fracos com foco sequencial no erro, só que com bastante regularização de fábrica e muita engenharia de performance em cima.

## Aspectos positivos e negativos

- (+) Constrói um classificador forte a partir de estimadores fracos e simples, com garantia teórica de convergência.
- (+) Costuma atingir viés bem mais baixo que um único estimador fraco, sem precisar de um estimador base complexo.
- (+) `α_t` dá um jeito natural de "confiar mais" nos estimadores que se saíram melhor, em vez de tratar todo mundo igual.
- (-) Treino inerentemente sequencial: ao contrário do bagging, não dá pra paralelizar o treino dos estimadores entre si.
- (-) Sensível a ruído e outliers: exemplos mal rotulados acumulam peso cada vez maior e podem distorcer rodadas seguintes.
- (-) Risco real de overfitting com `n_estimators` grande demais, ao contrário do bagging.
- (-) Perde a interpretabilidade fácil de um estimador único, ainda mais que o bagging (cada estimador pesa diferente na decisão final).

## Ver também

- `boosting/boosting.py`: AdaBoost calculado na mão num torneio de brinquedo com 10 lutadores, pesos crescendo nos lutadores mais difíceis a cada rodada, e o treino de verdade no dataset de fraude, incluindo o erro do comitê em função do número de rodadas (`n_estimators`).
- `01-supervisionado/bagging/notes/anotacoes.md`: teoria do bagging, o contraponto direto do boosting na família de comitês de classificadores.
- `03-outros/comites_classificadores/notes/anotacoes.md`: teoria geral de comitês (votação com peso, seleção estática x dinâmica), o guarda-chuva teórico por cima desta nota.
