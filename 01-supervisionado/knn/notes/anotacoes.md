# k-NN

- Objetos com características semelhantes pertencem ao mesmo grupo.
- Algoritmo que pode ser utilizado para classificação ou regressão.
- Algoritmo lazy (preguiçoso) ou baseado em memória (instance-based learning): todo o processo de "aprendizado" consiste em apenas memorizar os objetos, sem construir modelo nenhum durante o treino.
- Naturalmente incremental: chegou exemplo novo, só atualiza a memória (adiciona à lista), não precisa retreinar nada.

## Analogia central: a taverna que recruta pra guilda

Chega um aventureiro novo, sem ficha de classe definida. O taverneiro não faz um interrogatório tipo Chapéu Seletor, pergunta atrás de pergunta (essa é a mecânica da árvore de decisão). Ele faz outra coisa: olha pros K aventureiros já cadastrados que mais PARECEM com o novato (força parecida, agilidade parecida) e copia a classe que a maioria deles tem. "Diga-me com quem você anda parecido, e eu digo quem você é." Essa é a ideia inteira do k-NN, sem enfeite.

Repare na diferença de mecânica em relação a uma árvore: a árvore aprende perguntas durante o treino e depois é rápida pra decidir. O k-NN não aprende pergunta nenhuma, ele só guarda a lista cadastrada (o treino é só isso, memorizar) e faz toda a conta pesada na hora de classificar alguém novo, comparando com todo mundo que já está na lista.

### Passo a passo pra classificar um caso novo

1. Recebe o caso novo, com os atributos medidos mas sem classe.
2. Calcula a distância dele até TODOS os exemplos já cadastrados (não existe atalho: precisa medir com todo mundo).
3. Ordena essa lista do mais perto pro mais longe.
4. Pega só os K primeiros da fila (os K vizinhos mais próximos).
5. Vota: a classe mais comum entre esses K vizinhos vence, e essa é a previsão. Numa regressão (prever um número, não uma classe), troca-se o voto pela média dos valores dos K vizinhos.

## Distâncias

- Distância mais simples é a euclidiana (`d(a,b) = sqrt(Σ(a_i - b_i)²)`), mas podemos utilizar outras: Manhattan (`Σ|a_i - b_i|`, anda em quarteirão, sem diagonal), Minkowski (generalização das duas: `(Σ|a_i - b_i|^p)^(1/p)`, p=1 vira Manhattan, p=2 vira Euclidiana), Hamming (atributos categóricos: conta em quantas posições os dois exemplos diferem).
- Costuma exigir normalização: como a distância soma diferenças de TODOS os atributos juntos, um atributo em escala muito maior que os outros atropela sozinho a conta da distância, mesmo sem ter nada a ver com a classe do exemplo. É como comparar duas pessoas e deixar a "altura em milímetros" atropelar completamente o "peso em quilos" só porque o número é maior. Por isso todo atributo precisa estar na mesma escala antes de calcular distância (Min-Max ou Z-score).

## Superfície de decisão

- Superfícies de decisão podem ser complexas: no caso 1-NN, o espaço se divide num diagrama de Voronoi, poliedros convexos com centro em cada objeto de treino (cada célula é a região mais perto daquele objeto do que de qualquer outro).
- Pra k>1, a fronteira final é a fusão dessas células por classe majoritária, ficando mais suave (menos irregular) conforme k cresce.

## Escolha de K

- k é o número de vizinhos votantes (classificação), ou número de vizinhos usados pra fazer a média ou a mediana (regressão): média se o erro a minimizar for quadrático, mediana se for desvio absoluto.
- Valor de k costuma ser pequeno e ímpar (ímpar evita empate de voto em problema de 2 classes).
- K PEQUENO (ex.: K=1): a previsão depende só do vizinho mais próximo. Fronteira de decisão bem irregular, recortada, sensível a ruído e outlier (um único exemplo mal rotulado no treino já muda a resposta). Tende a overfitting.
- K GRANDE: a previsão passa a somar votos de vizinhos cada vez mais distantes, então a fronteira fica mais suave, mas o padrão local se dilui. No limite (K = todos os exemplos), o k-NN sempre responde a classe mais comum do dataset inteiro, ignorando o caso novo por completo. Tende a underfitting, e é particularmente ruim em dados DESBALANCEADOS: a classe rara (como fraude) quase nunca ganha votação numérica se K for grande.
- Não existe um K universal: na prática, testam-se vários valores de K com validação cruzada e fica-se com o que generaliza melhor.

## Voto ponderado por distância

Na votação simples, todo vizinho vale 1 voto, não importa se está colado no caso novo ou quase saindo da lista dos K. Uma variação mais esperta pesa cada voto pelo INVERSO da distância (1/distância): quem está mais perto pesa mais. Isso ajuda a resolver empates e reduz a influência de vizinhos "só de raspão" que entraram nos K por pouco, sem precisar diminuir K.

## Garantia teórica (Cover & Hart, 1967)

- Erro assintótico (n -> infinito) do 1-NN é majorado pelo dobro do erro do classificador Bayesiano ótimo: `R_1NN <= 2 R* (1 - R*) <= 2 R*`, onde R* é a taxa de erro do classificador ótimo (o menor erro possível, dado quanto as classes já se sobrepõem nos dados).
- Fazendo k crescer junto com n, mas mantendo k/n -> 0, o erro do k-NN tende pro erro de Bayes ótimo. Ou seja: com dados infinitos e k grande o bastante (mas pequeno perto de n), o k-NN se aproxima do melhor classificador teoricamente possível.

## Maldição da dimensionalidade

Imagine que você quer achar, numa multidão, alguém "parecido" com você. Comparando só pela altura, é fácil: um bocado de gente vai estar pertinho da sua altura. Agora compare por altura, peso, idade, tamanho do pé e cor dos olhos ao mesmo tempo: pra alguém ser parecido com você agora, precisa estar perto em TODAS essas características ao mesmo tempo, não só numa, o que é muito mais raro. Quanto mais características você adiciona, mais difícil fica achar alguém realmente parecido em tudo, e as pessoas começam a parecer todas mais ou menos igualmente "distantes" de você, até as que antes pareciam próximas.

É isso que acontece com os dados. Com poucos atributos, "estar perto" quer dizer alguma coisa de verdade. Conforme cresce o número de atributos:

- O espaço definido pelos atributos cresce exponencialmente com o número de atributos.
- A distância ao vizinho mais próximo se aproxima da distância ao vizinho mais afastado (as distâncias colapsam, todo mundo fica "parecido" de longe). Nesse cenário, o conceito de vizinhança perde força, e o k-NN sofre mais do que algoritmos como a árvore de decisão, que escolhe só os atributos mais úteis a cada pergunta.
- Afetado por atributos redundantes ou irrelevantes, que só atrapalham essa conta.
- Recomenda-se aplicar um algoritmo de redução dimensional (ex.: PCA) ou seleção de atributos, dada essa maldição da dimensionalidade.

Pra compensar essa "diluição" e ainda achar vizinhos de verdade, seria preciso uma quantidade de dados que cresce exponencialmente a cada atributo novo, o que raramente é viável na prática. É por isso que "maldição": o custo de ter mais informação (mais atributos) sai caro demais em quantidade de dados necessária.

## Custo computacional

- Treinamento requer pouco esforço computacional (só memoriza os objetos, é instantâneo), mas a inferência precisa calcular a distância do objeto a todos os objetos de treinamento, o que pode ser computacionalmente custoso. É o oposto de uma árvore já treinada, que é rápida pra prever.
- Quando aplicar kd-trees? Kd-tree organiza os exemplos de treino numa árvore que evita comparar com todo mundo, e funciona bem em dimensão baixa a moderada (regra de bolso: até uns 20 atributos). Com muitos atributos, a busca em kd-tree degrada e na prática vira tão lenta quanto força bruta, é a mesma maldição da dimensionalidade batendo na estrutura de busca. Ball tree lida melhor com dimensão mais alta e com métricas de distância diferentes da euclidiana.
- Dado que o teste pode ser lento no k-NN, podemos fazer cherry pick dos exemplos mais representativos: é a ideia de seleção de protótipos, com algoritmos como Condensed Nearest Neighbor (CNN: mantém só os exemplos necessários pra preservar a fronteira de decisão, descarta redundantes) e Edited Nearest Neighbor (ENN: remove exemplos ruidosos, que os próprios vizinhos classificariam errado).

## Relação com raciocínio baseado em casos (CBR)

Qual é a relação do k-NN com o raciocínio baseado em casos? Tem sim: Raciocínio Baseado em Casos (Case-Based Reasoning) é um paradigma mais amplo de IA que resolve um problema novo comparando com casos passados parecidos, seguindo um ciclo de 4 passos (os "4 Rs"): Retrieve (recupera os casos mais parecidos), Reuse (reaproveita a solução deles), Revise (adapta essa solução pro caso novo) e Retain (guarda o caso novo já resolvido, pra usar no futuro).

O k-NN é essencialmente o passo de RETRIEVE do CBR, aplicado a vetores de atributos numéricos com uma métrica de distância formal (euclidiana etc.) e sem os passos de adaptação: k-NN só vota ou tira a média entre os vizinhos, não ajusta a solução de um caso antigo pro caso novo como o CBR costuma fazer em domínios mais complexos (ex.: diagnóstico médico, onde a solução do caso parecido raramente serve pronta, sem adaptação).

## Aspectos positivos e negativos

- (+) Simples de entender e de implementar.
- (+) Não faz suposição sobre o formato dos dados (não assume fronteira linear, por exemplo): se adapta bem a padrões complicados.
- (+) Treino instantâneo (só guarda os dados).
- (-) Previsão cara: precisa guardar o dataset inteiro e medir distância toda vez.
- (-) Muito sensível à escala dos atributos: precisa normalizar antes.
- (-) Sofre com a maldição da dimensionalidade em datasets com muitas colunas.
- (-) Em dados desbalanceados, a classe rara tende a perder a votação (relevante no dataset de fraude do módulo 01), a não ser que se use voto ponderado por distância ou outra correção.

## Ver também

- `knn/knn.py`: contas feitas na mão (distância, votação simples e ponderada, efeito da normalização) com o exemplo dos aventureiros, e o treino de verdade no dataset de fraude.
