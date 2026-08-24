# k-NN

- Objetos com características semelhantes pertencem ao mesmo grupo.
- Algoritmo que pode ser utilizado para classificação ou regressão.
- Algoritmo lazy (preguiçoso) ou baseado em memória (instance-based learning): todo o processo de "aprendizado" consiste em apenas memorizar os objetos, sem construir modelo nenhum durante o treino.
- Naturalmente incremental: chegou exemplo novo, só atualiza a memória (adiciona à lista), não precisa retreinar nada.

## Distâncias

- Distância mais simples é a euclidiana (`d(a,b) = sqrt(Σ(a_i - b_i)²)`), mas podemos utilizar outras: Manhattan (`Σ|a_i - b_i|`, anda em quarteirão, sem diagonal), Minkowski (generalização das duas: `(Σ|a_i - b_i|^p)^(1/p)`, p=1 vira Manhattan, p=2 vira Euclidiana), Hamming (atributos categóricos: conta em quantas posições os dois exemplos diferem).
- Costuma exigir normalização: como a distância soma diferenças de todos os atributos, um atributo em escala maior atropela os outros na conta se não estiverem na mesma escala (Min-Max ou Z-score).

## Superfície de decisão

- Superfícies de decisão podem ser complexas: no caso 1-NN, o espaço se divide num diagrama de Voronoi, poliedros convexos com centro em cada objeto de treino (cada célula é a região mais perto daquele objeto do que de qualquer outro).
- Pra k>1, a fronteira final é a fusão dessas células por classe majoritária, ficando mais suave (menos irregular) conforme k cresce.

## Escolha de K

- k é o número de vizinhos votantes (classificação), ou número de vizinhos usados pra fazer a média ou a mediana (regressão): média se o erro a minimizar for quadrático, mediana se for desvio absoluto.
- Valor de k costuma ser pequeno e ímpar (ímpar evita empate de voto em problema de 2 classes).
- Estimar k por validação cruzada, ou associar um peso à contribuição de cada vizinho (voto ponderado por 1/distância, em vez de todo vizinho valer 1 voto igual).

## Garantia teórica (Cover & Hart, 1967)

- Erro assintótico (n -> infinito) do 1-NN é majorado pelo dobro do erro do classificador Bayesiano ótimo: `R_1NN <= 2 R* (1 - R*) <= 2 R*`, onde R* é a taxa de erro do classificador ótimo (o menor erro possível, dado quanto as classes já se sobrepõem nos dados).
- Fazendo k crescer junto com n, mas mantendo k/n -> 0, o erro do k-NN tende pro erro de Bayes ótimo. Ou seja: com dados infinitos e k grande o bastante (mas pequeno perto de n), o k-NN se aproxima do melhor classificador teoricamente possível.

## Maldição da dimensionalidade

- O espaço definido pelos atributos de um problema cresce exponencialmente com o número de atributos.
- Com o aumento da dimensionalidade, a distância ao vizinho mais próximo se aproxima da distância ao vizinho mais afastado (as distâncias colapsam, todo mundo fica "parecido" de longe).
- Afetado por atributos redundantes ou irrelevantes, que só atrapalham essa conta.
- Recomenda-se aplicar um algoritmo de redução dimensional (ex.: PCA) ou seleção de atributos, dada essa maldição da dimensionalidade.

## Custo computacional

- Treinamento requer pouco esforço computacional (só memoriza os objetos), mas a inferência precisa calcular a distância do objeto a todos os objetos de treinamento, o que pode ser computacionalmente custoso.
- Quando aplicar kd-trees? Kd-tree organiza os exemplos de treino numa árvore que evita comparar com todo mundo, e funciona bem em dimensão baixa a moderada (regra de bolso: até uns 20 atributos). Com muitos atributos, a busca em kd-tree degrada e na prática vira tão lenta quanto força bruta, é a mesma maldição da dimensionalidade batendo na estrutura de busca. Ball tree lida melhor com dimensão mais alta e com métricas de distância diferentes da euclidiana.
- Dado que o teste pode ser lento no k-NN, podemos fazer cherry pick dos exemplos mais representativos: é a ideia de seleção de protótipos, com algoritmos como Condensed Nearest Neighbor (CNN: mantém só os exemplos necessários pra preservar a fronteira de decisão, descarta redundantes) e Edited Nearest Neighbor (ENN: remove exemplos ruidosos, que os próprios vizinhos classificariam errado).

## Relação com raciocínio baseado em casos (CBR)

Qual é a relação do k-NN com o raciocínio baseado em casos? Tem sim: Raciocínio Baseado em Casos (Case-Based Reasoning) é um paradigma mais amplo de IA que resolve um problema novo comparando com casos passados parecidos, seguindo um ciclo de 4 passos (os "4 Rs"): Retrieve (recupera os casos mais parecidos), Reuse (reaproveita a solução deles), Revise (adapta essa solução pro caso novo) e Retain (guarda o caso novo já resolvido, pra usar no futuro).

O k-NN é essencialmente o passo de RETRIEVE do CBR, aplicado a vetores de atributos numéricos com uma métrica de distância formal (euclidiana etc.) e sem os passos de adaptação: k-NN só vota ou tira a média entre os vizinhos, não ajusta a solução de um caso antigo pro caso novo como o CBR costuma fazer em domínios mais complexos (ex.: diagnóstico médico, onde a solução do caso parecido raramente serve pronta, sem adaptação).
