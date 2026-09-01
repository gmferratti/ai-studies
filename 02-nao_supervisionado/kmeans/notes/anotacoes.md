# K-means

- Aprendizado não supervisionado, igual o clustering hierárquico: não existe rótulo pra guiar o algoritmo, ele só recebe pontos e um número K, e tem que descobrir sozinho como dividir esses pontos em K grupos que façam sentido.
- Diferente do clustering hierárquico (que constrói uma árvore inteira de fusões), o k-means já entrega direto uma partição final em K grupos, sem árvore nenhuma no meio do caminho.

## Analogia central: guerra de território

Imagina duas facções brigando por um mapa cheio de postos avançados espalhados: cada facção tem um quartel-general (HQ), e cada posto se filia ao HQ mais perto dele. Só que o mapa muda: depois que os postos escolheram facção, cada HQ se muda pro centro de gravidade do próprio território (a média das posições de quem é fiel a ele). Só que aí a distância de alguns postos até o HQ (que se mexeu) muda, então alguns postos trocam de facção de novo. Repete essa dança de "escolher facção" e "mudar de sede" até ninguém mais trocar de lado: nesse ponto, os HQs pararam de se mexer porque já estão exatamente no centro dos próprios territórios.

Essa dança de duas etapas é o k-means inteiro: a etapa de atribuição (cada ponto escolhe o centroide mais perto) e a etapa de atualização (cada centroide vira a média dos pontos que escolheram ele). Ver essa dança rodando de verdade, rodada por rodada, com números de coordenada mudando na tela, em `kmeans.py`.

## Vocabulário básico

- Centroide: o "quartel-general" de um cluster, a posição média de todos os pontos atribuídos a ele. No começo é chutado (aleatório ou por alguma regra de inicialização); no fim, é literalmente o centro de massa do próprio grupo.
- K: o número de clusters que você PRECISA informar de antemão. Diferente do clustering hierárquico, o k-means não descobre K sozinho, ele só sabe dividir em exatamente K pedaços, o K que for.
- Inércia (também chamada de WCSS, within-cluster sum of squares): a soma das distâncias ao quadrado de cada ponto até o centroide do seu próprio cluster. É a "régua de bagunça" do k-means: quanto menor, mais compacto (mais parecidos entre si) cada grupo ficou.

  `Inércia = Σ (pra cada cluster k) Σ (pra cada ponto x no cluster k) ||x - centroide_k||²`

- Algoritmo de Lloyd: o nome técnico do algoritmo de "atribuir, recalcular, repetir" descrito acima. É o algoritmo padrão por trás do k-means.
- Convergência: o momento em que nenhum ponto troca mais de cluster entre uma rodada e a próxima (os centroides pararam de se mover). O algoritmo de Lloyd tem uma garantia matemática bacana: a inércia NUNCA aumenta de uma rodada pra outra, sempre cai ou fica igual, o que garante que ele sempre converge (para de se mexer) em algum momento, ainda que não necessariamente na melhor partição possível (ver mínimos locais, abaixo).

## Como o algoritmo funciona

1. Escolhe K posições iniciais de centroide (aleatoriamente, ou com uma regra mais esperta, ver k-means++ abaixo).
2. Etapa de atribuição: cada ponto do dataset é atribuído ao centroide mais próximo dele (por distância euclidiana).
3. Etapa de atualização: cada centroide se move pra média (centro de massa) de todos os pontos que foram atribuídos a ele naquela rodada.
4. Repete os passos 2 e 3 até os centroides pararem de se mover (convergência) ou até bater um número máximo de rodadas.

## Escolhendo K: o método do cotovelo

Como K tem que ser escolhido de antemão, e normalmente ninguém sabe o K "certo" de verdade, uma forma comum de escolher é o método do cotovelo: roda-se o k-means pra vários valores de K (1, 2, 3, ...) e plota-se a inércia final de cada um. A inércia sempre cai conforme K aumenta (no limite, com K = número de pontos, a inércia vira zero, cada ponto é o centroide de si mesmo), então "menor inércia" sozinho não ajuda a escolher K. O que se procura é o "cotovelo" do gráfico: o ponto onde aumentar K deixa de reduzir a inércia de forma significativa, sinal de que os clusters extras que K maior traria não estão comprando organização de verdade, só estão fatiando um grupo que já fazia sentido. Ver esse gráfico rodando no dataset de fraude, em `kmeans.py`.

Outra régua pra escolher K (ou avaliar se ele fez sentido) é a silhueta (silhouette score): pra cada ponto, compara a distância média até os outros pontos do PRÓPRIO cluster com a distância média até os pontos do cluster vizinho mais próximo. Fica entre -1 e 1: perto de 1 é um ponto bem encaixado no próprio grupo e bem longe dos outros grupos, perto de 0 é um ponto na fronteira entre dois grupos, negativo é sinal de que o ponto provavelmente foi parar no cluster errado.

## Inicialização importa: k-means++ e mínimos locais

O algoritmo de Lloyd garante que a inércia só cai (ou empata), nunca sobe, mas isso só garante um MÍNIMO LOCAL, não o melhor agrupamento possível (o mínimo global). Se os centroides iniciais nascerem numa posição ruim (por exemplo, todos colados perto um do outro, no meio do mapa), o algoritmo pode convergir rapidinho pra uma partição pior do que a que sairia de uma inicialização melhor, e ele não tem como "perceber" que ficou preso ali, porque cada rodada individual continua sendo uma melhora local válida.

Duas saídas pra esse problema, usadas juntas pelo `KMeans` do scikit-learn por padrão:

- k-means++: em vez de sortear os K centroides iniciais totalmente ao acaso, sorteia o primeiro centroide aleatoriamente e depois vai sorteando os próximos com probabilidade proporcional à distância ao centroide mais próximo já escolhido. Na prática isso espalha os centroides iniciais pelo mapa em vez de deixá-los colados, reduzindo bastante a chance de cair num mínimo local ruim logo de cara.
- `n_init`: roda o algoritmo inteiro várias vezes (por padrão, com inicializações diferentes) e fica só com o resultado de menor inércia final. É uma defesa estatística contra o risco que sobrou mesmo com k-means++: testando várias vezes, a chance de TODAS as tentativas caírem num mínimo local ruim ao mesmo tempo é bem menor.

## Limitações que caem em prova

- Só enxerga clusters (aproximadamente) esféricos e de tamanho parecido, porque a distância euclidiana até um único centroide não capta formatos alongados, espirais ou clusters de densidade bem diferente. Clustering hierárquico com o linkage certo, ou algoritmos baseados em densidade (DBSCAN), lidam melhor com esses casos.
- Sensível à escala dos atributos, pelo mesmo motivo que o k-NN: um atributo em escala de milhares atropela um atributo em escala de 0 a 1 na conta de distância. Por isso o pipeline de dados de verdade normaliza tudo antes (`utils/data_utils.py`, `StandardScaler`).
- Sensível a outliers: um ponto muito distante puxa a média (o centroide) na direção dele, deformando o cluster inteiro.
- Precisa de K definido de antemão, e escolher K errado (K pequeno demais funde grupos que deveriam ficar separados; K grande demais fatia um grupo que já fazia sentido) muda o resultado inteiro.

## K-means x Clustering hierárquico

| | K-means | Clustering hierárquico |
|---|---|---|
| Precisa saber K antes? | Sim | Não (corta a árvore depois, no K que quiser) |
| Determinístico? | Não (depende da inicialização) | Sim (mesma entrada, mesma árvore) |
| Custo computacional | O(n · K · iterações), escala bem | O(n²) pra cima, não escala bem |
| Formato de cluster que encontra bem | Esférico, tamanho parecido | Depende do linkage, mais flexível |
| Saída | Uma partição final | Uma árvore inteira (dendrograma) |

## Ver também

- `kmeans/kmeans.py`: rodadas de atribuir-e-recalcular feitas na mão com o exemplo de brincadeira dos postos avançados, incluindo o teste de inicialização ruim x k-means++, e o treino de verdade no dataset de fraude com método do cotovelo.
