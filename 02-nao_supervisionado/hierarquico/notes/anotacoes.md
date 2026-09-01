# Clustering Hierárquico

- Aprendizado não supervisionado: os dados não têm rótulo nenhum. O algoritmo não sabe se existe fraude, se existem 2 grupos ou 20, nada disso; só recebe pontos e tem que descobrir sozinho quem parece com quem.
- Clustering (agrupamento) é encontrar grupos naturais nos dados a partir só da distância entre eles. Clustering hierárquico é uma família de algoritmos de agrupamento que, além de te dar os grupos, te entrega a ÁRVORE de como esses grupos foram formados, do indivíduo isolado até o grupo geral com todo mundo dentro.

## Analogia central: fusão à la Dragon Ball

Lembra da Fusão do Dragon Ball, aquela em que dois guerreiros se combinam num só, mais forte que a soma das partes? O clustering hierárquico aglomerativo (o tipo mais comum, e o que esse repositório implementa) funciona exatamente assim, só que aplicado a pontos de dado: começa com cada ponto sendo o seu próprio "guerreiro" isolado, acha os dois mais parecidos do momento inteiro e funde os dois num "guerreiro-fusão" só. Repete esse processo, sempre fundindo a dupla mais parecida que sobrou (que agora pode ser um guerreiro sozinho ou já um grupo fundido antes), até sobrar um guerreiro-fusão gigante com todo mundo dentro.

Toda essa sequência de fusões, junto com "quão parecidos" cada par estava na hora de se fundir, é o que vira o dendrograma: um diagrama em forma de árvore que mostra, de baixo pra cima, cada fusão e a que altura (distância) ela aconteceu. Ver essa árvore nascendo de verdade, com 6 guerreiros de espécies diferentes (saiyajins, namekuseijins, humanos) se fundindo em ordem, em `hierarquico.py`.

## Vocabulário básico

- Dendrograma: o diagrama em árvore com todas as fusões, do zero (cada ponto sozinho) até o topo (um cluster só). O eixo vertical é a distância (altura) em que cada fusão aconteceu.
- Linkage (ligação): a regra que decide "qual é a distância entre dois CLUSTERES" (não entre dois pontos, entre dois grupos de pontos). É a peça central do algoritmo; tipos diferentes de linkage dão dendrogramas diferentes pros mesmos dados (ver seção abaixo).
- Cortar o dendrograma: escolher uma altura e "serrar" a árvore ali, transformando a árvore inteira numa partição concreta com K clusters. Cortar mais embaixo (perto da raiz das folhas) dá mais clusters, cada um bem específico; cortar mais em cima (perto do topo) dá menos clusters, mais genéricos.
- Aglomerativo (bottom-up): começa com cada ponto no seu próprio cluster e vai fundindo. É o padrão, e o que este repositório implementa.
- Divisivo (top-down): o caminho inverso, começa com todo mundo num cluster só e vai DIVIDINDO recursivamente (parecido com o espírito de uma árvore de decisão, só que sem rótulo pra guiar a divisão). Bem mais raro na prática, porque decidir "qual a melhor forma de dividir um grupo grande" é mais caro computacionalmente do que "qual a dupla mais parecida pra fundir".

## Como o algoritmo aglomerativo funciona

1. Cada ponto começa como seu próprio cluster (se há N pontos, começa com N clusters).
2. Calcula a distância entre TODOS os pares de clusters atuais.
3. Funde o par de clusters com menor distância (o "linkage" define como medir distância entre clusters que já têm mais de um membro).
4. Registra essa fusão (quem fundiu com quem, e a que altura) e volta pro passo 2, agora com um cluster a menos.
5. Repete até sobrar um cluster só, contendo todo mundo.

Repara que isso é um algoritmo guloso (greedy), igual a árvore de decisão: em cada passo funde a MELHOR opção do momento, sem nunca reconsiderar ou desfazer uma fusão já feita. Ver esse loop rodando de verdade, com a matriz de distâncias encolhendo fusão a fusão, em `hierarquico.py`.

## Tipos de linkage (decorar as diferenças, cai em prova)

Dado dois clusters A e B (cada um podendo ter vários pontos dentro), como medir "a distância entre A e B"?

- Ligação simples (single linkage): a MENOR distância entre qualquer ponto de A e qualquer ponto de B. `d(A,B) = min(d(a,b))` pra todo a em A, b em B. Tende a formar clusters compridos e "encadeados" (efeito cadeia/chaining): basta um par de pontos vizinhos ligando dois grupos bem diferentes pra eles serem fundidos cedo demais.
- Ligação completa (complete linkage): a MAIOR distância entre qualquer ponto de A e qualquer ponto de B. `d(A,B) = max(d(a,b))`. O oposto do efeito cadeia: só funde quando até os pontos MAIS distantes dos dois grupos já são parecidos, o que tende a formar clusters mais compactos e de tamanho parecido.
- Ligação média (average linkage): a média de TODAS as distâncias entre um ponto de A e um ponto de B. Fica no meio do caminho entre single e complete.
- Linkage de Ward: não olha pra distância entre pontos diretamente; a cada fusão possível, calcula o quanto a VARIÂNCIA interna dos clusters aumentaria se aquela fusão acontecesse, e escolhe a fusão que aumenta menos essa variância. Na prática tende a gerar clusters bem compactos e de tamanho parecido, e costuma ser a escolha padrão quando não se sabe qual linkage usar (é o que o `AgglomerativeClustering` do scikit-learn usa por padrão).
- Pegadinha de prova: não existe "o melhor linkage" fixo, cada um tem viés próprio (single puxa pra cadeia comprida, complete e Ward puxam pra grupos compactos e parecidos em tamanho). No exemplo de fraude em `hierarquico.py`, os quatro linkages dão números bem diferentes: Ward foi o único que separou uma fatia razoável de fraude num cluster à parte, average teve a MAIOR silhueta (0,74, o cluster "parece" mais limpo geometricamente) mas quase nenhuma relação com a fraude de verdade (ARI perto de 0), a prova viva de que uma métrica interna (silhueta) não garante nada sobre bater com uma estrutura externa que você já conhece.

## Cortando a árvore em K clusters

O dendrograma sozinho não devolve "os clusters", devolve a história INTEIRA de fusões possíveis, de N clusters até 1. Pra sair com uma resposta concreta, corta-se a árvore numa altura: tudo que já tinha se fundido abaixo daquela altura vira um cluster, tudo que ainda não se fundiu continua separado.

- Cortar bem embaixo (perto de altura 0): quase ninguém se fundiu ainda, então sobra quase um cluster por ponto (K alto, clusters bem específicos, pouca generalização).
- Cortar bem em cima (perto da fusão final): quase tudo já se fundiu, sobra pouquíssimos clusters (K baixo, clusters bem genéricos).
- Truque visual pra escolher a altura de corte: procurar o maior "salto" vertical entre duas fusões consecutivas no dendrograma. Um salto grande significa que, pra ir de K clusters pra K-1, foi preciso fundir dois grupos que já estavam bem longe um do outro, sinal de que K clusters era uma divisão mais "natural" dos dados do que K-1.
- Ver esse corte acontecendo na prática, com o exemplo dos 6 guerreiros: cortar em 3 clusters recupera exatamente as 3 espécies (saiyajins, namekuseijins, humanos) que ninguém disse ao algoritmo que existiam, cortar em 2 já funde duas espécies num time só, e cortar em 1 é a fusão final com todo mundo junto (`hierarquico.py`).

## Complexidade computacional (pegadinha de prova)

Calcular a distância entre CADA par de pontos custa O(n²) só pra montar a matriz de distâncias inicial, e o algoritmo completo (recalculando distâncias entre clusters a cada fusão) costuma sair em O(n² log n) ou O(n³) dependendo da implementação. Isso é bem mais caro que k-means, que custa O(n · k · iterações) por rodada. Na prática, isso significa que clustering hierárquico não escala pra datasets grandes: rodar nas quase 285 mil transações do dataset de fraude inteiro é inviável (a matriz de distâncias sozinha teria bilhões de pares). Por isso `hierarquico.py` treina numa amostra pequena (todas as ~500 fraudes mais algumas centenas de transações normais), não no dataset inteiro.

## Aspectos Positivos

- Não precisa decidir o número de clusters ANTES de rodar (diferente do k-means, que exige K de entrada): a árvore inteira fica disponível, e dá pra cortar em qualquer K depois de ver o resultado.
- O dendrograma é uma ferramenta visual rica: mostra não só os grupos finais, mas a estrutura de "quem é mais parecido com quem" em várias escalas ao mesmo tempo.
- Determinístico: rodando de novo com os mesmos dados e o mesmo linkage, sempre dá a mesma árvore (ao contrário do k-means, que depende de inicialização aleatória).

## Aspectos Negativos

- Caro computacionalmente (O(n²) pra cima), não escala pra datasets grandes.
- Guloso: uma fusão feita cedo nunca é desfeita, mesmo que mais tarde ela pareça uma escolha ruim.
- Sensível ao linkage escolhido, e não existe uma resposta única sobre qual usar.
- Sensível a outliers, principalmente com ligação simples (um único ponto ruidoso pode "colar" dois grupos que deveriam ficar separados).

## Ver também

- `hierarquico/hierarquico.py`: fusões calculadas na mão (matriz de distância, ligação simples) com o exemplo dos guerreiros de Dragon Ball, e o treino de verdade numa amostra do dataset de fraude, comparando os quatro tipos de linkage.
