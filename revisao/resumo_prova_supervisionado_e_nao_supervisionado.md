# Resumo pra prova escrita: Aprendizado Supervisionado e Não Supervisionado

*Redação corrida, pra copiar à mão e ir absorvendo aos poucos, não uma
lista de tópicos pra decorar solta. Cobre os dois temas do edital que já
têm notas completas neste repositório (Aprendizado Supervisionado e
Aprendizado Não Supervisionado), usando os mesmos exemplos de brinquedo
dos scripts (Pokémon, a taverna da guilda, dois reinos rivais, o
anti-cheat de um jogo online, um julgamento estilo Danganronpa, um
torneio de lutadores, uma batalha royale numa floresta, a fusão de
Dragon Ball, uma guerra de território, decks de um card game, o modo
foto de um jogo 3D) como apoio de memória, não decoração. Não cobre
"formas de avaliação" nem PAC-learning, os dois ainda sem notas
completas neste repositório. Cabe em oito páginas manuscritas com letra
normal, dois almaços.*

---

Aprendizado supervisionado é o processo de aprender uma regra geral a
partir de exemplos que já vêm com a resposta certa anotada. Aprendizado
não supervisionado, o segundo tema deste resumo, troca completamente
essa premissa: os exemplos chegam sem gabarito nenhum, e o algoritmo tem
que descobrir sozinho que estrutura existe escondida ali. Aprendizado
por reforço, fora do escopo aqui, aprende de um jeito diferente dos
dois, por tentativa e recompensa. Nos dois temas que interessam pra esse
resumo, o problema nunca foi só decorar os dados já vistos, qualquer
tabela pode ser decorada de cor. O que importa é generalizar: no
supervisionado, produzir uma regra que continue acertando em exemplos
novos, nunca vistos antes; no não supervisionado, encontrar uma
estrutura que reflita algo real sobre os dados, não um padrão que só
existe por acaso naquela amostra específica. É nessa distância entre
decorar e generalizar que mora o fio condutor do texto inteiro.

Formalizando um pouco o lado supervisionado: existe uma regra verdadeira
que se quer aprender, chamada de conceito-alvo, e um espaço de
hipóteses, que é o conjunto de todas as regras que um certo algoritmo é
capaz de considerar. O algoritmo olha pro treino e escolhe, dentro desse
espaço, a hipótese que parece melhor. Só que, com um número finito de
exemplos, quase sempre existe mais de uma hipótese que explica o treino
igualmente bem, e pra escolher entre elas o algoritmo precisa carregar
alguma suposição extra, além dos próprios dados: isso é o viés indutivo.
Uma árvore de decisão carrega o viés de preferir regras mais simples
entre as que resolvem o treino; um SVM com kernel linear carrega o viés
de só considerar fronteiras retas, nem cogita as outras. Sem viés
indutivo nenhum, não haveria como escolher, e é por isso que ele não é
um defeito, é uma condição para aprender qualquer coisa a partir de
exemplos.

Vale abrir um parêntese aqui porque a palavra "viés" reaparece daqui a
pouco com outro sentido, e confundir os dois é um erro clássico de
prova. O viés indutivo, que acabamos de ver, é uma suposição estrutural
sobre que tipo de regra vale a pena considerar. Já o viés no sentido de
viés-variância, o dilema que vai atravessar boa parte deste texto, é uma
medida do quanto um modelo erra sistematicamente por ser simples demais
pra capturar o padrão de verdade. Os dois estão relacionados, porque um
viés indutivo mais restritivo tende a produzir mais desse segundo tipo
de erro sistemático, mas não são a mesma coisa: o primeiro é uma escolha
de desenho do algoritmo, o segundo é uma consequência mensurável dessa
escolha. O erro esperado de qualquer modelo pode ser pensado como a soma
de três parcelas: viés (o erro sistemático de um modelo simples demais),
variância (o quanto o modelo muda de resultado quando o treino muda de
leve, um sintoma de estar ajustado demais às particularidades daquele
treino específico) e um ruído que nenhum modelo consegue eliminar. Um
modelo com viés alto e variância baixa está underfitando, simples demais
pra capturar o padrão real; um modelo com viés baixo e variância alta
está overfitando, decorando detalhes do treino que não generalizam.
Nenhum algoritmo escapa desse trade-off, só escolhe em que ponto dele
prefere operar, e é justamente por isso que não existe um algoritmo
universalmente melhor que os outros para todo problema, o chamado
teorema do No Free Lunch: em média, sobre todos os problemas possíveis,
todo algoritmo se sai igual, e cada um só ganha nos problemas específicos
onde seu viés indutivo particular combina bem com a estrutura real dos
dados.

Os quatro algoritmos clássicos supervisionados deste módulo ilustram
esse dilema cada um à sua maneira, e cada um carrega uma analogia que
ajuda a fixar a mecânica por trás da fórmula. A árvore de decisão
funciona como o Chapéu Seletor de Hogwarts: não escaneia o aluno inteiro
de uma vez e já cospe a casa certa, vai sondando aos poucos, uma
pergunta de cada vez, cada resposta eliminando possibilidades até sobrar
uma decisão só. No exemplo de brinquedo do repositório, 14 Pokémon (de
Tipo Fogo, Água ou Grama, já evoluídos ou não) são separados entre "vale
a pena treinar pra Ginásio" ou não: a primeira pergunta, "Qual o Tipo?",
já deixa o grupo Água inteiro puro, mas Fogo e Grama continuam
misturados, e é dentro desses dois galhos que a árvore repete a mesma
lógica com a segunda pergunta, "Já evoluiu?", até sobrar só folha pura.
Formalmente, a árvore constrói a regra fazendo perguntas em sequência, a
cada passo escolhendo o atributo que mais separa os exemplos em grupos
puros (por entropia ou índice Gini) e repetindo essa lógica dentro de
cada grupo que ainda ficou misturado. Deixada crescer sem limite, ela
consegue separar perfeitamente até o último exemplo do treino, viés
baixíssimo, mas às custas de uma variância alta: pequenas mudanças no
conjunto de treino mudam bastante o formato da árvore final. A poda,
seja limitando a profundidade de antemão ou cortando galhos depois de
crescer tudo, troca de propósito um pouco desse viés por uma redução de
variância bem maior, o que costuma valer a pena. No dataset de fraude
usado nos experimentos deste módulo, a árvore rendeu F1 de 0,7864 na
classe rara.

O k-NN torna esse mesmo trade-off ainda mais explícito, e sua analogia é
a taverna que recruta pra guilda: chega um aventureiro novo, sem ficha
de classe definida, e em vez de um interrogatório tipo Chapéu Seletor, o
taverneiro olha pros K aventureiros já cadastrados que mais PARECEM com
o novato e copia a classe que a maioria deles tem, "diga-me com quem
você anda parecido, e eu digo quem você é". Esse "parecido" mora
diretamente num único parâmetro: o número K de vizinhos consultados na
hora de votar. K pequeno significa que a previsão depende de pouquíssimos
exemplos vizinhos, uma fronteira bem recortada e sensível a ruído, viés
baixo e variância alta; K grande dilui esse voto entre muito mais
vizinhos, suavizando a fronteira até o ponto de prejudicar o
reconhecimento da classe rara, viés mais alto e variância mais baixa.
Diferente da árvore, o k-NN não constrói modelo nenhum durante o treino,
só guarda os exemplos e faz toda a conta na hora de prever, e essa conta
de distância exige atributos normalizados, senão um atributo de escala
maior domina a distância sozinho, mascarando o sinal dos outros. Com
muitos atributos ao mesmo tempo, a própria noção de "vizinho próximo"
perde força, porque todo mundo passa a parecer igualmente longe de todo
mundo, a chamada maldição da dimensionalidade. Com K=3 nesse mesmo
dataset de fraude, o k-NN chegou a F1 de 0,8663, o melhor resultado
entre os quatro algoritmos individuais.

O SVM ataca o mesmo dilema por outro ângulo, com a analogia da linha de
frente entre dois reinos rivais: em vez de ajustar um parâmetro de
vizinhança, ele escolhe deliberadamente a fronteira mais segura possível
entre as tropas dos dois lados, a que fica o mais longe possível da
tropa mais avançada de cada reino ao mesmo tempo, a margem máxima, como
um general cuidadoso decidindo onde cravar a fronteira na terra de
ninguém. Só os exemplos bem na linha de frente, os vetores de suporte,
decidem essa fronteira; mover qualquer outro exemplo mais atrás não muda
nada. Existe embasamento formal (a teoria do aprendizado estatístico de
Vapnik) mostrando que uma margem mais larga corresponde a um modelo de
menor complexidade, ou seja, menos variância, sem precisar contar
atributos nem ajustar complexidade manualmente. Como poucos problemas
reais são perfeitamente separáveis, existe uma versão "suave" dessa
margem, controlada por um parâmetro que decide o quanto custa tolerar um
exemplo mal posicionado: valor alto empurra o modelo pra tentar acertar
tudo, aumentando variância e encolhendo a margem; valor baixo aceita
mais erro de treino em troca de uma fronteira mais estável. Pra
fronteiras curvas, o truque do kernel calcula a semelhança entre
exemplos como se eles tivessem sido projetados num espaço com mais
dimensões, sem nunca fazer essa projeção cara de verdade. No dataset de
fraude, a versão linear do SVM rendeu F1 de 0,69, com precisão alta mas
recall baixo: quando aponta fraude geralmente acerta, mas deixa passar
bastante fraude de verdade, um sinal de que a fronteira reta, sendo
simples demais pra esse problema específico, carrega viés alto aqui.

O Naive Bayes é o caso mais extremo de viés alto por escolha, e funciona
como o sistema anti-cheat de um jogo online: em vez de seguir um
fluxograma de perguntas tipo árvore de decisão, ele junta várias pistas
de comportamento e calcula, pra cada classe (bot ou humano, fraude ou
normal), o quão provável ela já era de antemão multiplicado por quão bem
cada pista observada combina com aquela classe, supondo, de forma
deliberadamente ingênua, que as pistas não se influenciam entre si dado
a classe. Essa suposição raramente é verdadeira, mas tem uma vantagem
prática grande: um modelo tão simples tem variância muito baixa, precisa
de pouquíssimos exemplos pra estimar suas probabilidades e não overfita
com facilidade. O problema aparece quando o viés dessa suposição
simplificada não bate com a realidade dos dados, exatamente o que
aconteceu no dataset de fraude, onde a classe rara não segue de fato uma
distribuição de sino limpa: o resultado foi um recall alto mas uma
precisão baixíssima, F1 de apenas 0,11, o pior entre os quatro
algoritmos individuais, uma consequência direta de um viés que não
combinou com o problema.

Como nenhum desses quatro algoritmos resolve o dilema, só escolhe um
ponto diferente dele, faz sentido que a próxima ideia seja não escolher
um só, e sim combinar vários. Essa é a lógica dos comitês (bagging,
boosting e random forest): em vez de apostar tudo numa única hipótese,
treinar várias e deixar elas votarem. Só que isso só ajuda se os membros
do comitê discordarem de verdade entre si; se todos erram exatamente nos
mesmos exemplos, combinar as opiniões não muda nada. O interessante é
que cada uma das três técnicas ataca um lado diferente e específico do
dilema viés-variância.

O bagging treina vários modelos instáveis (tipicamente árvores sem
poda, viés baixo e variância alta), cada um numa amostra sorteada com
reposição do mesmo dataset original, de forma totalmente independente e
em paralelo, e depois tira um voto simples entre eles, igual o
julgamento de um anime de mistério escolar estilo Danganronpa: cada
investigador recebe, por sorteio, um monte de pistas tiradas ao acaso do
baralho de evidências (podendo até repetir pista), alguns chegam numa
conclusão errada por puro azar do sorteio, mas a turma inteira não
decide por um investigador só, todo mundo vota e vence a maioria. O
efeito é reduzir a VARIÂNCIA sem mexer no viés: os erros "por sorte da
amostra" de cada árvore individual tendem a apontar em direções
diferentes e se cancelam parcialmente quando agregados, mas se o modelo
base já carrega um viés sistemático, o comitê inteiro herda esse mesmo
viés. No dataset de fraude, o bagging chegou a F1 de 0,8770, o melhor
resultado deste módulo inteiro. O random forest é a mesma receita com um
ingrediente a mais, e sua analogia é uma batalha royale numa floresta de
verdade: se existisse uma arma claramente mais forte que qualquer outra
no mapa (um atributo campeão disparado), praticamente todo grupo ia
correr atrás dela primeiro, e a estratégia de quase todo grupo acabaria
idêntica, exatamente o que acontece com o bagging puro quando um
atributo domina, quase toda árvore do comitê aprende a mesma primeira
pergunta. O random forest força esse atributo a ficar de fora em
algumas divisões, sorteando só um punhado dos atributos disponíveis em
cada uma, o que deixa as árvores genuinamente mais diferentes entre si e
derruba a variância ainda mais que o bagging puro, na condição de
existir mesmo essa concentração de força num só atributo. No dataset de
fraude de verdade essa condição não se confirmou (os 30 componentes de
PCA já vêm razoavelmente decorrelacionados entre si, sem nenhum campeão
disparado), e por isso `max_features=None` (F1 de 0,8877) bateu tanto
`max_features='sqrt'` (0,8743, o valor usado como padrão da família)
quanto `max_features=3` (0,8729): não é bug, é o contraponto real e
honesto ao exemplo de brinquedo, restringir atributos só custa viés
quando não existe decorrelação suficiente pra comprar de volta.

O boosting inverte a lógica: em vez de treinar em paralelo, treina em
sequência, com a analogia do torneio de lutadores contra um campeão
invicto. O primeiro lutador que a escola manda pra treinar contra o
campeão é só um pouco melhor que chute aleatório, um especialista fraco,
que já acerta a estratégia certa contra alguns golpes do campeão mas
apanha feio dos outros; a escola não manda o próximo lutador
aleatoriamente, olha exatamente ONDE o primeiro apanhou mais e treina o
segundo focado bem naqueles golpes específicos, e assim por diante. A
decisão final também é uma votação, mas ponderada: um modelo que se saiu
melhor no próprio treino pesa mais na decisão final do que um que mal
passou de sorte. Encadeando vários desses estimadores fracos dessa
forma, o comitê inteiro consegue reduzir o VIÉS de forma consistente, ao
contrário do bagging. O preço é que o boosting pode aumentar a variância
se for longe demais: com rodadas em excesso, um exemplo genuinamente
ruidoso continua acumulando peso sem parar, e o comitê acaba se
contorcendo pra tentar acertar justamente esse ruído, um overfitting de
um tipo que o bagging dificilmente sofre. No dataset de fraude, isso
apareceu de forma bem concreta: o desempenho no próprio treino continuou
subindo rodada após rodada enquanto o desempenho no teste estacionou bem
antes, e o F1 final ficou em 0,7347, abaixo dos outros comitês. As
versões mais usadas na prática hoje em dia, como XGBoost e LightGBM,
seguem essa mesma ideia de corrigir erro rodada a rodada, mas vêm de
fábrica com vários mecanismos de controle (taxa de aprendizado pequena,
penalidades de complexidade, parada antecipada) pensados especificamente
pra segurar esse risco de variância crescendo demais.

O mesmo dilema que aparece formalizado logo no início, viés contra
variância, explica tanto por que cada algoritmo individual erra do jeito
que erra quanto por que os comitês existem e funcionam do jeito que
funcionam, e o próprio fato de existirem tantos algoritmos e formas de
combiná-los, cada um bom em pontos diferentes desse trade-off, é a prova
prática do teorema do No Free Lunch: se um método resolvesse tudo de
vez, não precisaríamos de nenhum dos outros. Esse mesmo teorema vale,
sem nenhuma modificação, do outro lado da moeda: aprendizado não
supervisionado, onde não existe rótulo nenhum guiando o algoritmo.

Sem rótulo, o problema muda de figura, mas o viés indutivo continua tão
necessário quanto antes: só que agora, em vez de escolher uma hipótese
pra prever um rótulo, o algoritmo precisa escolher uma noção própria do
que conta como "padrão que importa", e essa escolha (uma distância, um
suporte mínimo, uma direção no espaço) já é, sozinha, um viés, sem
nenhum gabarito posterior pra confirmar se ele acertou. Os quatro
algoritmos deste segundo tema respondem cada um a uma pergunta diferente
sobre a mesma pilha de dados sem rótulo: clustering pergunta quem se
parece com quem; regras de associação perguntam o que costuma aparecer
junto; redução de dimensionalidade pergunta quais direções carregam a
informação de verdade, e quais são só redundância.

Clustering hierárquico aglomerativo funciona como a fusão à la Dragon
Ball: cada ponto começa como seu próprio guerreiro isolado, o algoritmo
acha os dois mais parecidos do momento inteiro e funde os dois num
guerreiro-fusão só, repetindo esse processo, sempre fundindo a dupla
mais parecida que sobrou, até restar um guerreiro-fusão gigante com todo
mundo dentro. No exemplo de brinquedo do repositório, 6 guerreiros de
espécies diferentes (saiyajins, namekuseijins, humanos) se fundem em
ordem, e cortar essa árvore de fusões (o dendrograma) em 3 clusters
recupera exatamente as 3 espécies que ninguém disse ao algoritmo que
existiam; cortar em 2 já funde duas espécies num time só. A peça central
do algoritmo é o linkage, a regra que decide a distância entre dois
clusters (não entre dois pontos): ligação simples usa a menor distância
entre qualquer par de pontos dos dois grupos e tende a formar clusters
compridos e encadeados; ligação completa usa a maior distância e tende a
clusters compactos; ligação média fica no meio do caminho; o linkage de
Ward, o padrão do scikit-learn, escolhe a cada passo a fusão que menos
aumenta a variância interna dos clusters, também tendendo a grupos
compactos e de tamanho parecido. Não existe "o melhor linkage" fixo,
cada um carrega seu próprio viés, e isso aparece na prática: numa
amostra de 1.000 linhas do dataset de fraude (492 fraudes e 508 normais,
porque o custo O(n²) do algoritmo inviabiliza rodar nas 285 mil
transações inteiras), o linkage average teve a MAIOR silhueta (0,7442, o
cluster com aparência geométrica mais limpa) mas ARI de praticamente
zero (0,0001, nenhuma relação com fraude de verdade), enquanto Ward teve
silhueta menor (0,6322) mas foi o único a separar uma fatia razoável de
fraude num cluster à parte (106 das 492 fraudes da amostra caíram
sozinhas num cluster de 106 linhas). É a prova concreta de que uma
métrica interna, que não usa rótulo nenhum, não garante nada sobre bater
com uma estrutura externa que por acaso você já conhece: as duas medem
coisas diferentes.

O k-means resolve um problema parecido de um jeito bem mais barato
computacionalmente, com a analogia de uma guerra de território: duas
facções brigando por um mapa de postos avançados, cada uma com um
quartel-general (HQ); cada posto se filia ao HQ mais perto, e depois
cada HQ se muda pro centro de gravidade do próprio território, o que
pode fazer alguns postos trocarem de facção de novo, repetindo essa
dança de "escolher facção" e "mudar de sede" até ninguém mais trocar de
lado. Essas duas etapas, atribuição (cada ponto escolhe o centroide mais
perto) e atualização (cada centroide vira a média de quem escolheu ele),
são o algoritmo de Lloyd inteiro, e ele garante que a inércia (a soma
das distâncias ao quadrado de cada ponto até seu centroide, a régua de
bagunça do k-means) nunca aumenta de uma rodada pra outra, o que garante
convergência, mas só pra um mínimo LOCAL, não necessariamente o melhor
agrupamento possível. Centroides iniciais mal posicionados podem prender
o algoritmo num resultado ruim, e é por isso que o scikit-learn usa, por
padrão, k-means++ (espalha os centroides iniciais pelo mapa em vez de
deixá-los colados, sorteando cada novo centroide com probabilidade
proporcional à distância do centroide mais próximo já escolhido)
combinado com `n_init` (roda várias vezes e fica só com a de menor
inércia final). Diferente do clustering hierárquico, o k-means exige que
K seja escolhido de antemão, e uma forma comum de escolher é o método do
cotovelo: plotar a inércia final pra vários valores de K e procurar o
ponto onde aumentar K deixa de reduzir a inércia de forma significativa.
No dataset de fraude, com K=2 e k-means++ (rodando no conjunto de treino
inteiro, porque o custo O(n·K·iterações) do k-means escala bem melhor
que o do clustering hierárquico), a curva do cotovelo desceu suave, sem
quebra nítida nenhuma, e o resultado confirmou isso: ARI praticamente
zero (-0,0000) e silhueta baixa (0,0662), as 394 fraudes do treino
espalhadas quase 50/50 entre os dois clusters. Não é bug, é honesto:
clustering não supervisionado agrupa pelo que domina a VARIÂNCIA dos
dados, e nada garante que "ser fraude" seja o eixo de maior variância,
ao contrário de k-NN ou árvore de decisão, que aprendem a fronteira
certa porque VEEM o rótulo durante o treino.

Regras de associação respondem uma pergunta diferente: não "quem se
parece", mas "o que aparece junto". O Apriori funciona como um grupo de
jogadores mostrando os decks que montaram pra um card game: cada deck é
uma cesta de itens (cada carta um item), e o algoritmo quer achar
combinações que aparecem juntas com frequência maior do que seria só
coincidência. O suporte de um itemset é a fração das transações que
contêm todos os seus itens; a confiança de uma regra `A -> B` é
`suporte(A e B) / suporte(A)`, a fração das vezes que, tendo A, também
apareceu B; o lift é `confiança(A -> B) / suporte(B)`, e mede se essa
associação é de verdade ou só reflexo de B já ser popular sozinho (lift
= 1 é independência, lift > 1 é associação real, lift < 1 é associação
negativa). O algoritmo nunca testa todas as combinações possíveis de uma
vez, o que explodiria (`2^n - 1` subconjuntos possíveis pra n itens); em
vez disso cresce nível por nível, um item a mais por rodada, jogando
fora cedo o que já dá pra saber que não vai servir, graças à propriedade
Apriori: se um itemset é frequente, TODOS os seus subconjuntos também
são, então se um subconjunto já falhou, nenhum itemset que o contenha
precisa nem ser contado. No exemplo de brinquedo com 10 decks e 5 cartas
(Dragão, Espada, Escudo, Poção, Grimório), Grimório é podado logo no
nível 1 por suporte baixo (0,1); no nível 2, dos 6 pares possíveis entre
as 4 cartas restantes, dois caem por suporte baixo na contagem real; no
nível 3, a poda por subconjunto elimina 3 das 4 trincas candidatas sem
nem contar nos decks de novo, e a única sobrevivente ainda cai no
suporte mínimo, encerrando o algoritmo ali. A pegadinha clássica de
prova mora nas regras finais: "Escudo -> Dragão" tem confiança de 0,667
(parece uma regra boa), mas lift de 0,952, abaixo de 1, porque Dragão já
é popular demais sozinho (suporte 0,7) pra aquilo ser uma associação de
verdade. No dataset de fraude, discretizando o valor da compra, o
período do dia e os atributos V mais correlacionados com fraude (V17,
V14, V12) em faixas, a confiança de qualquer regra que aponte pra fraude
é baixíssima em valor absoluto (a maior é 1,47%, porque fraude é só
0,17% das transações), mas o lift conta a história de verdade: as três
faixas baixas de V17, V14 e V12 sozinhas multiplicam a chance de fraude
por 1,6 a 1,9 vezes, e juntar as três num único itemset multiplica por
8,5 vezes, a mesma lógica de combinar sinais fracos que aparece no
boosting e no random forest.

Redução de dimensionalidade responde a terceira pergunta: quais
direções carregam informação de verdade. O PCA funciona como o modo
foto de um jogo 3D: pra tirar aquela screenshot perfeita de um chefe
gigante, você gira a câmera até achar o ângulo que mostra o contorno
inteiro mais espalhado na tela, sem partes escondidas atrás de outras; o
PCA faz a mesma coisa girando os EIXOS ao redor da nuvem de pontos do
dataset, procurando a combinação de atributos (um componente principal)
que deixa os pontos mais espalhados possível quando projetados, ou seja,
a direção de maior variância. Calcular isso passa por centralizar os
dados (subtrair a média de cada coluna), montar a matriz de covariância,
decompor essa matriz em autovalores e autovetores (`Cov · v = λ · v`,
autovetor é a direção do componente, autovalor é quanta variância ele
carrega) e ordenar do maior autovalor pro menor: PC1 é o de maior
autovalor, PC2 o segundo, e assim por diante, sempre ortogonais entre
si. No exemplo de brinquedo com 8 personagens de RPG (Força e
Resistência, dois atributos propositalmente correlacionados), PC1
sozinho já explica 97,4% da variância, e a ordem dos personagens nessa
única coordenada já separa quem é "tanque" de quem não é. Duas
pegadinhas clássicas de prova: PCA não é seleção de atributos, ele cria
eixos NOVOS que são combinação de todos os originais, custando
interpretabilidade; e PCA não enxerga rótulo nenhum, maximiza a
variância dos dados sem saber se aquela é a direção que separa as
classes (quem faz isso, usando o rótulo, é a LDA, algoritmo
supervisionado diferente, com no máximo `n_classes - 1` componentes,
contra o `min(n_amostras - 1, n_atributos)` do PCA). No dataset de
fraude, a curva de variância explicada fica quase reta, sem cotovelo
nítido: são precisos 27 dos 30 componentes pra reter 95% da variância,
porque as colunas V1 a V28 já são o resultado de um PCA que o próprio
Kaggle aplicou antes de publicar os dados (anonimização), então já
chegam razoavelmente decorrelacionadas entre si, o mesmo motivo, aliás,
que fez `max_features` não ajudar o random forest nesse dataset: sem uma
direção de redundância clara sobrando, não tem o que decorrelacionar.

Fechando os dois temas juntos: viés indutivo e No Free Lunch não são
exclusividade do aprendizado supervisionado, atravessam o não
supervisionado inteiro também, só que disfarçados de escolha de
distância, de linkage, de K, de suporte mínimo ou de quantos componentes
manter. Nenhum desses algoritmos descobre uma verdade neutra escondida
nos dados; cada um enxerga só o tipo de estrutura que seu próprio viés
permite enxergar, e é por isso que o k-means não achou fraude nenhuma
(ela não é o eixo de maior variância) enquanto o Apriori achou um sinal
claro (V17, V14 e V12 baixos juntos), exatamente os mesmos três
atributos que o random forest, um algoritmo supervisionado completamente
diferente, também apontou como mais importantes: dois vieses indutivos
bem diferentes, um sem rótulo e outro com, concordando sobre onde mora o
sinal de verdade nesse dataset. Se um único algoritmo, com um único
viés, resolvesse todo problema de aprendizado, supervisionado ou não,
não precisaríamos de nenhum dos outros: é o mesmo teorema, começo e fim.
