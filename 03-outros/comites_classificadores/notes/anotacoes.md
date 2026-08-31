# Comitês de Classificadores (Modelos Múltiplos / Ensembles)

- Comitê de classificadores (ensemble, ou "modelos múltiplos preditivos") é o nome geral pra qualquer estratégia que combina VÁRIOS modelos preditivos numa única decisão, em vez de confiar num modelo só. Bagging, boosting e random forest (`01-supervisionado/bagging`, `01-supervisionado/boosting`, `01-supervisionado/random_forest`) são os três exemplos mais cobrados de prova, mas são só uma fatia de um assunto bem mais amplo: como combinar, quando combinar e por que combinar.
- Esta nota cobre a teoria geral do capítulo de comitês (o guarda-chuva que fica por cima do bagging/boosting/random forest), não é ela mesma um algoritmo específico pra rodar, por isso mora em `03-outros/` e não em `01-supervisionado/`.

## Por que combinar modelos: o teorema do "no free lunch"

O teorema do "no free lunch" (Wolpert e Macready) diz, em resumo: não existe um algoritmo de aprendizado que seja o melhor pra TODOS os problemas possíveis. Em média, calculada sobre todos os problemas possíveis (inclusive os mais estranhos e artificiais, sem nenhum padrão real por trás), todo algoritmo tem o mesmo desempenho.

Isso soa desanimador, mas a lição prática é outra: como ninguém sabe de antemão, com certeza absoluta, qual algoritmo (qual hipótese, dentro de qual espaço de hipóteses) é o certo pro problema específico que você tem na mão, apostar tudo num único modelo é uma aposta arriscada. Só dá pra saber qual algoritmo funciona melhor testando empiricamente no problema real, não por dedução teórica pura. Comitês de classificadores nascem dessa incerteza: em vez de apostar tudo numa hipótese só, combinar várias hipóteses diferentes reduz o risco de ter escolhido a errada.

## O requisito de diversidade: por que os membros precisam discordar

Combinar modelos só ajuda se os modelos combinados tiverem um nível substancial de DESACORDO entre si. Se todo membro do comitê comete exatamente os mesmos erros, nos mesmos exemplos, juntar as opiniões deles não muda nada: o comitê inteiro erra junto, do mesmo jeito que um membro sozinho erraria. A matemática por trás disso já aparece explicitamente na nota do bagging (ver `01-supervisionado/bagging/notes/anotacoes.md`, seção "Por que bagging sozinho não decorrelaciona o suficiente", repetida aqui em resumo):

```
Var(média de B modelos) = ρσ² + (1 - ρ)σ²/B
```

Onde `ρ` é a correlação par a par entre os modelos. Quando `ρ` é alto (modelos concordam demais, pouca diversidade), o primeiro termo trava a variância num piso alto, e somar mais modelos ao comitê (aumentar `B`) quase não ajuda. Diversidade baixa = comitê caro (mais modelos pra treinar e manter) sem ganho real. Toda a engenharia por trás de bagging (amostras bootstrap diferentes), random forest (mais o sorteio de atributos por divisão) e boosting (cada estimador focado especificamente no que os anteriores erraram) existe justamente pra GARANTIR esse desacordo de propósito, em vez de torcer pra ele acontecer sozinho.

## Votação: por que não escolher só uma hipótese

Um jeito de enxergar filosoficamente por que combinar ajuda: imagine o espaço de hipóteses inteiro (todas as regras que um algoritmo poderia ter aprendido com aquele treino). Normalmente um algoritmo de aprendizado escolhe UMA hipótese só (a que minimiza o erro no treino, por exemplo) e descarta todas as outras hipóteses que também eram razoavelmente consistentes com os dados. Só que, com uma quantidade finita de exemplos de treino, várias hipóteses diferentes podem explicar o treino igualmente bem e discordar bastante em exemplos novos, nunca vistos: escolher só uma é jogar fora informação sobre a incerteza real do problema.

O método de votação ataca isso de frente: em vez de escolher uma hipótese só, usa VÁRIAS hipóteses aceitáveis do espaço de hipóteses ao mesmo tempo, e deixa elas "votarem" na previsão final. É parecido, em espírito, com a ideia de combinação Bayesiana de modelos (Bayesian model averaging): em vez de apostar tudo na hipótese mais provável, pondera a opinião de várias hipóteses plausíveis.

## Votação uniforme x votação com peso

- Votação uniforme: todo membro do comitê vale o mesmo voto, não importa quão bom ou ruim ele é individualmente. É o que o bagging faz (ver `01-supervisionado/bagging/notes/anotacoes.md`): cada investigador vota, ganha a maioria simples.
- Votação com peso: membros melhores pesam mais na decisão final. É o que o boosting (AdaBoost) faz com o `α_t` (ver `01-supervisionado/boosting/notes/anotacoes.md`): o peso de voto de cada estimador é calculado a partir do próprio desempenho ponderado dele durante o treino, então um especialista que acertou mais tem mais influência no veredito final do que um que mal passou de chute aleatório.

## Regras clássicas de combinação de saídas

Voto majoritário (contar quantos membros escolheram cada classe) é só uma das formas de combinar. Quando os classificadores do comitê devolvem SCORES ou PROBABILIDADES por classe (não só um rótulo seco), existe um leque de regras clássicas de combinação, cada uma com uma personalidade diferente sobre como tratar discordância:

| Regra | Como calcula | Personalidade |
|---|---|---|
| Soma | Soma o score de cada classe entre todos os classificadores, escolhe a maior soma | Equilibrada: um classificador confiante pode compensar vários incertos |
| Média | Média dos scores em vez da soma (dá no mesmo ranking se todo classificador pesa igual) | Igual à soma, só numa escala diferente (0 a 1 em vez de 0 a N) |
| Média geométrica | Multiplica os scores e tira a raiz N-ésima | Mais dura com discordância: um score bem baixo de um só classificador já puxa a média geométrica pra baixo |
| Produto | Multiplica os scores direto, sem tirar raiz | Ainda mais dura: um único classificador atribuindo probabilidade perto de zero pra uma classe já quase elimina ela, mesmo que os outros a favoreçam |
| Máximo | Olha o maior score único, em qualquer classificador, pra qualquer classe | Otimista: confia na opinião mais confiante que existir em qualquer lugar do comitê |
| Mínimo | Pra cada classe, pega o PIOR (menor) score que algum classificador deu, escolhe a classe com o maior "pior caso" | Pessimista/conservadora: só favorece uma classe se NINGUÉM do comitê discordar muito dela |
| Seriação (ranking) | Cada classificador RANQUEIA as classes (1º lugar, 2º lugar...) em vez de dar um score bruto; os rankings são somados (tipo contagem de Borda) | Útil quando os scores de classificadores diferentes não são comparáveis na mesma escala (ex.: comitê heterogêneo, cada membro calibrado de um jeito); usa só a ORDEM de preferência, não a magnitude |

A escolha entre essas regras importa mais quando o comitê é heterogêneo (classificadores de tipos diferentes, com escalas de score diferentes) ou quando se quer controlar explicitamente o quão "conservador" (regra do mínimo) ou "arriscado" (regra do máximo) o comitê deve ser diante de discordância.

## Seleção estática x seleção dinâmica de classificadores

- Seleção/combinação ESTÁTICA: a forma de combinar (ou os pesos de cada membro) é decidida UMA VEZ, durante o treino, e aplicada do mesmo jeito pra qualquer exemplo novo, sem olhar as particularidades daquele exemplo específico. Bagging (voto uniforme fixo) e boosting (pesos `α_t` fixos após o treino) são os dois exemplos já vistos, ambos estáticos.
- Seleção/combinação DINÂMICA: a decisão de QUAL membro confiar (ou quanto pesar cada um) é tomada NA HORA da previsão, olhando pra região do espaço de atributos onde aquele exemplo específico cai. A ideia central é que um classificador pode ser ótimo numa região dos dados e ruim em outra, então a "melhor combinação" pode mudar de exemplo pra exemplo. Um exemplo clássico dessa família é a seleção dinâmica por acurácia local (DCS-LA): pra um exemplo novo, olha o desempenho de cada classificador só nos vizinhos mais próximos DAQUELE exemplo específico no espaço de treino, e confia mais em quem foi bem ali perto.

Sobre os métodos específicos citados (MAI, SCANN): SCANN é uma técnica de combinação que usa análise de correspondência (uma técnica estatística de redução de dimensionalidade) pra combinar as saídas de classificadores heterogêneos, olhando o espaço reduzido resultante com um classificador de vizinho mais próximo (daí o nome, Stacked Correspondence Analysis and Nearest Neighbor). Não estou com confiança total na definição exata de "MAI" nesse contexto específico do capítulo, então vale conferir a definição direto na fonte (Faceli et al.) antes de levar pra prova; se você colar o trecho do livro aqui, eu incorporo a definição certinha nesta nota.

## Combinação de classificadores homogêneos: bagging

Quando todos os membros do comitê são do MESMO tipo de algoritmo (ex.: um monte de árvores de decisão), a forma clássica de gerar diversidade entre eles é o bagging: treinar cada membro numa amostra bootstrap diferente do mesmo dataset. Teoria completa, incluindo a matemática de por que isso reduz variância, em `01-supervisionado/bagging/notes/anotacoes.md`; random forest (`01-supervisionado/random_forest/notes/anotacoes.md`) é a evolução direta disso, somando o sorteio de atributos por divisão pra decorrelacionar ainda mais.

## Combinação de classificadores heterogêneos: stacking e cascading

Quando os membros do comitê são de tipos DIFERENTES (ex.: uma árvore de decisão, um SVM e um k-NN no mesmo comitê), bootstrap sozinho não faz muito sentido como fonte de diversidade (os modelos já são estruturalmente diferentes entre si, isso já basta pra gerar discordância). Duas estratégias clássicas de combinar esse tipo de comitê:

### Generalização em pilha (stacking)

Treina vários classificadores BASE (de tipos diferentes), cada um aprendendo a prever `y` a partir de `x`, do jeito normal. Depois, treina um classificador NOVO, chamado meta-classificador (ou classificador de nível 1), cujo `x` de entrada não são mais os atributos originais, e sim as PREVISÕES (ou probabilidades) dos classificadores base, e cujo alvo continua sendo o `y` original. O meta-classificador aprende, a partir dos dados, a melhor forma de combinar as opiniões dos classificadores base, em vez de usar uma regra fixa decidida à mão (tipo voto majoritário ou média).

Pegadinha clássica de prova: se o meta-classificador for treinado nas previsões que os classificadores base deram SOBRE O PRÓPRIO CONJUNTO DE TREINO deles, essas previsões vão ser artificialmente boas demais (cada classificador base já "decorou" parte desses exemplos), e o meta-classificador aprende uma relação otimista demais que não se sustenta em dados novos. A correção padrão é gerar as previsões de treino do meta-classificador usando validação cruzada (previsões fora-da-dobra, out-of-fold): cada classificador base prevê só os exemplos que NÃO viu no próprio treino daquela dobra, replicando de forma mais honesta o que vai acontecer com dados de teste de verdade.

### Generalização em cascata (cascading)

Encadeia classificadores em ESTÁGIOS: o primeiro estágio (normalmente um classificador simples e barato) tenta decidir; se ele estiver confiante o suficiente, a decisão dele é usada e o processo para ali, sem gastar mais computação. Só quando o estágio atual não está confiante o suficiente (a previsão fica ambígua, perto do limiar de decisão) é que o exemplo é passado adiante pro próximo estágio, geralmente mais complexo e mais caro computacionalmente.

A motivação central é diferente da do stacking (combinar opiniões) e também diferente da do boosting (corrigir erro): cascading existe principalmente por EFICIÊNCIA, gastar processamento caro só nos casos difíceis, resolvendo os casos fáceis rápido e barato logo no primeiro estágio. É a mesma lógica usada, por exemplo, em detectores de rosto em tempo real (cascata de Viola-Jones): a maioria das regiões de uma imagem não tem rosto nenhum, e um primeiro estágio bem barato já descarta a maior parte delas, deixando só as regiões ambíguas pra estágios mais caros analisarem.

## Metaaprendizado

"Metaaprendizado" (aprender sobre aprender) aparece em dois sentidos relacionados nesse contexto:

- No sentido mais estrito, é literalmente o que o meta-classificador do stacking faz: aprender, a partir dos dados, como combinar as saídas de outros modelos, em vez de usar uma regra de combinação fixada à mão.
- No sentido mais amplo (o campo de metaaprendizado em geral), é usar características do PROBLEMA (meta-atributos de um dataset, tipo número de exemplos, número de atributos, quão desbalanceadas as classes são) e o histórico de desempenho de vários algoritmos em vários datasets passados, pra RECOMENDAR de antemão qual algoritmo (ou qual configuração de hiperparâmetros) tende a funcionar melhor num dataset novo, sem precisar testar tudo do zero. É uma forma de atacar, na prática, o problema que o teorema do no-free-lunch levanta: já que nenhum algoritmo domina todos os problemas, metaaprendizado tenta aprender QUANDO cada algoritmo costuma ganhar.

## Sistemas híbridos

Sistemas híbridos combinam modelos de NATUREZAS fundamentalmente diferentes, não só instâncias diferentes de uma mesma família estatística. Exemplos clássicos: juntar um sistema simbólico baseado em regras (interpretável, mas rígido, só lida bem com o que foi explicitamente codificado) com um modelo conexionista/estatístico (flexível, aprende padrões direto dos dados, mas difícil de interpretar); ou combinar lógica fuzzy (boa pra representar graus de incerteza e conceitos vagos, tipo "quente" ou "rápido") com uma rede neural (boa pra aprender padrões complexos automaticamente). A aposta é juntar pontos fortes complementares que nenhum paradigma sozinho tem: interpretabilidade e regras de domínio de um lado, capacidade de aprender padrão complexo direto do dado do outro.

## Decomposição de problemas multiclasse

Uma forma diferente (mas ainda dentro do espírito de "comitê") de lidar com um problema de mais de duas classes: quebrar ele em VÁRIOS problemas BINÁRIOS menores, resolver cada um com um classificador próprio, e depois combinar as respostas. É outra situação em que as regras de combinação vistas antes (voto, soma, etc.) voltam a ser úteis, agora combinando classificadores binários em vez de classificadores multiclasse completos.

- **Um-contra-todos (One-vs-Rest, OvR)**: pra `k` classes, treina `k` classificadores binários, cada um respondendo "é a classe X, ou é qualquer uma das outras?". Na previsão, roda os `k` classificadores e escolhe a classe cujo classificador ficou mais confiante.
- **Um-contra-um (One-vs-One, OvO)**: treina um classificador binário pra CADA PAR de classes (`k*(k-1)/2` classificadores no total), cada um só decidindo entre aquelas duas classes específicas. Na previsão, roda todos os pares e combina por votação (cada "duelo" vota na classe vencedora daquele par, ganha a classe com mais vitórias no total).
- **Códigos corretores de erro (ECOC, error-correcting output codes)**: dá pra cada classe um código binário único (uma sequência de 0s e 1s, tipo um "código de barras" da classe), treina um classificador binário pra cada POSIÇÃO desse código (cada um aprende a prever um bit específico do código), e na previsão compara o padrão de bits previstos com o código de cada classe, escolhendo a classe cujo código fica mais perto (menor distância de Hamming) do padrão previsto. A redundância dos códigos (mais bits do que o estritamente necessário pra distinguir as classes) dá certa tolerância a erro: mesmo que um ou dois classificadores binários errem o próprio bit, o código previsto pode continuar mais perto do código certo do que de qualquer outro, e a classe certa ainda vence.

## Ver também

- `01-supervisionado/bagging/notes/anotacoes.md`: bagging, o exemplo mais desenvolvido de combinação homogênea com votação uniforme.
- `01-supervisionado/boosting/notes/anotacoes.md`: boosting, o exemplo mais desenvolvido de votação com peso e combinação sequencial.
- `01-supervisionado/random_forest/notes/anotacoes.md`: random forest, bagging com uma segunda camada de aleatoriedade pra aumentar a diversidade.
- `03-outros/fundamentos_aprendizado/notes/anotacoes.md`: espaço de hipóteses e viés indutivo, o vocabulário usado na seção de votação como "espaço de versões".
