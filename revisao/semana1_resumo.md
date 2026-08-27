# Resumo Semana 1: Aprendizado Supervisionado (clássico)

Cobre o que já foi estudado até agora: fundamentos do aprendizado
supervisionado e os quatro algoritmos de `01-supervisionado/`. Não cobre
"formas de avaliação" (ementa de Terça 01/09, Semana 2, ainda sem notas
escritas) nem aprendizado não supervisionado.

## Fundamentos (`03-outros/fundamentos_aprendizado/notes/anotacoes.md`)

Aprender é escolher uma hipótese `h`, de dentro de um espaço de hipóteses
`H`, a partir de um conjunto de treino `D`, tentando se aproximar do
conceito-alvo `c` (a regra verdadeira, desconhecida) o bastante pra
generalizar bem, não só decorar `D`. Esse processo de generalizar do
particular (exemplos vistos) pro geral (uma regra que vale pra qualquer
exemplo futuro) se chama indução.

Duas medidas de erro que não podem se confundir: erro empírico (só nos
exemplos de treino, fácil de medir mas engana) e erro de generalização
(sobre TODOS os exemplos possíveis, o que realmente importa).

Isso só funciona porque todo algoritmo carrega um viés indutivo, uma
suposição extra além dos dados observados, sem a qual não tem como
escolher entre as infinitas hipóteses consistentes com o treino:

- Viés de restrição (limita `H` de antemão): SVM com kernel linear só
  considera hiperplanos; Naive Bayes só considera classificadores que
  assumem independência condicional entre atributos.
- Viés de preferência (a busca prefere certas hipóteses): a árvore de
  decisão gulosa empurra a busca em direção a árvores mais simples
  (versão prática da navalha de Occam).

No Free Lunch Theorem: em média, sobre todos os problemas possíveis,
nenhum algoritmo é melhor que outro. Um algoritmo só se sai melhor nos
problemas onde o viés indutivo dele combina com a estrutura real daqueles
dados. É por isso que existem vários algoritmos diferentes.

PAC-learning (`03-outros/pac_learning/notes/anotacoes.md`) formaliza isso
com números: quantos exemplos `m` garantem, com probabilidade `1-δ`, erro
`≤ ε`. No caso realizável, `m ≥ (1/ε)(ln|H| + ln(1/δ))`; espaço de
hipóteses maior (`H` mais rico, ou dimensão VC maior) exige mais
exemplos.

## Árvore de decisão

Indução top-down gulosa: em cada nó, escolhe o atributo que mais reduz a
impureza do grupo (entropia ou índice Gini), e repete recursivamente
dentro de cada subgrupo. CART (usado pelo scikit-learn) sempre faz
perguntas binárias e usa Gini; ID3 e C4.5 usam entropia/ganho de
informação e permitem divisão multiway.

- Ganho de informação = quanto a entropia cai depois de uma pergunta.
- Sem limite, decora o treino inteiro (overfitting). Pré-poda
  (`max_depth`, `min_samples_leaf`) freia o crescimento; pós-poda
  (cost-complexity, `ccp_alpha`) deixa crescer e depois corta.
- Viés indutivo: preferência por árvores simples.
- (+) interpretável, não precisa normalizar, lida com número e categoria
  juntos. (-) instável (muda muito com pouca mudança no treino), gulosa
  (não garante árvore ótima), tende à classe majoritária em dados
  desbalanceados.
- Resultado no dataset de fraude: `criterion='entropy'`, sem poda,
  precisão 0,7500, recall 0,8265, **F1 = 0,7864** (esse número ainda
  precisa ser copiado pra tabela de `01-supervisionado/README.md`).

## k-NN

Lazy learning (aprendizado preguiçoso): não constrói modelo nenhum
durante o treino, só memoriza os exemplos. Toda a conta pesada acontece
na hora de prever: calcula distância até todo mundo, ordena, pega os K
mais próximos, vota (ou tira a média, em regressão).

- Exige normalização: a distância soma diferenças de todos os atributos,
  e um atributo em escala maior domina a conta sozinho se não estiver na
  mesma escala dos outros.
- K pequeno: fronteira recortada, sensível a ruído, overfitting. K
  grande: fronteira suave demais, underfitting, classe rara perde
  votação. Voto ponderado por distância (peso = 1/distância) ajuda a
  resolver empates sem precisar diminuir K.
- Maldição da dimensionalidade: com muitos atributos, a distância ao
  vizinho mais próximo se aproxima da distância ao mais distante (todo
  mundo fica "igualmente longe"), e o conceito de vizinhança perde
  sentido. Correção: redução de dimensionalidade (PCA) ou seleção de
  atributos.
- Viés indutivo: exemplos parecidos (próximos no espaço de atributos)
  tendem a ter o mesmo rótulo.
- Resultado no dataset de fraude: K=3, voto simples, precisão 0,9101,
  recall 0,8265, **F1 = 0,8663**, o melhor resultado desta família até
  agora.

## SVM

Maximiza a margem de separação entre as classes: entre todos os
hiperplanos que separam corretamente, escolhe o que fica mais longe dos
exemplos mais próximos de cada classe. Só os vetores de suporte (os
exemplos exatamente em cima da margem) decidem onde fica a fronteira;
mover qualquer outro exemplo não muda nada.

- Embasado na Teoria do Aprendizado Estatístico (Vapnik e Chervonenkis):
  margem larga corresponde a uma dimensão VC menor, que generaliza
  melhor (princípio da minimização do risco estrutural).
- Margem rígida exige separação perfeita (sem dado de treino entre as
  margens); como isso é raro em problemas reais, a margem suave introduz
  variáveis de folga (`ξ_i`) e o parâmetro `C`: `C` alto tenta acertar
  tudo (margem estreita, sensível a ruído), `C` baixo tolera erro em
  troca de margem mais larga.
- Truque do kernel: separa fronteiras não lineares calculando a
  similaridade entre pares de exemplos como se estivessem num espaço com
  mais dimensões (`K(x,x') = φ(x)·φ(x')`), sem nunca calcular `φ(x)`
  explicitamente. Kernels comuns: linear, polinomial, RBF.
- Problema de otimização convexo: único mínimo global, algoritmo
  determinístico (mesmo resultado, não importa a ordem dos dados de
  treino).
- Também serve pra regressão (SVR, com um "tubo" de tolerância em vez de
  duas classes) e pra detecção de anomalia/agrupamento (One-Class SVM).
- Requer atributos numéricos normalizados; computacionalmente custoso
  (O(n²) a O(n³)), mas robusto em alta dimensionalidade.
- Downsides: sensível à escolha de `C`, kernel e `γ`; difícil de
  interpretar.
- Resultado no dataset de fraude: `LinearSVC` no treino completo,
  precisão 0,8286, recall 0,5918, **F1 = 0,6905**.

## Naive Bayes

Teorema de Bayes (`P(classe|pista) = P(pista|classe)·P(classe)/P(pista)`)
combinado com a suposição "ingênua" de independência condicional entre
atributos, dado a classe. Critério MAP: calcula `P(classe) × produtório
das verossimilhanças` pra cada classe e escolhe a maior.

- Problema do produto de zero: uma única combinação atributo-classe nunca
  vista no treino zera o produtório inteiro. Corrigido pela suavização de
  Laplace (soma 1 em cada contagem antes de dividir).
- Variantes: GaussianNB (atributo contínuo, assume curva de sino dentro
  de cada classe), MultinomialNB (contagem, ex. palavras num texto),
  BernoulliNB (presença/ausência).
- Viés indutivo: independência condicional entre atributos dado a
  classe (raramente verdadeira, mas a classificação final costuma
  continuar certa mesmo com a suposição errada, porque só importa qual
  classe vence a comparação).
- (+) rápido de treinar, funciona com pouco dado, boa baseline. (-)
  probabilidades mal calibradas quando a suposição de independência é
  muito violada; GaussianNB sofre quando os dados não seguem distribuição
  normal.
- Resultado no dataset de fraude: `GaussianNB`, recall alto (0,8469) mas
  precisão baixíssima (0,0588), **F1 = 0,1099**: a classe fraude não
  segue uma curva gaussiana limpa, então o modelo classifica fraude
  demais como positivo.

## Tabela comparativa (classe fraude, dataset de detecção de fraude)

| Algoritmo | Precisão | Recall | F1 |
|---|---|---|---|
| k-NN (K=3) | 0,9101 | 0,8265 | **0,8663** |
| Árvore de decisão (entropy) | 0,7500 | 0,8265 | 0,7864 |
| SVM (LinearSVC) | 0,8286 | 0,5918 | 0,6905 |
| Naive Bayes (GaussianNB) | 0,0588 | 0,8469 | 0,1099 |

Fio condutor pra prova: dá pra explicar cada posição dessa fila pelo
viés indutivo de cada algoritmo batendo (ou não) com a forma real dos
dados de fraude. k-NN e árvore não fazem suposição forte sobre a forma
da fronteira, só se adaptam aos dados; SVM linear assume fronteira reta
(razoável, mas não ideal); Naive Bayes assume independência e distribuição
gaussiana, duas suposições que não combinam com esse dataset em
particular.

## O que ainda falta (não está neste resumo)

- Formas de avaliação (matriz de confusão, AUC-ROC, validação cruzada):
  ementa de Terça 01/09, sem notas escritas ainda.
- Ensemble (bagging, boosting, random forest): pastas criadas em
  `01-supervisionado/`, ainda sem código.
- Aprendizado não supervisionado inteiro (`02-nao_supervisionado/`).
