# PCA (Análise de Componentes Principais)

Redução de dimensionalidade: pegar um dataset com muitos atributos e
resumir ele em poucos, perdendo o mínimo de informação possível. Diferente
dos algoritmos supervisionados, o PCA nunca olha pro rótulo (não existe
rótulo aqui, é técnica não supervisionada); ele só olha pra como os
próprios atributos variam e se relacionam entre si.

## Analogia central: o modo foto de um jogo 3D

Imagina que você quer tirar aquela screenshot perfeita de um chefe gigante
num jogo 3D, o tipo de imagem que vai pra capa do álbum de conquistas. Se
você posicionar a câmera de qualquer jeito, metade do bicho fica cortada
ou escondida atrás de uma perna. Você gira a câmera em volta dele até
achar o ângulo que mostra o contorno inteiro, o mais "espalhado" possível
na tela, sem partes se sobrepondo por trás de outras.

É exatamente isso que o PCA faz, só que em vez de girar uma câmera ao
redor de um monstro 3D, ele gira os EIXOS ao redor de uma nuvem de pontos
(o dataset, com um atributo por eixo). O "ângulo" que ele procura é a
combinação de atributos que faz os pontos ficarem mais espalhados possível
quando projetados numa reta ou num plano, ou seja, a direção de maior
variância. Assim como a screenshot 2D nunca captura 100% da informação de
um objeto 3D (perde profundidade), o PCA também perde informação ao
achatar muitos atributos em poucos, a ideia é perder o mínimo possível,
igual escolher o ângulo de câmera que menos esconde o bicho.

## Vocabulário básico

- Componente principal: cada um dos novos eixos que o PCA cria. São
  combinações lineares dos atributos originais (uma "receita" que mistura
  um pouco de cada atributo), não um atributo original isolado.
- PC1, PC2, ...: os componentes, sempre ordenados do que captura MAIS
  variância pro que captura MENOS.
- Autovalor (eigenvalue): quanta variância aquele componente captura.
  Quanto maior o autovalor, mais "informativo" (no sentido de espalhar os
  dados) é aquele eixo.
- Autovetor (eigenvector): a direção do componente no espaço dos
  atributos originais. É o "pra onde a câmera aponta".
- Loading (peso): o valor de cada atributo original dentro da receita de
  um componente. Loading alto (em módulo) num atributo significa que ele
  pesa bastante naquele componente.
- Variância explicada: a fração da variância total que um componente
  captura (autovalor dele dividido pela soma de todos os autovalores).

## Como o PCA é calculado, passo a passo

1. Centralizar os dados: subtrair de cada coluna a própria média, pra
   trabalhar em cima do desvio de cada ponto em relação ao "ponto médio",
   não do valor bruto.
2. Calcular a matriz de covariância dos atributos centralizados: `Cov =
   XᵀX / (n - 1)`. Na diagonal fica a variância de cada atributo sozinho;
   fora da diagonal fica a covariância entre pares de atributos (o quanto
   eles crescem ou caem juntos).
3. Decompor essa matriz em autovalores e autovetores: `Cov · v = λ · v`.
   Cada par (autovalor, autovetor) é um componente candidato.
4. Ordenar os componentes do maior autovalor pro menor. O primeiro
   (maior autovalor) é PC1, o segundo é PC2, e assim por diante.
5. Escolher quantos componentes manter (ver seção de variância explicada
   abaixo) e projetar os dados centralizados nesses componentes:
   `Y = X_centralizado · V_k`, onde `V_k` são os `k` autovetores
   escolhidos, um em cada coluna.

Ver esse processo acontecendo de verdade, com um exemplo de 8 personagens
de RPG (Força e Resistência, dois atributos bem correlacionados que o PCA
resume num só), em `pca.py`.

Detalhe de implementação: na prática, bibliotecas como o scikit-learn não
decompõem a matriz de covariância diretamente, elas usam SVD
(decomposição em valores singulares) de `X` centralizado: `X = U · S ·
Vᵀ`. Os autovetores da covariância são as colunas de `V`, e os
autovalores são `S² / (n - 1)`. É matematicamente equivalente ao que está
descrito acima, só numericamente mais estável (evita montar a matriz de
covariância inteira, o que acumula mais erro de arredondamento).

## Quantos componentes manter

Ideia central: cada componente a mais que você mantém preserva mais
informação, mas também mantém mais dimensão (o que você queria reduzir em
primeiro lugar). O equilíbrio entre os dois é medido pela variância
explicada.

- Variância explicada acumulada: soma a fração de variância dos `k`
  primeiros componentes. É comum escolher `k` de forma que essa soma
  passe de um limiar, tipo 90% ou 95% da variância total.
- Regra do cotovelo (scree plot): plota a variância explicada de cada
  componente em ordem e procura o ponto onde a curva "quebra" e vira
  quase reta (os componentes seguintes contribuem cada vez menos, quase
  achatado). Esse cotovelo costuma ser um bom candidato pra `k`.
- Pegadinha de prova: o número máximo de componentes que existem é
  `min(n_amostras - 1, n_atributos)`. Não dá pra "inventar" mais variância
  do que a que já existe nos dados.

## Padronização antes do PCA

Como o PCA maximiza variância, um atributo medido numa escala numérica
maior (tipo salário em milhares de reais, contra idade em anos) domina a
conta sozinho, mesmo que não seja de fato mais "importante", só porque os
números dele são maiores. Por isso é praticamente obrigatório padronizar
os atributos (média 0, desvio-padrão 1, o `StandardScaler` de
`utils/data_utils.py`) antes de rodar o PCA, a não ser que todos os
atributos já estejam na mesma unidade e escala por natureza.

## PCA não é seleção de atributos, e não enxerga rótulo

Duas pegadinhas de prova comuns:

- PCA não descarta atributos originais, ele cria NOVOS eixos que são
  combinação de todos eles. Depois do PCA você não tem mais "Idade" ou
  "Salário" isolados, tem "PC1", que é uma mistura dos dois (e de todo o
  resto). Isso custa interpretabilidade: fica mais difícil explicar pra
  alguém o que exatamente um componente principal "significa" no mundo
  real, ao contrário de um atributo original.
- PCA é não supervisionado: ele maximiza a variância dos atributos, sem
  saber nem se importar com o rótulo (classe) de cada exemplo. Pode
  acontecer de a direção de maior variância NÃO ser a direção que melhor
  separa as classes, nesse caso reduzir dimensionalidade com PCA antes de
  treinar um classificador pode até piorar o resultado. Quem maximiza a
  separação ENTRE classes, usando o rótulo, é a LDA (Linear Discriminant
  Analysis), um algoritmo supervisionado diferente que parece com PCA na
  matemática (também usa autovalores e autovetores) mas resolve um
  problema diferente.

| | PCA | LDA |
|---|---|---|
| Tipo | Não supervisionado | Supervisionado |
| Objetivo | Maximizar variância dos dados | Maximizar separação entre classes |
| Usa o rótulo? | Não | Sim |
| Número máximo de componentes | `min(n_amostras - 1, n_atributos)` | `n_classes - 1` |

## Os componentes principais são sempre ortogonais

Autovetores de uma matriz de covariância (que é sempre simétrica) são
ortogonais entre si, formam ângulo reto uns com os outros no espaço dos
atributos. Na prática isso quer dizer que os componentes principais nunca
são correlacionados linearmente entre si: PC1 e PC2 medem coisas
"independentes" nesse sentido, mesmo que os atributos originais fossem
todos correlacionados uns com os outros.

## Aspectos Positivos

- Reduz dimensionalidade preservando o máximo de variância possível pra
  aquele número de componentes escolhido.
- Remove (ou reduz bastante) a colinearidade entre atributos: como os
  componentes são ortogonais, alimentar um modelo com componentes
  principais em vez dos atributos originais evita o problema de
  atributos redundantes competindo entre si.
- Ajuda a visualizar dados de muitas dimensões, projetando em 2D ou 3D
  pra enxergar padrões e agrupamentos a olho.
- Acelera o treino de outros algoritmos (menos atributos = menos conta),
  e componentes de variância muito baixa costumam ser ruído, então
  descartá-los às vezes até melhora a generalização.

## Aspectos Negativos

- Perde interpretabilidade: os componentes são misturas lineares dos
  atributos originais, não têm um significado direto e fácil de explicar.
- Só captura relações LINEARES entre atributos. Se a estrutura de
  verdade dos dados for curva ou torta, o PCA não enxerga isso (existem
  variantes não lineares, tipo Kernel PCA, t-SNE ou UMAP, pra esses
  casos, fora do escopo aqui).
- Sensível à escala dos atributos: sem padronizar antes, o resultado
  fica dominado pelos atributos de maior variância bruta, não
  necessariamente os mais relevantes.
- Não usa o rótulo: pode descartar exatamente a direção que seria útil
  pra separar as classes num problema de classificação (ver comparação
  com LDA acima).

## Ver também

- `pca/pca.py`: contas feitas na mão (centralização, matriz de
  covariância, autovalores e autovetores, variância explicada) com o
  exemplo dos 8 personagens de RPG, e a aplicação de verdade no dataset
  de fraude, incluindo a curiosidade de que boa parte dos atributos desse
  dataset (V1 a V28) já vieram anonimizados via PCA pelo próprio Kaggle.
