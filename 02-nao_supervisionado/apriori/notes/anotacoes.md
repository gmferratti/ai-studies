# Apriori (Regras de Associação)

- Aprendizado não supervisionado: não existe rótulo "certo" a prever, o
  algoritmo só procura padrões de coocorrência escondidos nos dados.
- Objetivo: descobrir regras do tipo "quem tem X também costuma ter Y",
  sem ninguém precisar dizer de antemão o que procurar.
- Aplicação clássica: análise de cesta de compras (market basket
  analysis), a origem da história "quem compra fralda também compra
  cerveja". Serve também pra recomendação de produtos, análise de
  combinações de sintomas em diagnóstico e, como no `apriori.py` deste
  repositório, achar combinações de faixas de valores associadas a fraude.

## Analogia central: montando decks de um card game

Pensa numa mesa de torneio com vários jogadores mostrando os decks que
montaram. Cada deck é uma "cesta de compras": um conjunto de cartas que
aquele jogador escolheu colocar junto. O Apriori olha pra centenas desses
decks e pergunta: quais cartas aparecem juntas com frequência maior do
que seria só coincidência? Se toda vez que alguém coloca a Carta A no
deck ele também coloca a Carta B, isso é uma combo de verdade, não
acaso, e vale a pena anotar como regra.

A mecânica de busca é a mesma do resto do algoritmo: começa contando
cada carta sozinha, descarta as raras demais pra importar, e só então
tenta juntar pares dessas cartas que sobraram, depois trincas, e assim
por diante, sempre reaproveitando o que já foi descoberto no passo
anterior. Ver esse processo acontecendo de verdade, nível por nível, com
o exemplo dos decks (Dragão, Espada, Escudo, Poção, Grimório), em
`apriori.py`.

## Vocabulário básico

- Item: um elemento individual (uma carta, um produto no carrinho).
- Transação: um conjunto de itens que apareceram juntos de uma vez (um
  deck, uma compra no mercado).
- Itemset: qualquer subconjunto de itens que se quer testar junto, do
  tamanho que for (`{Dragão}`, `{Dragão, Espada}`, `{Dragão, Espada,
  Escudo}`).
- Itemset frequente: um itemset cujo suporte (ver fórmulas abaixo) passa
  de um limiar mínimo escolhido de antemão (`min_suporte`).
- Regra de associação: uma seta `antecedente -> consequente` entre dois
  itemsets sem itens em comum (ex.: `{Dragão} -> {Espada}`), lida como
  "quando o antecedente aparece, o consequente costuma aparecer junto".
- Antecedente e consequente: o lado esquerdo e o lado direito da seta.
  Repara que o mesmo itemset frequente `{Dragão, Espada}` pode virar
  regra nos dois sentidos, `Dragão -> Espada` ou `Espada -> Dragão`, cada
  sentido com sua própria confiança.

## Como o algoritmo funciona: geração de candidatos e poda, nível por nível

Testar toda combinação possível de itens seria inviável: com `n` itens
distintos existem `2^n - 1` subconjuntos não vazios pra considerar, um
número que explode rapidinho (só 20 produtos diferentes já dão mais de 1
milhão de itemsets possíveis). O truque do Apriori é nunca testar tudo
de uma vez, e sim crescer devagar, um item por rodada, jogando fora cedo
o que já dá pra saber que não vai servir:

1. **Nível 1**: conta o suporte de cada item sozinho e descarta os que
   não batem o suporte mínimo. No exemplo dos decks, com suporte mínimo
   de 0,4: Dragão (0,7), Espada (0,7), Escudo (0,6) e Poção (0,6) passam;
   Grimório (0,1, aparece só num deck) é podado aqui mesmo.
2. **Geração de candidatos (passo de junção)**: junta os itens que
   sobreviveram no nível anterior em itemsets do próximo tamanho. Com os
   4 itens que sobraram, isso gera 6 pares candidatos.
3. **Poda pela propriedade Apriori**, antes de contar de novo: um
   itemset só pode ser frequente se TODOS os seus subconjuntos também
   forem (propriedade de anti-monotonicidade, prova formal logo abaixo).
   No nível 2, como todo par vem de dois itens já frequentes, ninguém é
   podado aqui ainda; é no nível 3 que essa poda derruba 3 dos 4
   candidatos de trinca só olhando os pares, sem sequer voltar aos
   decks.
4. **Contagem real** dos candidatos que sobraram da poda, descartando de
   novo quem não bate o suporte mínimo. No exemplo, `{Escudo, Poção}` e
   `{Espada, Poção}` têm suporte 0,3, abaixo do mínimo, e caem aqui.
5. Repete os passos 2 a 4 pro próximo tamanho de itemset, usando só quem
   sobreviveu até agora, até não sobrar mais nenhum itemset frequente
   novo pra tentar juntar. No exemplo dos decks, isso acontece já no
   nível 3: a única trinca que sobrevive à poda por subconjunto,
   `{Dragão, Escudo, Espada}`, tem suporte real de 0,3, abaixo do
   mínimo, então o algoritmo para ali, nível 4 nem chega a ser tentado.

**Prova da propriedade Apriori (por que a poda do passo 3 é segura):**
se um itemset `X` é frequente, então `suporte(X) >= min_suporte`. Qualquer
subconjunto `Y ⊂ X` aparece em pelo menos todas as transações onde `X`
aparece (toda transação que contém `X` contém `Y` também, já que `Y` é
uma parte de `X`), logo `suporte(Y) >= suporte(X) >= min_suporte`, ou
seja, `Y` também é frequente. Virando ao contrário (contrapositiva): se
um subconjunto `Y` NÃO é frequente, nenhum itemset `X` que contenha `Y`
pode ser frequente. É exatamente essa garantia que permite jogar fora um
candidato inteiro sem gastar nenhum tempo contando ele nos dados de
novo, só verificando se algum pedacinho dele já falhou antes.

## Fórmulas (decorar)

- **Suporte**: `suporte(X) = (número de transações que contêm X) / (número total de transações)`.
  É simplesmente "quão comum é esse itemset", igual em qualquer ordem: o
  suporte de `{Dragão, Espada}` é o mesmo suporte de `{Espada, Dragão}`.
- **Confiança**: `confiança(A -> B) = suporte(A ∪ B) / suporte(A)`.
  É a versão da regra pra probabilidade condicional: "dado que A
  apareceu, em quantos por cento das vezes B também apareceu?". Repara
  que confiança NÃO é simétrica: `confiança(A -> B)` e
  `confiança(B -> A)` costumam dar números diferentes, porque os
  denominadores (`suporte(A)` e `suporte(B)`) são diferentes.
- **Lift**: `lift(A -> B) = confiança(A -> B) / suporte(B) = suporte(A ∪ B) / (suporte(A) * suporte(B))`.
  Mede o quanto a presença de A muda a chance de B, comparado a B
  aparecer sozinho por acaso. `lift = 1` quer dizer que A e B são
  independentes (achar A não muda nada a chance de achar B); `lift > 1`
  é associação positiva de verdade; `lift < 1` é associação negativa (A e
  B tendem a se EVITAR). Ao contrário da confiança, o lift É simétrico:
  `lift(A -> B) = lift(B -> A)` sempre, porque a fórmula
  `suporte(A ∪ B) / (suporte(A) * suporte(B))` não muda se trocar A por
  B.
- **Pegadinha clássica de prova**: confiança alta não garante associação
  de verdade. No exemplo dos decks, `Escudo -> Dragão` tem confiança de
  0,667 (bem alta), mas lift de 0,952 (abaixo de 1). Isso acontece porque
  Dragão já é extremamente popular sozinho (suporte 0,7): achar Dragão
  junto de Escudo não é surpresa nenhuma, é só reflexo dessa popularidade
  geral, não uma combo de verdade entre as duas cartas. É por isso que
  toda regra de associação decente reporta suporte, confiança E lift
  juntos, nunca confiança sozinha.

## Parâmetros

- `min_suporte`: corta itemsets raros demais pra importar. Suporte
  mínimo alto gera poucas regras, mas bem estabelecidas; suporte mínimo
  baixo demais deixa o algoritmo lento (poda menos candidato) e pode
  gerar regras espúrias, só coincidência de amostra pequena.
- `min_confiança`: corta regras onde o consequente não aparece com
  frequência suficiente dado o antecedente. Não filtra sozinho o
  problema da pegadinha acima (uma regra pode ter confiança alta e lift
  baixo ao mesmo tempo).
- `min_lift` (opcional, mas recomendado): filtra direto pelas regras que
  são associação de verdade (`lift > 1`), resolvendo a pegadinha da
  confiança de uma vez.
- Em dado raro (como fraude, 0,17% das transações), `min_suporte`
  precisa ser bem menor que de costume, do contrário nenhum itemset que
  contenha o evento raro chega a ser considerado frequente nunca. Ver
  `apriori.py`, Parte 3: mesmo as melhores regras encontradas têm
  confiança absoluta baixa (a maior é 1,5%), porque fraude é rara demais
  pra qualquer confiança absoluta parecer alta; o que importa ali é o
  lift, que passa de 8 pra combinação das três faixas mais fortes.

## Apriori x FP-Growth

- Apriori gera candidatos explicitamente, item por nível, e escaneia o
  dataset inteiro de novo a cada nível pra contar suporte. Simples de
  entender e implementar, mas caro: muitas passadas pelos dados, muitos
  candidatos intermediários gerados na memória.
- FP-Growth (Frequent Pattern Growth) evita gerar candidatos: comprime o
  dataset inteiro numa estrutura de árvore (a FP-tree, uma árvore de
  prefixos compartilhados entre transações parecidas) numa única
  passada, e depois minera os itemsets frequentes direto dessa árvore,
  sem nunca montar e testar combinação por combinação.
- Pegadinha de prova: os dois algoritmos, com o mesmo `min_suporte`,
  encontram exatamente os MESMOS itemsets frequentes no final (o
  resultado matemático é idêntico); a diferença entre eles é só de
  desempenho computacional, FP-Growth costuma ser bem mais rápido em
  datasets grandes ou com muitos itens distintos.

## Aspectos Positivos

- Não supervisionado: não precisa de rótulo nenhum, só das transações.
- Resultado fácil de interpretar: cada regra já vem numa frase pronta
  ("quem tem A tem B").
- A propriedade Apriori garante que a poda nunca descarta por engano um
  itemset que seria de fato frequente (não é uma aproximação, é exato).

## Aspectos Negativos

- Escala mal com muitos itens distintos e `min_suporte` baixo: o número
  de candidatos intermediários pode crescer bastante antes da poda
  conseguir cortar, principalmente nos primeiros níveis.
- Não captura ordem nem quantidade, só presença/ausência do item na
  transação (não diferencia comprar 1 ou comprar 10 do mesmo produto).
- Precisa dado categórico (itens discretos). Atributo contínuo (como os
  componentes de PCA do dataset de fraude) tem que ser discretizado em
  faixas antes, uma etapa de pré-processamento que introduz suas
  próprias escolhas (quantos "baldes", onde cortar), como no
  `_discretizar_em_cesta` de `apriori.py`.
- Muitas regras encontradas são redundantes ou óbvias (regras com lift
  perto de 1, como no exemplo de `Escudo -> Dragão`); filtrar as regras
  que sobram exige julgamento humano além dos números.

## Ver também

- `apriori/apriori.py`: geração de candidatos e poda feitas na mão, nível
  por nível, com o exemplo dos decks, e a mineração de verdade (via
  `mlxtend`, o scikit-learn não implementa Apriori) no dataset de fraude
  discretizado em faixas.
