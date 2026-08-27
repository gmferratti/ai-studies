# SVM (Support Vector Machines)

- Classificador que desenha uma fronteira entre classes maximizando a distância de segurança até os exemplos mais próximos de cada lado.
- Nasce pra classificação binária (duas classes), mas se estende pra várias classes, pra regressão e até pra agrupamento, mudando o que a função de otimização tenta minimizar.
- Fortemente embasado na Teoria do Aprendizado Estatístico (Vapnik e Chervonenkis): a escolha de maximizar a margem não é só uma ideia geométrica bonita, tem prova matemática por trás de por que isso ajuda a generalizar melhor.

## Analogia central: a linha de frente entre dois reinos rivais

Imagine dois reinos em guerra, cada um com suas tropas espalhadas pelo mapa. Alguém precisa desenhar a fronteira entre os dois territórios, e a pior escolha possível é colar a linha bem em cima das últimas tropas de um dos lados: qualquer avanço mínimo do inimigo já invade território. A escolha mais segura é traçar a fronteira bem no meio da "terra de ninguém", o mais longe possível das tropas de ambos os lados ao mesmo tempo. Essa distância de segurança, dos dois lados da fronteira até a tropa mais avançada de cada reino, é a margem. O SVM é como se fosse um general cuidadoso, entre todas as fronteiras possíveis que separam os dois exércitos, ele escolhe a que deixa a maior margem de segurança.

Só que a fronteira final não depende de todo mundo. Só as tropas que estão bem na linha de frente, coladas na margem, é que decidem onde a fronteira fica. Essas tropas da linha de frente são os vetores de suporte que dão nome ao algoritmo. Mover qualquer soldado da retaguarda pra qualquer lugar, contanto que continue bem atrás da linha de frente do próprio reino, não muda essa fronteira.

## Vocabulário básico

- Hiperplano: a fronteira que separa as classes. Em 2D é uma reta, em 3D é um plano, em mais dimensões é um "hiperplano" (o nome genérico pra essa fronteira reta em qualquer número de dimensões). Equação: `w · x + b = 0`, onde `w` é o vetor de pesos (a orientação da fronteira) e `b` é o viés (o quanto ela se desloca da origem).
- Margem: a distância entre o hiperplano e o exemplo mais próximo de cada classe. O SVM maximiza essa distância.
- Vetor de suporte: exemplo de treino que fica exatamente em cima da margem (a tropa da linha de frente). São os únicos exemplos que participam da conta final do hiperplano.
- C: parâmetro que controla o quanto o modelo tolera exemplos dentro da margem ou mal classificados, em troca de uma margem mais larga (ver seção de margem suave).
- Kernel: uma função que troca a distância "reta" original por uma outra forma de medir semelhança entre exemplos, permitindo fronteiras curvas sem sair do formato matemático de "hiperplano" (ver seção do truque do kernel).

## Fronteira linear e maximização da margem

Com duas classes rotuladas como +1 e -1, o hiperplano `w · x + b = 0` separa o espaço em dois lados: exemplos onde `w · x + b > 0` são previstos como +1, onde `w · x + b < 0` são previstos como -1. Só que existem infinitos hiperplanos que separam corretamente um conjunto de pontos linearmente separáveis. O SVM escolhe, entre todos eles, o que maximiza a margem.

A margem tem um valor exato: `margem = 2 / ||w||`, onde `||w||` é o comprimento do vetor `w` (a raiz quadrada da soma dos quadrados dos seus componentes). Maximizar a margem é equivalente a minimizar `||w||`, sujeito à restrição de que nenhum exemplo de treino fique dentro da margem:

```
minimizar (1/2) ||w||²
sujeito a  y_i (w · x_i + b) >= 1,  para todo exemplo i
```

Aqui `y_i` é o rótulo do exemplo i (+1 ou -1). A restrição `y_i (w · x_i + b) >= 1` é uma forma compacta de dizer "todo exemplo tem que estar do lado certo da fronteira, e a pelo menos uma margem de distância dela": se `y_i=+1`, precisa que `w·x_i+b >= 1`; se `y_i=-1`, precisa que `w·x_i+b <= -1`. Isso é justamente a restrição de que não haja nenhum dado de treino dentro da faixa entre as duas margens: essa versão sem tolerância nenhuma é chamada de margem rígida (hard margin).

Os exemplos onde essa restrição vale com igualdade exata (`y_i(w·x_i+b) = 1`) são os vetores de suporte: estão exatamente em cima da margem. Ver o cálculo à mão desse hiperplano, com só dois vetores de suporte (um de cada reino), em `svm.py`.

## Por que margem larga generaliza melhor: Teoria do Aprendizado Estatístico

Vapnik e Chervonenkis mostraram que o erro que um classificador vai cometer em dados novos (nunca vistos) pode ser limitado por uma soma de duas coisas: o erro que ele já comete nos dados de treino, e a complexidade do modelo (medida por um número chamado dimensão VC). Complexidade demais é a raiz do overfitting: um modelo complexo demais decora o treino e generaliza mal, mesmo com erro zero no treino.

Uma fronteira de margem larga corresponde a um modelo de complexidade menor nesse sentido formal: existem menos hiperplanos possíveis com margem grande do que hiperplanos com margem apertada, então restringir a busca aos de margem larga já poda o espaço de hipóteses, sem depender de contar atributos ou ajustar manualmente a complexidade do modelo. É esse resultado teórico, o princípio da minimização do risco estrutural, que dá o nome de "aprendizado estatístico" à teoria por trás do SVM: em vez de só minimizar o erro observado no treino (risco empírico), o SVM otimiza uma combinação que já leva em conta a complexidade do modelo, o que se traduz na prática em maximizar a margem.

## Fronteiras não lineares: o truque do kernel

Fronteira reta ajuda quando as classes já são separáveis por uma reta ou plano, mas boa parte dos problemas reais tem classes entrelaçadas de um jeito que nenhuma reta separa direito (o exemplo clássico é um XOR: duas classes em diagonais opostas de um quadrado). A saída não é abandonar o hiperplano, é mudar o espaço onde ele é desenhado.

A ideia: existe uma função `φ` (fi) que "levanta" cada ponto pra um espaço com mais dimensões, onde as classes que pareciam embaralhadas em 2D viram separáveis por um hiperplano reto (mesmo que esse hiperplano, quando projetado de volta pro espaço original, pareça uma curva). O problema é que calcular `φ(x)` explicitamente pode ser caro ou até ter dimensão infinita. O truque do kernel evita esse cálculo: em vez de calcular `φ(x)` pra cada ponto e depois multiplicar, existe uma função `K(x, x') = φ(x) · φ(x')` que calcula diretamente o produto final, sem nunca precisar montar o vetor `φ(x)` inteiro. Como o algoritmo do SVM (na forma dual, ver seção abaixo) só usa produtos entre pares de exemplos, basta trocar cada produto `x · x'` por `K(x, x')` em todas as contas, e o hiperplano linear no espaço "levantado" vira, de volta no espaço original, uma fronteira curva.

Kernels mais usados:

| Kernel | Fórmula | Quando usar |
|---|---|---|
| Linear | `K(x, x') = x · x'` | Classes já aproximadamente separáveis por reta/plano. |
| Polinomial | `K(x, x') = (x · x' + c)^d` | Fronteiras curvas suaves, grau `d` controla a complexidade da curva. |
| RBF (gaussiano) | `K(x, x') = exp(-γ ||x - x'||²)` | Fronteiras bem irregulares; o mais usado na prática quando não se sabe o formato da fronteira de antemão. |

No RBF, `γ` (gama) controla o "alcance" de cada exemplo: `γ` grande faz a influência de cada ponto cair rápido com a distância (fronteira mais recortada, parecida com K pequeno no k-NN), `γ` pequeno faz a influência se espalhar mais longe (fronteira mais suave). Ver a demonstração com um XOR de brinquedo em `svm.py`.

## Margem suave e variáveis de folga

Na prática, é raríssimo achar um problema real cujos dados sejam perfeitamente separáveis, seja por reta ou por qualquer kernel: quase sempre existe pelo menos um exemplo ruidoso, mal rotulado, ou genuinamente ambíguo, que cai do lado errado ou bem no meio da margem. Se o SVM insistisse na margem rígida (zero exemplos dentro da margem, sem exceção), bastaria um único exemplo assim pra tornar o problema inteiro impossível de resolver, ou pra forçar uma margem tão apertada e distorcida que perde toda a vantagem de generalização.

A correção é permitir folga: pra cada exemplo, uma variável `ξ_i` (csi, sempre >= 0) mede o quanto aquele exemplo especificamente pode violar sua margem. `ξ_i = 0` quer dizer que o exemplo está corretamente posicionado, fora da margem; `0 < ξ_i < 1` quer dizer que invadiu a margem mas ainda está do lado certo da fronteira; `ξ_i >= 1` quer dizer que foi parar do lado errado, um erro de classificação de fato. O problema de otimização vira:

```
minimizar (1/2) ||w||² + C * Σ ξ_i
sujeito a  y_i (w · x_i + b) >= 1 - ξ_i,  ξ_i >= 0
```

O parâmetro `C` é o preço que cada unidade de folga paga na conta: `C` grande torna cada violação muito cara, então o modelo se aproxima da margem rígida, tentando acertar até os exemplos mais teimosos, o que costuma estreitar a margem e ficar refém de ruído (um único "infiltrado" pode distorcer a fronteira inteira). `C` pequeno torna a folga barata, então o modelo aceita alguns exemplos mal posicionados de bom grado em troca de uma margem bem mais larga, geralmente mais estável em dados novos. `C` é, na prática, o parâmetro mais importante de ajustar num SVM. Ver esse efeito acontecendo com um "espião infiltrado" no meio do exército rival, em `svm.py`.

## Problema de otimização dual

O jeito como as fórmulas acima foram escritas (achar `w` e `b` direto) é chamado de forma primal. Na prática, o SVM quase sempre é resolvido numa forma equivalente chamada forma dual, que troca as variáveis `w` e `b` por um multiplicador `α_i` (alfa) pra cada exemplo de treino:

```
maximizar   Σ α_i - (1/2) ΣΣ α_i α_j y_i y_j K(x_i, x_j)
sujeito a   0 <= α_i <= C,   Σ α_i y_i = 0
```

Duas vantagens da forma dual explicam por que ela é a preferida na prática:

- Ela só depende de produtos `K(x_i, x_j)` entre pares de exemplos, nunca dos exemplos "crus" sozinhos. É exatamente essa propriedade que permite o truque do kernel: basta trocar `K` de fórmula pra mudar o formato da fronteira, sem reescrever o resto do algoritmo.
- No ótimo, a maioria dos `α_i` sai zerada. Os únicos exemplos com `α_i > 0` são justamente os vetores de suporte: é a forma dual que identifica matematicamente quem são as "tropas da linha de frente". Com os `α_i`, recupera-se `w = Σ α_i y_i x_i` (somando só quem tem `α_i > 0`), confirmando que exemplos com `α_i = 0` (a retaguarda) não pesam nada na conta final.

## Convexidade e determinismo

O problema de otimização do SVM, primal ou dual, é convexo (tanto a função a minimizar quanto a região de restrições têm o formato de uma "tigela" sem vales escondidos). Isso garante que existe um único mínimo global, e que qualquer algoritmo numérico correto vai encontrá-lo, sem risco de ficar preso num mínimo local pior, um problema clássico de redes neurais, por exemplo. Por consequência direta, o SVM é determinístico: pra um mesmo conjunto de treino e mesmos hiperparâmetros, o resultado final (o hiperplano encontrado) é sempre exatamente o mesmo, não importa a ordem em que os exemplos de treino foram apresentados ao algoritmo. Isso contrasta com métodos que dependem de inicialização aleatória ou de embaralhamento dos dados durante o treino, onde rodar duas vezes pode dar resultados levemente diferentes.

## Extensões: regressão e agrupamento

A mesma ideia de "margem" se adapta trocando o que a função de otimização penaliza:

- SVR (Support Vector Regression): em vez de separar duas classes, o objetivo passa a ser prever um número. A margem vira um "tubo" ao redor da reta (ou curva, com kernel) de previsão: erros menores que uma tolerância `ε` (épsilon) não são penalizados, só o que sai de dentro do tubo entra na conta de erro. É a mesma lógica de "só quem está na fronteira importa", adaptada pra regressão.
- One-Class SVM / agrupamento: usado quando não há rótulo nenhum (aprendizado não supervisionado) ou quando só existe uma classe "normal" e o objetivo é detectar anomalias. A função de otimização muda pra encontrar a fronteira que separa a região de maior densidade de dados do resto do espaço, em vez de separar duas classes rotuladas.

Em todos os casos, a estrutura matemática (hiperplano, margem, kernel, forma dual) se mantém, só a função objetivo e as restrições do problema de otimização mudam pra caber no novo objetivo.

## Requisitos e custo computacional

- Só funciona com atributos numéricos: a conta de distância/produto interno por trás do hiperplano não faz sentido em cima de categorias.
- Normalização é necessária pelo mesmo motivo do k-NN: um atributo em escala muito maior que os outros domina sozinho o cálculo de `w · x`, mesmo sem ter relação real com a separação entre classes.
- Resolver o problema de otimização (primal ou dual) fica mais caro conforme o número de exemplos de treino cresce, tipicamente entre O(n²) e O(n³) dependendo do kernel e do solver, porque a matriz de produtos `K(x_i, x_j)` entre todos os pares de exemplos precisa ser calculada e manipulada. Isso torna o SVM com kernel não linear pouco prático em datasets muito grandes (centenas de milhares de exemplos ou mais), diferente de algoritmos como árvore de decisão ou regressão logística, que escalam melhor. Uma variante linear (tipo o `LinearSVC` do scikit-learn, que usa um solver diferente, sem kernel) escala bem melhor pra esses casos, ao custo de só conseguir fronteiras retas. Ver a comparação de tempo de treino em `svm.py`.
- Apesar do custo em número de exemplos, o SVM é surpreendentemente robusto a datasets com muitos atributos (alta dimensionalidade), inclusive quando há mais atributos do que exemplos de treino (comum em classificação de texto ou dados genéticos): como o kernel evita calcular o espaço "levantado" explicitamente, o custo não explode com o número de dimensões da mesma forma que explodiria em outros métodos.

## Aspectos positivos e negativos

- (+) Boa generalização: a maximização da margem tem embasamento formal na Teoria do Aprendizado Estatístico, não é só uma heurística geométrica.
- (+) Funciona bem mesmo com poucos exemplos de treino, desde que as classes sejam razoavelmente separáveis.
- (+) Robusto em datasets de alta dimensionalidade, incluindo casos com mais atributos do que exemplos.
- (+) Problema de otimização convexo: único mínimo global, e o algoritmo é determinístico (mesmo resultado independente da ordem dos dados de treino).
- (+) O truque do kernel permite fronteiras não lineares sem precisar desenhar manualmente novos atributos.
- (-) Pode ser computacionalmente custoso em datasets muito grandes, especialmente com kernels não lineares.
- (-) Sensível à escolha de hiperparâmetros (`C`, tipo de kernel, `γ`): encontrar uma boa combinação geralmente exige busca (grid search, validação cruzada), sem valor "óbvio" de antemão.
- (-) Modelo difícil de interpretar: ao contrário de uma árvore de decisão, não dá pra explicar uma previsão individual como uma sequência de perguntas; o hiperplano com kernel não linear é essencialmente uma caixa-preta geométrica.
- (-) Só lida com atributos numéricos e exige normalização antes de treinar.

## Ver também

- `svm/svm.py`: cálculo à mão do hiperplano de margem máxima com dois vetores de suporte, demonstração de margem suave com um "infiltrado", comparação de kernels num XOR de brinquedo, e o treino de verdade no dataset de fraude (`LinearSVC` no treino completo x `SVC` com kernel RBF numa amostra).
