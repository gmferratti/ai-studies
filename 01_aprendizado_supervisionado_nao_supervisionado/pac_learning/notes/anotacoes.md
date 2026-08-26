# PAC-Learning (Probably Approximately Correct)

- Formalização proposta por Valiant (1984), retomada no AIMA (Russell & Norvig) no capítulo de aprendizado a partir de exemplos.
- Pergunta que ela responde: com um número FINITO de exemplos de treino, quando dá pra garantir que a hipótese aprendida vai se sair bem em exemplos novos, nunca vistos? Sem essa garantia, "aprender" seria só decorar o treino, sem nenhuma promessa sobre o futuro.

## Vocabulário básico

- Conceito alvo (`c`): a regra verdadeira que queremos aprender (ex.: "esta transação é fraude", de verdade, não a versão estimada).
- Espaço de hipóteses (`H`): o conjunto de todas as regras que o algoritmo é capaz de considerar (ex.: todas as árvores de decisão até profundidade 5, ou todas as retas separadoras no caso do SVM linear).
- Hipótese aprendida (`h`): a regra específica que o algoritmo escolheu dentro de `H`, depois de olhar pro treino.
- Distribuição `D`: a distribuição de probabilidade de onde os exemplos (treino e teste) são sorteados. Assume-se que treino e exemplos futuros vêm da MESMA distribuição, senão não tem base nenhuma pra generalizar.
- Erro verdadeiro de `h`: `erro(h) = P(h(x) != c(x))`, ou seja, a probabilidade de `h` errar num exemplo `x` sorteado de `D`. É diferente do erro medido no treino: o erro verdadeiro é sobre TODOS os exemplos possíveis, não só os que já vimos.

## O nome "Provavelmente Aproximadamente Correto"

O nome descreve os dois parâmetros que controlam a garantia:

- **Aproximadamente correto**: não exigimos `erro(h) = 0` (impossível de garantir na prática), só `erro(h) <= epsilon`, onde `epsilon` é o quanto de erro toleramos (ex.: epsilon = 0,05 aceita até 5% de erro).
- **Provavelmente**: como o treino é uma amostra aleatória, pode calhar de sair uma amostra "azarada" que engana o algoritmo. Por isso a garantia não é absoluta, só vale com probabilidade `>= 1 - delta`, onde `delta` é a chance que aceitamos de dar azar (ex.: delta = 0,01 aceita 1% de chance de a garantia falhar).

Juntando os dois: queremos que, com probabilidade pelo menos `1 - delta` (sobre o sorteio do treino), a hipótese devolvida tenha `erro(h) <= epsilon`.

## Complexidade amostral (quantos exemplos preciso?)

Caso mais simples: `H` finito, e o algoritmo sempre devolve uma hipótese CONSISTENTE (que acerta 100% do treino, o chamado caso "realizável", onde o conceito alvo está mesmo dentro de `H`).

`m >= (1/epsilon) * (ln|H| + ln(1/delta))`

Traduzindo cada pedaço:

- `m`: número mínimo de exemplos de treino necessários.
- `|H|`: tamanho do espaço de hipóteses (quantas regras diferentes o algoritmo poderia escolher). Espaço de hipóteses maior (algoritmo mais flexível, capaz de representar mais regras) exige mais exemplos, porque tem mais chance de alguma hipótese ruim acertar o treino por pura coincidência.
- `1/epsilon`: quanto menor a margem de erro tolerada, mais exemplos precisa.
- `1/delta`: quanto menor a chance de falha aceita, mais exemplos precisa (mas cresce devagar, dentro de um logaritmo).

De onde vem essa fórmula (raciocínio, não decoreba): uma hipótese "ruim" é aquela com `erro(h) > epsilon`. A chance de UMA hipótese ruim específica acertar por acaso um exemplo é `<= (1 - epsilon)`, e acertar `m` exemplos seguidos (o treino inteiro) é `<= (1 - epsilon)^m <= e^(-epsilon*m)`. Como podem existir até `|H|` hipóteses ruins diferentes, soma-se essa chance de azar `|H|` vezes (union bound: `P(A ou B) <= P(A) + P(B)`), e pede-se que o total fique abaixo de `delta`. Resolvendo `|H| * e^(-epsilon*m) <= delta` pra `m`, cai na fórmula acima.

## Caso agnóstico e VC dimension

O caso realizável assume que o conceito alvo está dentro de `H` e que dá pra achar uma hipótese com erro zero no treino. Na prática (dados com ruído, ou problema mais complexo do que `H` consegue representar), isso quase nunca acontece: é o caso agnóstico.

- Pra `H` infinito (ex.: todas as retas possíveis no plano, no caso de um SVM linear), `ln|H|` não faz sentido, porque `|H|` é infinito. A teoria PAC troca `ln|H|` pela dimensão VC de `H` (Vapnik-Chervonenkis dimension): o tamanho do maior conjunto de pontos que `H` consegue separar de TODAS as formas possíveis (conceito chamado de "shatter"). Uma reta no plano, por exemplo, tem VC dimension 3: consegue separar 3 pontos de qualquer jeito, mas já existe alguma configuração de 4 pontos que nenhuma reta separa.
- A fórmula de complexidade amostral no caso agnóstico fica `m = O((1/epsilon²) * (VCdim(H) + ln(1/delta)))`: repara que agora `epsilon` aparece ao quadrado no denominador, ficando mais caro (precisa de mais exemplos) do que no caso realizável.

## Relação com viés-variância e generalização

- `H` mais rico (mais hipóteses possíveis, ou VC dimension maior) consegue representar regras mais complexas, viés menor, mas exige mais exemplos pra generalizar bem, porque tem mais chance de "acertar por sorte" no treino com uma hipótese ruim.
- `H` mais restrito (menos hipóteses) precisa de menos exemplos, mas corre o risco de nem conseguir representar o conceito alvo, viés maior.
- É o mesmo trade-off viés x variância que aparece em poda de árvore e em k, só que aqui formalizado em termos de quantos exemplos são necessários pra cada nível de expressividade do modelo.

## Limitação prática

Os limites que a teoria PAC calcula costumam ser bem folgados (pessimistas): na prática, algoritmos generalizam bem com muito menos exemplos do que a fórmula exige. A utilidade principal não é calcular o `m` exato a usar em produção, é entender a DIREÇÃO dos efeitos: mais exemplos reduzem erro e aumentam confiança; espaço de hipóteses mais expressivo exige mais exemplos; e existe sempre um limite formal pra generalização, aprender não é mágica, é estatística.