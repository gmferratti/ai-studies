# Treino de redação: prova escrita

## Tema sorteado (simulado)

**1. Aprendizado Supervisionado: Fundamentos, principais algoritmos e
formas de avaliação.**

## Como treinar

1. Cronometre. Se não souber o tempo real da sua prova, use 45-50 minutos
   pra esse tema (ele pede três partes: fundamentos, algoritmos, avaliação).
2. Não consulte `revisao/semana1_resumo.md` nem as `notes/anotacoes.md`
   enquanto escreve. A prova não vai ter consulta; treinar com consulta
   treina a mão, não a memória.
3. Escreva direto na seção "Sua redação" deste arquivo, ou no papel se
   preferir simular a prova de verdade.
4. Só depois de terminar, revise com o checklist lá embaixo e, se quiser,
   me mande o texto pra eu corrigir como uma banca corrigiria.

## Estrutura esperada (não é gabarito, é o que uma resposta forte cobre)

1. **Introdução curta**: o que é aprendizado supervisionado (aprende a
   partir de exemplos rotulados, `D = {(x_i,y_i)}`), contraste rápido com
   não supervisionado (sem rótulo) e por reforço (aprende por
   recompensa/punição, não por exemplo rotulado).
2. **Fundamentos**: formalização do problema (`X`, `Y`, conceito-alvo
   `c`, espaço de hipóteses `H`, hipótese aprendida `h`, indução), viés
   indutivo (por que é necessário, os dois tipos: restrição e
   preferência, um exemplo de cada), erro empírico x erro de
   generalização.
3. **Principais algoritmos**: não liste decorado um atrás do outro, sem
   costura. Agrupe por família de viés indutivo: baseados em regras
   (árvore de decisão), baseados em distância (k-NN), baseados em margem
   (SVM), probabilísticos (Naive Bayes). Pra cada um: a mecânica central
   em uma ou duas frases, e o viés indutivo específico dele.
4. **Formas de avaliação**: por que acurácia sozinha engana em dados
   desbalanceados, matriz de confusão, precisão, recall, F1, AUC-ROC,
   validação cruzada.
5. **Conclusão**: amarre tudo. A escolha de algoritmo e de métrica de
   avaliação dependem do problema (No Free Lunch de novo: não existe
   algoritmo nem métrica universalmente melhor).

## Aviso sobre a parte de avaliação

"Formas de avaliação" ainda não tem nota escrita neste repositório (é
ementa de Terça 01/09, Semana 2). Escreva essa parte com o que você já
viveu na prática esta semana: toda tabela de resultado usou precisão,
recall e F1 da classe fraude, nunca acurácia sozinha, porque o dataset
tem só 0,17% de fraude. Se escrever "acurácia" como única métrica pra um
problema desbalanceado, é exatamente o erro que a prova quer testar se
você não comete.

## Termos que uma resposta completa deveria conseguir usar sem gaguejar

Fundamentos: espaço de instâncias, espaço de hipóteses, conceito-alvo,
indução, viés indutivo (restrição/preferência), erro empírico, erro de
generalização, No Free Lunch.

Árvore de decisão: indução top-down, gulosa, entropia, ganho de
informação, índice Gini, pré-poda, pós-poda, overfitting.

k-NN: lazy learning, distância euclidiana, K, voto ponderado,
normalização, maldição da dimensionalidade.

SVM: hiperplano, margem máxima, vetor de suporte, margem suave, variável
de folga, parâmetro C, kernel, truque do kernel.

Naive Bayes: Teorema de Bayes, prior, verossimilhança, posterior,
independência condicional, critério MAP, suavização de Laplace.

Avaliação: matriz de confusão, precisão, recall, F1, AUC-ROC, validação
cruzada.

## Checklist de autocorreção (preencha depois de escrever)

- [ ] Toquei nas três partes do tema (fundamentos, algoritmos, avaliação),
      não só numa ou duas?
- [ ] Defini os termos formais (`X`, `H`, `h`, `c`) em vez de só usar os
      nomes soltos?
- [ ] Expliquei o que é viés indutivo e dei pelo menos um exemplo de
      viés de restrição e um de preferência?
- [ ] Descrevi pelo menos 3 dos 4 algoritmos com a mecânica central
      certa (não só o nome)?
- [ ] Expliquei por que acurácia sozinha não basta em dados
      desbalanceados, com um exemplo?
- [ ] Usei pelo menos uma métrica além de acurácia (precisão, recall, F1
      ou AUC-ROC) e disse quando cada uma importa mais?
- [ ] A conclusão amarrou fundamentos, algoritmos e avaliação, ou ficou
      cada parte solta?
- [ ] Escrevi dentro do tempo cronometrado?

## Sua redação

<!-- Escreva aqui embaixo, sem consultar as notas. -->
