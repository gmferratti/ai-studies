# Instruções para este repositório

Repositório de estudo pessoal: revisão de fundamentos
de Machine Learning e IA, do clássico ao estado da arte (ver `README.md`
pros módulos). O objetivo não é só ter código que funciona, mas ter material
didático bom o bastante pra assimilar os algoritmos.

O arquivo de referência de estilo é
`01_aprendizado_supervisionado_nao_supervisionado/decision_tree/decision_tree.py`.
Antes de escrever um novo script didático (`knn.py`, `svm.py`,
`naive_bayes.py`, ou qualquer algoritmo de outro módulo), leia esse arquivo
inteiro pra pegar o tom.

## Estrutura de cada script didático

Cada algoritmo mora na sua própria subpasta dentro do módulo, com o mesmo
nome do arquivo principal, por exemplo:

```
01_aprendizado_supervisionado_nao_supervisionado/
├── decision_tree/
│   ├── decision_tree.py
│   └── images/           <- todo gráfico gerado pelo script vai aqui
├── knn/
│   ├── knn.py
│   └── images/
```

Dentro do `.py`, a ordem é sempre:

1. **Docstring do módulo com a teoria**, explicada com analogia antes da
   fórmula, nunca só a fórmula solta. Sem pré-requisito de programação ou
   estatística pra acompanhar.
2. **Exemplo de brinquedo** (poucos exemplos, dataset inventado na mão) que
   reproduz o cálculo do algoritmo passo a passo, com prints no terminal.
   Se o algoritmo tem uma etapa recursiva ou iterativa (tipo uma árvore que
   divide de novo dentro de um galho misturado), o exemplo de brinquedo
   precisa mostrar isso acontecendo de verdade, com mais de uma variável se
   for o caso, não só descrever em texto que "aconteceria".
3. **Visualização** do mesmo exemplo de brinquedo (gráfico, diagrama,
   fronteira de decisão, o que fizer sentido pro algoritmo), salva em
   `images/` dentro da pasta do script.
4. **Treino de verdade**, reaproveitando `utils/data_utils.py` pra usar o
   mesmo dataset e split dos outros algoritmos do módulo (dataset de fraude
   em cartão de crédito, no módulo 01).

## Tom e linguagem

- Escreva pra alguém "nerdola" que curte referências de jogo, RPG, anime,
  ficção. Escolha uma analogia que combine com a MECÂNICA do algoritmo (não
  é decoração, é pra ajudar a lembrar como o algoritmo funciona). Exemplos já
  usados: Chapéu Seletor de Hogwarts e Akinator pra árvore de decisão
  (perguntas em sequência até uma decisão), Pokémon pro exemplo numérico.
  Não repita a mesma referência em todo script; escolha a que encaixa melhor
  com o algoritmo em questão (ex.: k-NN combina com "quem são meus vizinhos
  de time/clã mais próximos", SVM combina com "a maior distância de
  segurança entre dois grupos rivais").
- Assuma que quem está lendo perdeu o contato com exatas. Mastigue tudo: intuição
  antes de fórmula, tradução de cada símbolo em português claro, nada de
  jargão sem explicar.
- Nunca use travessão (—). Use ponto, vírgula ou dois-pontos.
- Nada de linguagem que soa gerada por IA: sem "vamos explorar", "é
  importante notar", "em suma", "sem dúvida", "cabe destacar" e afins. Escreva
  como quem está explicando pra um amigo.
- Sem comentário ou docstring que só repete o nome da função. Comentário só
  quando explica um porquê não óbvio.

## Código

- Python idiomático, nomes de função e variável em português, verbos no
  infinitivo pra ação (`plotar_...`, `desenhar_...`, `explicar_...`) e
  substantivo pra cálculo (`entropia`, `distancia`, `margem`).
- Funções pequenas, uma responsabilidade cada. Se uma função de demonstração
  passar de uns 40-50 linhas, quebre em funções auxiliares (prefixo `_` pra
  helper interno do módulo).
- Nenhuma duplicação de setup entre funções de plot (backend do matplotlib,
  caminho de saída, etc.): extraia um helper compartilhado.
- Todo `matplotlib` roda com backend `Agg` (sem display) e salva em arquivo,
  nunca `plt.show()`.
- Sempre rode o script inteiro (`uv run <script>`) depois de qualquer mudança
  pra confirmar que os números batem e os gráficos saem certos antes de
  considerar a tarefa pronta.

## Outros lembretes do repositório

- `utils/data_utils.py` é compartilhado por todos os algoritmos do módulo 01;
  mudanças ali afetam todo mundo, teste com mais de um script se mexer nele.
- O dataset de fraude (`data/creditcard.csv`) não é versionado (`.gitignore`);
  instruções de download estão no topo de `utils/data_utils.py`.
- Cada módulo (`01_...` a `06_...`) tem seu próprio README com o checklist de
  algoritmos e a tabela de comparação de resultados. Atualize a tabela do
  módulo quando um novo algoritmo terminar de treinar.
