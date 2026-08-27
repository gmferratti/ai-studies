# Aprendizado Supervisionado e Não Supervisionado

Revisão de algoritmos clássicos de aprendizado de máquina, cada um com uma
explicação mastigada (analogia antes da fórmula), um exemplo de brinquedo
calculado na mão, visualização, e o treino de verdade sobre um dataset real
comum a todos: detecção de fraude em cartão de crédito.

Os algoritmos ficam divididos em duas famílias: supervisionado (aprende a
partir de exemplos rotulados) e não supervisionado (busca padrões em dados
sem rótulo). Ver `ementa.md` pro cronograma de estudo.

## Estrutura do repositório

```
.
├── utils/
│   └── data_utils.py          # carga e pré-processamento compartilhados
├── 01-supervisionado/          # árvore de decisão, k-NN, SVM, Naive Bayes, bagging, boosting, random forest
├── 02-nao_supervisionado/      # clustering, Apriori, PCA
├── 03-outros/                  # fundamentos, avaliação de modelos, PAC-learning: teoria que não é um método em si
└── ementa.md                  # cronograma de estudo desta fase
```

- [`01-supervisionado/README.md`](01-supervisionado/README.md): checklist de
  algoritmos e tabela de comparação de resultados.
- [`02-nao_supervisionado/README.md`](02-nao_supervisionado/README.md): idem,
  pra família não supervisionada.
- [`03-outros/README.md`](03-outros/README.md): formalização do problema de
  aprendizado, viés indutivo, métricas de avaliação e PAC-learning.

## Como rodar

O projeto usa [uv](https://docs.astral.sh/uv/) como gerenciador de ambientes:

```bash
uv sync
```

Cada script pode ser executado isoladamente via `uv run`, por exemplo:

```bash
uv run 01-supervisionado/decision_tree/decision_tree.py
```

## Dataset base

Uso o dataset público [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)
como base comum para comparar os algoritmos clássicos de classificação sob
um cenário real de forte desbalanceamento de classes. Os dados não são
versionados neste repositório (ver `.gitignore`); instruções de download
estão em `utils/data_utils.py`.

## Outros repositórios

Esse é um de vários repositórios de revisão de fundamentos de Machine
Learning e IA, cada um dedicado a um tópico:

- [deep-learning](https://github.com/gmferratti/deep-learning): redes neurais e deep learning
- [reinforcement-learning](https://github.com/gmferratti/reinforcement-learning): aprendizado por reforço
- [large-language-models](https://github.com/gmferratti/large-language-models): grandes modelos de linguagem
- [nlp](https://github.com/gmferratti/nlp): processamento de linguagem natural
- [time-series](https://github.com/gmferratti/time-series): séries temporais
- [streaming-ml](https://github.com/gmferratti/streaming-ml): aprendizado em fluxo
- [ai-ethics-explainability](https://github.com/gmferratti/ai-ethics-explainability): ética, explicabilidade e representação do conhecimento

## Licença

MIT
