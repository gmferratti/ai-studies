# Revisão de Fundamentos Clássicos e Modernos de Machine Learning e IA

Projeto pessoal de estudo, implementação prática e revisão de conceitos de Machine Learning e Inteligência Artificial, do básico ao estado da arte: aprendizado supervisionado e não supervisionado, redes neurais e deep learning, aprendizado por reforço, grandes modelos de linguagem, processamento de linguagem natural, séries temporais, aprendizado em fluxo, e os temas mais conceituais de ética, explicabilidade e representação do conhecimento.

O objetivo é ir além da teoria: cada módulo tem implementações práticas comparáveis entre si, sobre um dataset comum, para consolidar intuição sobre quando e por que usar cada abordagem.

## Estrutura do repositório

```
.
├── utils/
│   └── data_utils.py                 # funções compartilhadas de carga e pré-processamento de dados
├── 01_aprendizado_supervisionado_nao_supervisionado/
│   ├── decision_tree.py
│   ├── knn.py
│   ├── svm.py
│   ├── naive_bayes.py
│   └── README.md                     # comparação entre os algoritmos do módulo
├── 02_redes_neurais_deep_learning/
├── 03_aprendizado_por_reforco/
├── 04_grandes_modelos_de_linguagem/
├── 05_pln_series_temporais_streaming/
└── 06_etica_explicabilidade_representacao_conhecimento/
```

Cada módulo tem seu próprio README com contexto, decisões de implementação e resultados. O repositório é atualizado incrementalmente conforme os estudos avançam.

## Como rodar

O projeto usa [uv](https://docs.astral.sh/uv/) como gerenciador de ambientes:

```bash
uv sync
```

Cada script pode ser executado isoladamente via `uv run`, por exemplo:

```bash
uv run 01_aprendizado_supervisionado_nao_supervisionado/decision_tree.py
```

## Dataset base

Para o módulo de aprendizado supervisionado e não supervisionado, uso o dataset público [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud) como base comum para comparar os algoritmos clássicos de classificação sob um cenário real de forte desbalanceamento de classes. Os dados não são versionados neste repositório (ver `.gitignore`); instruções de download estão em `utils/data_utils.py`.

## Roadmap

- [ ] Aprendizado supervisionado e não supervisionado
- [ ] Redes neurais e deep learning
- [ ] Aprendizado por reforço
- [ ] Grandes modelos de linguagem
- [ ] PLN, séries temporais e aprendizado em fluxo
- [ ] Ética, explicabilidade e representação do conhecimento

## Licença

MIT
