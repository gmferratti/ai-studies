"""
Árvore de decisão aplicada ao dataset de detecção de fraude.

Conceitos revisados: indução top-down, critérios de divisão (entropia,
ganho de informação, índice Gini), poda (pré-poda e pós-poda).

---------------------------------------------------------------------------
COMO FUNCIONA UMA ÁRVORE DE DECISÃO
---------------------------------------------------------------------------
Uma árvore de decisão é um modelo de aprendizado supervisionado que
representa uma função de classificação (ou regressão) como um fluxograma:

    - Nó interno  -> testa o valor de UM atributo (ex.: "Salário > 5000?")
    - Ramo        -> um dos resultados possíveis do teste
    - Nó folha    -> uma classe prevista (ou valor, em regressão)

Para classificar uma instância nova, percorre-se a árvore da raiz até uma
folha, respondendo ao teste de cada nó interno pelo caminho.

1) INDUÇÃO TOP-DOWN (TDIDT - Top-Down Induction of Decision Trees)
   A árvore é construída recursivamente, do topo (raiz) para as folhas,
   pela estratégia "dividir para conquistar" (divide and conquer):

       a. Se todas as instâncias do nó são da mesma classe (ou algum
          critério de parada é atingido) -> o nó vira uma FOLHA.
       b. Caso contrário, escolhe-se o "melhor" atributo para dividir o
          conjunto de dados e cria-se um nó de decisão para ele.
       c. Para cada valor (ou faixa) do atributo escolhido, particiona-se
          os dados e repete-se o processo recursivamente em cada partição.

   É um algoritmo GULOSO (greedy): a cada passo escolhe a divisão que
   parece melhor NAQUELE MOMENTO, sem reconsiderar escolhas já feitas
   (sem backtracking). Rápido, mas sujeito a ótimos locais.

2) CRITÉRIOS DE DIVISÃO (como escolher o "melhor" atributo?)
   A ideia é escolher o atributo que mais reduz a IMPUREZA (mistura de
   classes) dos subconjuntos gerados.

   a. ENTROPIA (impureza/incerteza, da Teoria da Informação):

          Entropia(S) = - Σ p_i * log2(p_i)

      onde p_i é a proporção de instâncias da classe i em S.
      - Entropia = 0 -> conjunto puro (uma só classe)
      - Entropia = 1 -> máxima incerteza (classes 50/50, caso binário)

   b. GANHO DE INFORMAÇÃO (usado no ID3 e C4.5): quanto a entropia cai
      ao dividir S pelo atributo A:

          Ganho(S, A) = Entropia(S) - Σ (|S_v| / |S|) * Entropia(S_v)

      onde S_v é o subconjunto de S em que o atributo A vale v.
      Escolhe-se o atributo com MAIOR ganho de informação.
      (Viés: o ganho de informação favorece atributos com muitos valores
      distintos; o C4.5 corrige isso normalizando pelo "Split Info", o
      que dá a Razão de Ganho / Gain Ratio.)

   c. ÍNDICE GINI (usado no CART; é o padrão do scikit-learn):

          Gini(S) = 1 - Σ p_i²

      Mede a probabilidade de classificar errado uma instância sorteada
      de S, se ela fosse rotulada aleatoriamente conforme a distribuição
      de classes de S. Também vale 0 quando o conjunto é puro. Escolhe-se
      o atributo que MINIMIZA a impureza Gini ponderada dos filhos. É
      mais barato de calcular que a entropia (sem logaritmo) e costuma
      gerar árvores parecidas na prática.

3) PRINCIPAIS ALGORITMOS DE INDUÇÃO
   - ID3  (Quinlan, 1986): atributos categóricos, usa ganho de
     informação, não poda, não trata valores faltantes.
   - C4.5 (Quinlan, 1993): sucessor do ID3; usa razão de ganho, trata
     atributos contínuos (por limiares) e valores faltantes, e faz
     PÓS-PODA.
   - CART (Breiman et al., 1984): gera árvores BINÁRIAS; usa índice Gini
     (classificação) ou soma dos erros quadráticos/variância (regressão).
     É o algoritmo implementado pelo scikit-learn.

4) OVERFITTING E PODA
   Uma árvore sem restrição de crescimento tende a se ajustar perfeitamente
   aos dados de treino (inclusive ao ruído), causando overfitting (alta
   variância, baixo viés). Duas estratégias de combate:

   - PRÉ-PODA (early stopping): impede o crescimento excessivo durante a
     própria indução, via critérios de parada como profundidade máxima
     (max_depth), número mínimo de instâncias por folha (min_samples_leaf)
     ou ganho mínimo exigido para dividir um nó.
   - PÓS-PODA: deixa a árvore crescer totalmente e depois remove
     sub-árvores que não melhoram a performance em validação (ex.:
     Reduced Error Pruning) ou que não compensam seu custo de complexidade
     (Cost-Complexity Pruning, usado no CART e exposto no scikit-learn
     pelo parâmetro `ccp_alpha`).

5) VANTAGENS x DESVANTAGENS
   (+) Fácil de interpretar e visualizar (modelo "caixa branca").
   (+) Não exige normalização/escalonamento dos dados.
   (+) Lida naturalmente com atributos numéricos e categóricos.
   (-) Instável: pequenas mudanças nos dados podem gerar árvores bem
       diferentes (alta variância).
   (-) Tende a overfitting se não for limitada/podada.
   (-) Algoritmo guloso -> não garante a árvore globalmente ótima.
   (-) Tende a favorecer a classe majoritária em datasets desbalanceados
       (relevante aqui: fraude é um evento raro).
   Observação: Random Forest e Gradient Boosting combinam várias árvores
   para reduzir essa instabilidade e o overfitting.
---------------------------------------------------------------------------
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_utils import get_train_test_split, load_raw_data, build_preprocessing_pipeline

# Dataset clássico "Jogar Tênis" (Quinlan / Mitchell), usado só para a
# demonstração manual dos cálculos
DATASET_TENIS = [
    ("Ensolarado", "Não"), ("Ensolarado", "Não"), ("Ensolarado", "Sim"),
    ("Ensolarado", "Sim"), ("Ensolarado", "Não"),
    ("Nublado", "Sim"), ("Nublado", "Sim"), ("Nublado", "Sim"), ("Nublado", "Sim"),
    ("Chuvoso", "Sim"), ("Chuvoso", "Sim"), ("Chuvoso", "Sim"),
    ("Chuvoso", "Não"), ("Chuvoso", "Não"),
]


def entropia(rotulos: list) -> float:
    """Entropia de Shannon de uma lista de rótulos: -Σ p_i * log2(p_i)."""
    total = len(rotulos)
    contagens = Counter(rotulos)
    valor = -sum((n / total) * math.log2(n / total) for n in contagens.values())
    return max(0.0, valor)  # evita "-0.0" quando o conjunto já é puro


def indice_gini(rotulos: list) -> float:
    """Índice de impureza Gini de uma lista de rótulos: 1 - Σ p_i²."""
    total = len(rotulos)
    contagens = Counter(rotulos)
    return 1 - sum((n / total) ** 2 for n in contagens.values())


def demonstracao_manual():
    """
    Reproduz à mão, passo a passo, a conta que uma árvore de decisão faz
    internamente para escolher o atributo de divisão em UM nó: o mesmo
    cálculo que o scikit-learn faz por baixo dos panos, aqui explícito.
    """
    print("=" * 78)
    print("DEMONSTRAÇÃO MANUAL: como o algoritmo escolhe uma divisão")
    print("=" * 78)

    rotulos_totais = [rotulo for _, rotulo in DATASET_TENIS]
    print(f"\nConjunto S ({len(rotulos_totais)} instâncias): {dict(Counter(rotulos_totais))}")

    h_pai = entropia(rotulos_totais)
    g_pai = indice_gini(rotulos_totais)
    print(f"Entropia(S) = {h_pai:.3f}")
    print(f"Gini(S)     = {g_pai:.3f}")

    print("\nCandidato a divisão: atributo 'Tempo'")
    valores = sorted(set(v for v, _ in DATASET_TENIS))
    h_filhos_ponderada = 0.0
    g_filhos_ponderada = 0.0
    for valor in valores:
        subconjunto = [rotulo for v, rotulo in DATASET_TENIS if v == valor]
        peso = len(subconjunto) / len(DATASET_TENIS)
        h_sub = entropia(subconjunto)
        g_sub = indice_gini(subconjunto)
        h_filhos_ponderada += peso * h_sub
        g_filhos_ponderada += peso * g_sub
        print(
            f"  Tempo={valor:<11} {dict(Counter(subconjunto))!s:<20} "
            f"|S_v|/|S|={peso:.3f}  Entropia={h_sub:.3f}  Gini={g_sub:.3f}"
        )

    ganho_informacao = h_pai - h_filhos_ponderada
    reducao_gini = g_pai - g_filhos_ponderada

    print(f"\nEntropia ponderada dos filhos    = {h_filhos_ponderada:.3f}")
    print(f"Ganho(S, Tempo) = Entropia(S) - Entropia_filhos = {ganho_informacao:.3f}")
    print(f"\nGini ponderado dos filhos         = {g_filhos_ponderada:.3f}")
    print(f"Redução de impureza Gini (CART)  = Gini(S) - Gini_filhos = {reducao_gini:.3f}")
    print(
        "\n-> O algoritmo repete essas contas para TODOS os atributos "
        "candidatos naquele nó e escolhe o de maior ganho de informação "
        "(ID3/C4.5) ou maior redução de Gini (CART). Em seguida, repete "
        "tudo recursivamente dentro de cada partição criada — é isso que "
        "torna a indução 'top-down' e 'gulosa'."
    )
    print("=" * 78 + "\n")


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print("=" * 78)
    print("DATASET DE FRAUDE: comparando os critérios de divisão gini x entropy")
    print("=" * 78)

    for criterio in ("gini", "entropy"):
        modelo = DecisionTreeClassifier(criterion=criterio, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\n--- criterion='{criterio}' (profundidade da árvore: {modelo.get_depth()}) ---")
        print(classification_report(y_test, y_pred, digits=4))

    print("=" * 78)
    print("PRÉ-PODA (max_depth): árvore mais simples, para inspecionar as regras")
    print("=" * 78)

    modelo_podado = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    modelo_podado.fit(X_train, y_train)
    y_pred_podado = modelo_podado.predict(X_test)

    print("\n--- max_depth=4 ---")
    print(classification_report(y_test, y_pred_podado, digits=4))
    print("\nRegras aprendidas (árvore limitada a profundidade 4):\n")
    print(export_text(modelo_podado, feature_names=list(X_train.columns)))


if __name__ == "__main__":
    main()
