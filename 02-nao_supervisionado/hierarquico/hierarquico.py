"""
Clustering hierárquico aglomerativo aplicado ao dataset de detecção de fraude.

Teoria completa (linkage, dendrograma, corte da árvore, complexidade
computacional) está em `notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com guerreiros de Dragon Ball se fundindo,
     refazendo na mão a matriz de distâncias e as fusões sucessivas que o
     clustering hierárquico faz escondido, pra sentir o dendrograma
     nascendo antes de ver isso rodando em cima de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import math
import sys
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from utils.data_utils import build_preprocessing_pipeline, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 6
# guerreiros com Poder de luta e Ki, formando 3 duplas naturais (saiyajins,
# namekuseijins, humanos). A ideia é ver o clustering hierárquico
# redescobrir essas 3 "espécies" sozinho, sempre fundindo primeiro a dupla
# mais parecida, como uma fusão à la Dragon Ball (Goku + Vegeta = um
# guerreiro só, com a força dos dois).
GUERREIROS = {
    "Goku": (8, 9),
    "Vegeta": (9, 8),
    "Piccolo": (2, 7),
    "Nail": (3, 5),
    "Krillin": (1, 1),
    "Yamcha": (2, 3),
}


# ---------------------------------------------------------------------------
# Matemática de distância e fusão (linkage)
# ---------------------------------------------------------------------------


def distancia_euclidiana(a: tuple, b: tuple) -> float:
    """Distância euclidiana entre dois pontos: raiz da soma dos quadrados das diferenças."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def distancia_entre_clusters(membros_a: list, membros_b: list) -> float:
    """
    Distância de ligação simples (single linkage) entre dois clusters: a
    MENOR distância entre qualquer par de membros, um de cada lado. É
    como perguntar "qual é a dupla mais parecida entre os dois grupos?"
    e deixar só essa dupla decidir se os grupos se fundem.
    """
    return min(
        distancia_euclidiana(GUERREIROS[a], GUERREIROS[b])
        for a in membros_a
        for b in membros_b
    )


def _par_de_clusters_mais_proximo(clusters: dict) -> tuple:
    """Acha, entre todos os pares de clusters atuais, o par com menor distância de ligação simples."""
    candidatos = [
        (nome_a, nome_b, distancia_entre_clusters(membros_a, membros_b))
        for (nome_a, membros_a), (nome_b, membros_b) in combinations(clusters.items(), 2)
    ]
    return min(candidatos, key=lambda candidato: candidato[2])


def construir_historico_fusoes(nomes: list) -> list:
    """
    Roda o algoritmo aglomerativo até sobrar um cluster só: a cada passo,
    funde o par de clusters mais parecido e guarda um retrato de como a
    partição ficou. É exatamente essa sequência de fusões, com a altura
    (distância) de cada uma, que vira o dendrograma.
    """
    clusters = {nome: [nome] for nome in nomes}
    historico = []
    while len(clusters) > 1:
        nome_a, nome_b, distancia = _par_de_clusters_mais_proximo(clusters)
        membros_fundidos = clusters.pop(nome_a) + clusters.pop(nome_b)
        novo_nome = f"({nome_a}+{nome_b})"
        clusters[novo_nome] = membros_fundidos
        historico.append(
            {
                "fundidos": (nome_a, nome_b),
                "distancia": distancia,
                "n_clusters_restantes": len(clusters),
                "particao_atual": {nome: list(membros) for nome, membros in clusters.items()},
            }
        )
    return historico


def cortar_em_k_clusters(historico: list, k: int) -> dict:
    """
    "Corta" o dendrograma na altura certa pra sobrar exatamente k
    clusters: como o número de clusters cai de 1 em 1 a cada fusão, basta
    achar o primeiro passo do histórico em que sobraram k clusters.
    """
    for passo in historico:
        if passo["n_clusters_restantes"] == k:
            return passo["particao_atual"]
    raise ValueError(f"Nunca existiu uma partição com exatamente {k} clusters neste histórico.")


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: fusão dos guerreiros
# ---------------------------------------------------------------------------


def _imprimir_matriz_distancias(nomes: list):
    """Imprime a distância entre cada dupla de guerreiros, o ponto de partida do algoritmo."""
    print("\n--- Distância entre cada dupla de guerreiros (cada um começa como seu próprio cluster) ---")
    for a, b in combinations(nomes, 2):
        distancia = distancia_euclidiana(GUERREIROS[a], GUERREIROS[b])
        print(f"  {a:<8} <-> {b:<8}: {distancia:.3f}")


def _imprimir_historico_fusoes(historico: list):
    """Imprime, uma por uma, as fusões escolhidas pelo algoritmo e como a partição vai encolhendo."""
    print("\n--- Fusões, sempre o par mais parecido primeiro ---")
    for passo in historico:
        nome_a, nome_b = passo["fundidos"]
        print(f"\nFusão: {nome_a} + {nome_b}  (distância = {passo['distancia']:.3f})")
        print(f"  Restam {passo['n_clusters_restantes']} cluster(s):")
        for nome, membros in passo["particao_atual"].items():
            print(f"    {nome}: {membros}")


def _imprimir_cortes(historico: list):
    """Mostra a mesma árvore cortada em k=3, k=2 e k=1, revelando espécies, depois times, depois um só."""
    print("\n--- Cortando a árvore em diferentes alturas ---")
    for k in (3, 2, 1):
        particao = cortar_em_k_clusters(historico, k)
        print(f"\nCorte em {k} cluster(s):")
        for nome, membros in particao.items():
            print(f"  {nome}: {membros}")
    print(
        "\nRepara na ordem: primeiro reaparecem as 3 espécies originais "
        "(saiyajins, namekuseijins, humanos), cada uma tendo se fundido "
        "internamente primeiro. Depois namekuseijins e humanos se juntam "
        "num único time (os mais fracos se unindo pra sobreviver), e só "
        "na última fusão de todas os saiyajins orgulhosos entram na "
        "roda, formando o guerreiro-fusão definitivo com todo mundo "
        "junto. Ninguém disse ao algoritmo quem era saiyajin ou "
        "namekuseijin: essa estrutura em espécie nasceu só de quão perto "
        "cada um estava dos outros no gráfico de Poder x Ki."
    )


# ---------------------------------------------------------------------------
# Gráficos (salvos em images/, dentro da pasta deste script)
# ---------------------------------------------------------------------------


def _preparar_pyplot():
    """Configura o backend sem interface gráfica (necessário em servidor/terminal) e devolve o pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plotar_dendrograma_toy(nomes: list, caminho_saida: Path | None = None) -> Path:
    """
    Desenha o dendrograma de verdade (via scipy) dos 6 guerreiros, usando
    o mesmo linkage simples (single) da conta manual: confirma visualmente
    que as fusões calculadas na mão batem com a árvore "oficial".
    """
    plt = _preparar_pyplot()
    from scipy.cluster.hierarchy import dendrogram, linkage

    pontos = [GUERREIROS[nome] for nome in nomes]
    matriz_linkage = linkage(pontos, method="single")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    dendrogram(matriz_linkage, labels=nomes, ax=ax)
    ax.set_ylabel("distância da fusão (altura)")
    ax.set_title("Dendrograma dos guerreiros: quem se fundiu com quem, e em que altura")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "dendrograma_guerreiros.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Dendrograma salvo em: {caminho_saida}")
    return caminho_saida


def plotar_mapa_fusoes(historico: list, caminho_saida: Path | None = None) -> Path:
    """
    Espalha os 6 guerreiros no plano Poder x Ki, coloridos pelo corte em 3
    clusters: visualiza as 3 "espécies" que o algoritmo achou sozinho.
    """
    plt = _preparar_pyplot()

    particao_3 = cortar_em_k_clusters(historico, 3)
    cores = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for cor, membros in zip(cores, particao_3.values()):
        for guerreiro in membros:
            x, y = GUERREIROS[guerreiro]
            ax.scatter(x, y, color=cor, s=180, edgecolor="black", zorder=3)
            ax.annotate(guerreiro, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Poder de luta")
    ax.set_ylabel("Ki")
    ax.set_title("Corte em 3 clusters: as 3 'espécies' reencontradas sem ninguém dizer o rótulo")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "mapa_fusoes.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_dendrograma_fraude(X, caminho_saida: Path | None = None) -> Path:
    """
    Dendrograma (truncado nos últimos 30 nós, senão vira um borrão
    ilegível) da amostra de fraude, usando linkage de Ward.
    """
    plt = _preparar_pyplot()
    from scipy.cluster.hierarchy import dendrogram, linkage

    matriz_linkage = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(matriz_linkage, truncate_mode="lastp", p=30, show_leaf_counts=True, ax=ax)
    ax.set_ylabel("distância da fusão (altura)")
    ax.set_xlabel("cluster (número entre parênteses = quantas transações agrupadas ali)")
    ax.set_title("Dendrograma da amostra de fraude (linkage de Ward, truncado nos últimos 30 nós)")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "dendrograma_fraude.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Dendrograma salvo em: {caminho_saida}")
    return caminho_saida


def plotar_pca_cluster_vs_classe(X, y_real, rotulos_cluster, caminho_saida: Path | None = None) -> Path:
    """
    Projeta a amostra de fraude em 2D via PCA, lado a lado: à esquerda
    colorida pelo cluster que o algoritmo descobriu sozinho, à direita
    pela classe real (fraude ou não). Mostra visualmente se as duas
    colorações batem ou não.
    """
    plt = _preparar_pyplot()
    from sklearn.decomposition import PCA

    coordenadas = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    ax1.scatter(coordenadas[:, 0], coordenadas[:, 1], c=rotulos_cluster, cmap="coolwarm", s=15)
    ax1.set_title("Cor = cluster descoberto sozinho")
    ax1.set_xlabel("componente principal 1")
    ax1.set_ylabel("componente principal 2")

    ax2.scatter(coordenadas[:, 0], coordenadas[:, 1], c=y_real, cmap="coolwarm", s=15)
    ax2.set_title("Cor = classe real (fraude ou não)")
    ax2.set_xlabel("componente principal 1")

    fig.suptitle("Clustering hierárquico bate com a fraude de verdade?")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "pca_cluster_vs_classe.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


# ---------------------------------------------------------------------------
# Demonstração manual (Parte 1 e 2)
# ---------------------------------------------------------------------------


def _titulo(texto: str):
    """Imprime um cabeçalho de seção padronizado no terminal."""
    print("=" * 78)
    print(texto)
    print("=" * 78)


def demonstracao_manual() -> list:
    """
    Refaz à mão, com o exemplo de brincadeira dos guerreiros, a conta que
    o clustering hierárquico faz escondida por trás do
    `AgglomerativeClustering` do scikit-learn: medir distância entre
    todo mundo, fundir sempre a dupla mais parecida, e repetir até
    sobrar um cluster só.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine 6 guerreiros espalhados num campo de batalha, cada um "
        "com Poder de luta e Ki medidos, sem ninguém contar de qual "
        "espécie cada um é (saiyajin, namekuseijin ou humano). O "
        "clustering hierárquico não sabe quantos grupos existem de "
        "antemão: ele só vai fundindo, passo a passo, a dupla mais "
        "parecida do momento, igual uma fusão de Dragon Ball, até "
        "sobrar um guerreiro-fusão só com todo mundo dentro. A árvore "
        "de fusões que sobra no caminho é o dendrograma."
    )

    nomes = list(GUERREIROS)
    _imprimir_matriz_distancias(nomes)
    historico = construir_historico_fusoes(nomes)
    _imprimir_historico_fusoes(historico)
    _imprimir_cortes(historico)

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_dendrograma_toy(nomes)
    plotar_mapa_fusoes(historico)
    print("=" * 78 + "\n")
    return historico


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _preparar_amostra_fraude(df: pd.DataFrame, n_normais: int = 508):
    """
    Clustering hierárquico é O(n²) em memória e tempo, porque guarda a
    distância entre CADA par de pontos: rodar nas ~285 mil linhas do
    dataset inteiro é inviável (285000² pares não cabe em memória
    nenhuma). Por isso usa-se uma amostra pequena, com TODAS as fraudes
    (só 492 no dataset inteiro) mais uma amostra aleatória de transações
    normais: uma amostra proporcional (0,17% de fraude) teria só 1 ou 2
    fraudes em mil linhas, pouco pra ver algum cluster de fraude se
    separar de verdade.
    """
    fraudes = df[df["Class"] == 1]
    normais = df[df["Class"] == 0].sample(n=n_normais, random_state=42)
    amostra = pd.concat([fraudes, normais]).sample(frac=1, random_state=42)
    return amostra.drop(columns="Class"), amostra["Class"]


def _comparar_linkages(X, y_real):
    """Treina AgglomerativeClustering com cada tipo de linkage e compara ARI (contra a classe real) e silhueta."""
    for linkage in ("ward", "average", "complete", "single"):
        modelo = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        rotulos = modelo.fit_predict(X)
        ari = adjusted_rand_score(y_real, rotulos)
        silhueta = silhouette_score(X, rotulos)
        print(f"  linkage={linkage:<9} ARI (vs. classe real) = {ari:.4f}   silhueta = {silhueta:.4f}")


def _imprimir_crosstab(y_real, rotulos_cluster):
    """Cruza o cluster descoberto com a classe real, pra ver se os clusters 'sabem' separar fraude."""
    tabela = pd.crosstab(rotulos_cluster, y_real, rownames=["cluster"], colnames=["classe real"])
    print("\nCruzamento cluster descoberto x classe real:")
    print(tabela)


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X, y = _preparar_amostra_fraude(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    print(
        f"\nAmostra de {len(X)} transações ({(y == 1).sum()} fraudes + "
        f"{(y == 0).sum()} normais), pelo motivo explicado acima. "
        "Comparando os quatro tipos de linkage:"
    )
    _comparar_linkages(X, y)

    _titulo("LINKAGE ESCOLHIDO: Ward, dendrograma e comparação com a classe real")
    modelo = AgglomerativeClustering(n_clusters=2, linkage="ward")
    rotulos = modelo.fit_predict(X)
    _imprimir_crosstab(y, rotulos)
    plotar_dendrograma_fraude(X)
    plotar_pca_cluster_vs_classe(X, y, rotulos)


if __name__ == "__main__":
    main()
