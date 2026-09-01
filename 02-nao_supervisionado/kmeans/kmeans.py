"""
K-means aplicado ao dataset de detecção de fraude.

Teoria completa (algoritmo de Lloyd, escolha de K, inicialização
k-means++, mínimos locais, método do cotovelo) está em
`notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com dois quartéis-generais disputando
     território, refazendo na mão as rodadas de atribuir-e-recalcular
     que o k-means faz escondido, pra sentir o algoritmo convergindo
     antes de ver isso rodando em cima de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 10
# postos avançados espalhados num mapa (coordenadas x, y), metade num
# canto historicamente da Horda, metade num canto historicamente da
# Aliança, e dois portos neutros no meio, disputados pelas duas facções.
# O k-means não sabe de lore nenhuma: só vai ver 10 pontos num plano e
# tentar descobrir os territórios sozinho.
POSTOS = {
    "Orgrimmar": (1, 2),
    "Vale-de-Ferro": (2, 1),
    "Duotar": (1, 1),
    "Trono-do-Trovão": (3, 2),
    "Stormwind": (9, 8),
    "Forjaz": (8, 9),
    "Darnassus": (9, 9),
    "Exodar": (8, 7),
    "Rachai": (5, 5),
    "Baía-do-Butim": (6, 4),
}


# ---------------------------------------------------------------------------
# Algoritmo de Lloyd (atribuir -> recalcular -> repetir)
# ---------------------------------------------------------------------------


def distancia_euclidiana(a: tuple, b: tuple) -> float:
    """Distância euclidiana entre dois pontos: raiz da soma dos quadrados das diferenças."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _atribuir_postos(centroides: dict) -> dict:
    """Cada posto vai pro HQ (centroide) mais perto: a etapa de 'atribuição' do k-means."""
    atribuicoes = {hq: [] for hq in centroides}
    for posto, coordenada in POSTOS.items():
        hq_mais_perto = min(centroides, key=lambda hq: distancia_euclidiana(coordenada, centroides[hq]))
        atribuicoes[hq_mais_perto].append(posto)
    return atribuicoes


def _recalcular_centroides(centroides_atuais: dict, atribuicoes: dict) -> dict:
    """Move cada HQ pro centro de massa (média) dos postos que ele conquistou: a etapa de 'atualização'."""
    novos_centroides = {}
    for hq, postos_do_hq in atribuicoes.items():
        if not postos_do_hq:
            novos_centroides[hq] = centroides_atuais[hq]  # HQ sem território nenhum fica parado
            continue
        coordenadas = [POSTOS[posto] for posto in postos_do_hq]
        media_x = sum(c[0] for c in coordenadas) / len(coordenadas)
        media_y = sum(c[1] for c in coordenadas) / len(coordenadas)
        novos_centroides[hq] = (media_x, media_y)
    return novos_centroides


def _inercia(centroides: dict, atribuicoes: dict) -> float:
    """Soma das distâncias² de cada posto até o HQ que o conquistou: quanto menor, mais 'compacto' cada território."""
    return sum(
        distancia_euclidiana(POSTOS[posto], centroides[hq]) ** 2
        for hq, postos_do_hq in atribuicoes.items()
        for posto in postos_do_hq
    )


def rodar_kmeans_manual(centroides_iniciais: dict, max_iteracoes: int = 10) -> list:
    """
    Roda o algoritmo de Lloyd (atribuir -> recalcular -> repetir) até os
    territórios pararem de mudar, guardando cada rodada pra mostrar a
    convergência acontecendo de verdade.
    """
    centroides = dict(centroides_iniciais)
    historico = []
    for iteracao in range(1, max_iteracoes + 1):
        atribuicoes = _atribuir_postos(centroides)
        inercia = _inercia(centroides, atribuicoes)
        historico.append(
            {
                "iteracao": iteracao,
                "centroides": dict(centroides),
                "atribuicoes": atribuicoes,
                "inercia": inercia,
            }
        )

        novos_centroides = _recalcular_centroides(centroides, atribuicoes)
        if novos_centroides == centroides:
            break
        centroides = novos_centroides
    return historico


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: guerra de território
# ---------------------------------------------------------------------------


def _imprimir_historico_kmeans(historico: list):
    """Imprime, rodada por rodada, onde cada HQ está, quem ele conquistou e a inércia da rodada."""
    for passo in historico:
        print(f"\n--- Iteração {passo['iteracao']} ---")
        for hq, coordenada in passo["centroides"].items():
            print(f"  HQ '{hq}' está em ({coordenada[0]:.2f}, {coordenada[1]:.2f})")
        for hq, postos_do_hq in passo["atribuicoes"].items():
            print(f"    {hq} conquistou: {postos_do_hq}")
        print(f"  Inércia (soma das distâncias² até o HQ mais perto) = {passo['inercia']:.3f}")


def _demonstrar_convergencia_com_init_trocado() -> list:
    """
    Começa com os dois HQs em cantos TROCADOS de propósito (o HQ chamado
    'Aliança' nasce dentro do território Horda, e vice-versa), pra
    mostrar que o k-means não liga pro NOME que a gente deu ao centroide:
    ele só persegue a geografia dos dados, e a atribuição correta
    aparece de qualquer jeito depois de algumas rodadas.
    """
    print(
        "\n--- Rodando com os HQs iniciais nos cantos ERRADOS de propósito ---"
    )
    centroides_iniciais = {"Aliança": (0.0, 0.0), "Horda": (10.0, 10.0)}
    print(
        f"HQ inicial 'Aliança' em {centroides_iniciais['Aliança']} "
        "(literalmente em cima do território Horda)"
    )
    print(
        f"HQ inicial 'Horda' em {centroides_iniciais['Horda']} "
        "(literalmente em cima do território Aliança)"
    )

    historico = rodar_kmeans_manual(centroides_iniciais)
    _imprimir_historico_kmeans(historico)

    print(
        "\nO k-means não sabe que 'Aliança' devia ficar com Stormwind e "
        "companhia: ele só olha pra distância. Em pouquíssimas rodadas, "
        "o HQ chamado 'Aliança' (que nasceu perto da Horda) acaba "
        "conquistando território Horda mesmo, e vice-versa: os NOMES "
        "ficaram trocados, mas a PARTIÇÃO geográfica (quem fica com "
        "quem) converge certinha. Clustering não inventa rótulo, só "
        "agrupa; o rótulo humano ('isso aqui é a Horda') é posto por "
        "fora, depois."
    )
    return historico


def _demonstrar_sensibilidade_a_inicializacao():
    """
    Mostra o k-means (k=3) partindo de 3 HQs colados bem no meio do mapa,
    um cenário clássico de inicialização ruim, e compara a inércia final
    com a do k-means++ do scikit-learn (que testa várias inicializações e
    fica com a melhor). É o motivo de `n_init` existir.
    """
    print("\n--- k=3, agora comparando inicialização ruim x k-means++ ---")
    centroides_ruins = {"HQ-1": (5.0, 5.0), "HQ-2": (5.3, 5.0), "HQ-3": (5.0, 5.3)}
    print(f"Inicialização ruim: os 3 HQs nascem colados no meio do mapa: {centroides_ruins}")
    historico_ruim = rodar_kmeans_manual(centroides_ruins)
    inercia_ruim = historico_ruim[-1]["inercia"]
    print(f"Inércia final (inicialização ruim, colada): {inercia_ruim:.3f}")

    X = np.array(list(POSTOS.values()), dtype=float)
    modelo_bom = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42).fit(X)
    print(
        f"Inércia final (k-means++, 10 tentativas, fica com a melhor): "
        f"{modelo_bom.inertia_:.3f}"
    )

    if modelo_bom.inertia_ < inercia_ruim:
        print(
            "\nk-means++ achou uma partição mais compacta (inércia menor) "
            "do que a inicialização ruim. Isso é o próprio algoritmo de "
            "Lloyd fazendo o que promete (nunca piora a inércia rodada "
            "após rodada), só que ele só garante um MÍNIMO LOCAL: "
            "dependendo de onde os HQs nascem, ele pode ficar preso numa "
            "partição pior do que a melhor possível. Por isso o "
            "scikit-learn, por padrão, testa várias inicializações "
            "aleatórias (`n_init`) e fica só com a de menor inércia."
        )
    else:
        print(
            "\nNeste mapa em particular, até a inicialização ruim "
            "convergiu pra uma partição boa (a geografia aqui é fácil "
            "demais pra confundir o algoritmo). Isso não invalida o "
            "risco de mínimo local: com dados mais ambíguos ou mais "
            "clusters, a inicialização ruim pode sim ficar presa numa "
            "partição pior, e é exatamente por isso que o scikit-learn "
            "testa várias inicializações por padrão (`n_init`)."
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


def plotar_iteracoes_kmeans(historico: list, caminho_saida: Path | None = None) -> Path:
    """
    Desenha, lado a lado, uma iteração por painel: os postos coloridos
    pelo HQ que os conquistou naquela rodada, e o HQ (X grande) migrando
    pro centro do próprio território. Mostra a convergência acontecendo.
    """
    plt = _preparar_pyplot()
    cores = {"Aliança": "#4C72B0", "Horda": "#C44E52"}

    n = len(historico)
    fig, eixos = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    eixos = [eixos] if n == 1 else eixos
    for ax, passo in zip(eixos, historico):
        for hq, postos_do_hq in passo["atribuicoes"].items():
            cor = cores.get(hq, "#55A868")
            for posto in postos_do_hq:
                x, y = POSTOS[posto]
                ax.scatter(x, y, color=cor, s=90, edgecolor="black", zorder=3)
            cx, cy = passo["centroides"][hq]
            ax.scatter(cx, cy, color=cor, marker="X", s=280, edgecolor="black", linewidth=2, zorder=4)
        ax.set_title(f"Iteração {passo['iteracao']}\ninércia = {passo['inercia']:.2f}")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.grid(alpha=0.3)
    fig.suptitle("Convergência do k-means: HQs migrando pro centro do próprio território")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "convergencia_iteracoes.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_fronteira_territorios(centroides_finais: dict, caminho_saida: Path | None = None) -> Path:
    """
    Desenha a fronteira de Voronoi entre os HQs finais: toda coordenada
    do mapa pintada pela cor do HQ mais próximo dali. É literalmente a
    fronteira de decisão do k-means (mesma ideia da fronteira de decisão
    do k-NN, só que os "vizinhos" aqui são só os K centroides).
    """
    plt = _preparar_pyplot()

    nomes_hq = list(centroides_finais)
    cores = {"Aliança": "#4C72B0", "Horda": "#C44E52"}

    xx, yy = np.meshgrid(np.linspace(-1, 11, 300), np.linspace(-1, 11, 300))
    zz = np.array(
        [
            nomes_hq.index(min(nomes_hq, key=lambda hq: distancia_euclidiana((x, y), centroides_finais[hq])))
            for x, y in zip(xx.ravel(), yy.ravel())
        ]
    ).reshape(xx.shape)

    niveis = [i - 0.5 for i in range(len(nomes_hq) + 1)]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz, levels=niveis, colors=[cores.get(nome, "#55A868") for nome in nomes_hq], alpha=0.25)
    for posto, (x, y) in POSTOS.items():
        ax.scatter(x, y, color="black", s=40, zorder=3)
        ax.annotate(posto, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    for hq, (cx, cy) in centroides_finais.items():
        ax.scatter(cx, cy, color=cores.get(hq, "#55A868"), marker="X", s=320, edgecolor="black", linewidth=2, zorder=4, label=hq)
    ax.set_title("Território final: fronteira de Voronoi entre os HQs")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "fronteira_territorios.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_curva_cotovelo(inercia_por_k: dict, caminho_saida: Path | None = None) -> Path:
    """Plota inércia x K: o 'cotovelo' é onde a queda da inércia desacelera, sinal de K bom o bastante."""
    plt = _preparar_pyplot()

    ks = list(inercia_por_k)
    inercias = list(inercia_por_k.values())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, inercias, marker="o")
    ax.set_xlabel("k (número de clusters)")
    ax.set_ylabel("inércia (soma das distâncias² até o centroide)")
    ax.set_title("Método do cotovelo: onde a queda da inércia desacelera")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "curva_cotovelo.png"
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
    Refaz à mão, com o exemplo de brincadeira dos postos avançados, a
    conta que o k-means faz escondida por trás do `KMeans` do
    scikit-learn: atribuir cada ponto ao centroide mais perto, recalcular
    o centroide como a média dos pontos atribuídos, e repetir até parar
    de mudar.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine um mapa com 10 postos avançados e duas facções "
        "brigando por território. O k-means recebe só as coordenadas dos "
        "postos e um número K de facções pra encontrar (aqui, K=2): ele "
        "chuta duas posições de quartel-general (HQ), manda cada posto "
        "se filiar ao HQ mais perto, muda o HQ pro centro de gravidade "
        "de quem se filiou a ele, e repete essa dança até ninguém mais "
        "trocar de facção."
    )

    historico = _demonstrar_convergencia_com_init_trocado()
    _demonstrar_sensibilidade_a_inicializacao()

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_iteracoes_kmeans(historico)
    plotar_fronteira_territorios(historico[-1]["centroides"])
    print("=" * 78 + "\n")
    return historico


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _calcular_inercia_por_k(X, ks=range(1, 9)) -> dict:
    """Treina KMeans pra cada K da faixa e devolve {k: inércia}, a matéria-prima do método do cotovelo."""
    inercia_por_k = {}
    for k in ks:
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        modelo.fit(X)
        inercia_por_k[k] = modelo.inertia_
        print(f"  k={k}: inércia = {modelo.inertia_:.1f}")
    return inercia_por_k


def _treinar_k_final(X, y_real, k: int = 2):
    """Treina o KMeans final com K escolhido e compara com a classe real via ARI, silhueta e crosstab."""
    modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    rotulos = modelo.fit_predict(X)

    silhueta = silhouette_score(X, rotulos, sample_size=5000, random_state=42)
    ari = adjusted_rand_score(y_real, rotulos)
    print(f"\nk={k}: silhueta (amostrada) = {silhueta:.4f}   ARI (vs. classe real) = {ari:.4f}")

    tabela = pd.crosstab(rotulos, y_real, rownames=["cluster"], colnames=["classe real"])
    print("\nCruzamento cluster descoberto x classe real:")
    print(tabela)
    return modelo, rotulos


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, _, y_train, _ = get_train_test_split(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    print("\nMétodo do cotovelo, K de 1 a 8, treinado no conjunto de treino inteiro:")
    inercia_por_k = _calcular_inercia_por_k(X_train)
    plotar_curva_cotovelo(inercia_por_k)

    _titulo("K=2: o clustering encontra sozinho a fronteira fraude/normal?")
    _treinar_k_final(X_train, y_train, k=2)


if __name__ == "__main__":
    main()
