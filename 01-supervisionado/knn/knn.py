"""
k-NN (k-vizinhos mais próximos) aplicado ao dataset de detecção de fraude.

Teoria completa (lazy learning, métricas de distância, escolha do K, voto
ponderado, normalização, maldição da dimensionalidade) está em
`notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com aventureiros de RPG, refazendo na mão a
     conta de distância e votação que o k-NN faz escondido, pra sentir o
     algoritmo escolhendo os vizinhos antes de ver isso rodando em cima
     de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 10
# aventureiros de RPG cadastrados numa guilda, com Força, Agilidade e Ouro
# carregado, marcados como Guerreiro ou Mago. "Ouro" entra só na parte 6 da
# demonstração (normalização); nas partes 1 a 5 usa-se só Força e Agilidade.
DATASET_AVENTUREIROS = [
    # (nome, forca, agilidade, ouro, classe)
    ("Thorin", 9, 3, 390, "Guerreiro"),
    ("Brunhilda", 8, 4, 500, "Guerreiro"),
    ("Grommash", 7, 2, 80, "Guerreiro"),
    ("Ragnar", 8, 5, 300, "Guerreiro"),
    ("Conan", 6, 3, 200, "Guerreiro"),
    ("Elowen", 2, 8, 150, "Mago"),
    ("Zephyra", 3, 9, 600, "Mago"),
    ("Mystral", 1, 7, 90, "Mago"),
    ("Sylvana", 3, 7, 250, "Mago"),
    ("Nimue", 2, 9, 700, "Mago"),
]

# O aventureiro novo, sem classe, que a guilda quer classificar.
KAEL_NOME = "Kael"
KAEL_FORCA_AGILIDADE = (5, 6)
KAEL_OURO = 400


# ---------------------------------------------------------------------------
# Matemática de distância e votação
# ---------------------------------------------------------------------------


def distancia_euclidiana(a: tuple, b: tuple) -> float:
    """Distância euclidiana entre dois pontos: raiz da soma dos quadrados das diferenças."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _vetor_stats(linha: tuple) -> tuple:
    """Extrai (força, agilidade) de uma linha do DATASET_AVENTUREIROS."""
    return (linha[1], linha[2])


def _vetor_com_ouro(linha: tuple) -> tuple:
    """Extrai (força, agilidade, ouro) de uma linha do DATASET_AVENTUREIROS."""
    return (linha[1], linha[2], linha[3])


def calcular_vizinhos_ordenados(consulta: tuple, vetor_fn=_vetor_stats) -> list:
    """
    Calcula a distância da consulta até cada aventureiro cadastrado
    (usando os atributos escolhidos por vetor_fn) e devolve a lista
    ordenada do mais perto pro mais longe: [(nome, classe, distância), ...]
    """
    vizinhos = [
        (linha[0], linha[4], distancia_euclidiana(consulta, vetor_fn(linha)))
        for linha in DATASET_AVENTUREIROS
    ]
    return sorted(vizinhos, key=lambda v: v[2])


def votar(vizinhos_k: list, ponderado: bool = False) -> tuple:
    """
    Vota a classe vencedora entre os vizinhos_k [(nome, classe, distância), ...].
    Voto simples: cada vizinho vale 1. Voto ponderado: cada vizinho vale
    1/distância (quem está mais perto pesa mais). Devolve (classe
    vencedora ou None se empatar, contagem/peso por classe).
    """
    votos = Counter()
    for _, classe, distancia in vizinhos_k:
        peso = 1.0 if not ponderado else (1.0 / distancia if distancia > 0 else float("inf"))
        votos[classe] += peso

    maior_voto = max(votos.values())
    vencedoras = [classe for classe, voto in votos.items() if voto == maior_voto]
    vencedora = vencedoras[0] if len(vencedoras) == 1 else None
    return vencedora, dict(votos)


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: classificando o Kael
# ---------------------------------------------------------------------------


def _imprimir_tabela_distancias(vizinhos: list):
    """Imprime a lista de aventureiros ordenada por distância até o Kael."""
    for i, (nome, classe, distancia) in enumerate(vizinhos, start=1):
        print(f"  {i:>2}º mais perto: {nome:<10} ({classe:<9}) distância = {distancia:.3f}")


def _explicar_distancia_euclidiana():
    """Aquecimento: mostra a conta de Pitágoras por trás da distância euclidiana com um par concreto."""
    print("\n--- Aquecimento: de onde vem a distância euclidiana ---")
    thorin = DATASET_AVENTUREIROS[0]
    a, b = _vetor_stats(thorin), KAEL_FORCA_AGILIDADE
    dx, dy = a[0] - b[0], a[1] - b[1]
    print(
        f"Thorin tem (Força={a[0]}, Agilidade={a[1]}); Kael tem "
        f"(Força={b[0]}, Agilidade={b[1]})."
    )
    print(f"Diferença em Força = {a[0]}-{b[0]} = {dx}; diferença em Agilidade = {a[1]}-{b[1]} = {dy}.")
    print(
        f"Distância = sqrt(({dx})² + ({dy})²) = sqrt({dx**2} + {dy**2}) = "
        f"{distancia_euclidiana(a, b):.3f}  (é literalmente o teorema de Pitágoras)"
    )


def _classificar_kael_por_varios_k():
    """
    Classifica o Kael usando K = 1, 3, 5, 6 e 7, mostrando que a resposta
    MUDA conforme K muda, e que K=6 (par) empata a votação simples.
    """
    print(f"\n--- Distância de {KAEL_NOME} (Força={KAEL_FORCA_AGILIDADE[0]}, "
          f"Agilidade={KAEL_FORCA_AGILIDADE[1]}) até cada aventureiro cadastrado ---")
    vizinhos = calcular_vizinhos_ordenados(KAEL_FORCA_AGILIDADE)
    _imprimir_tabela_distancias(vizinhos)

    print("\nVotando com K diferentes (voto simples, 1 vizinho = 1 voto):")
    for k in (1, 3, 5, 6, 7):
        vencedora, votos = votar(vizinhos[:k], ponderado=False)
        resultado = vencedora if vencedora else "EMPATE"
        print(f"  K={k}: {k} vizinhos mais próximos, votos = {votos}  -> previsão: {resultado}")

    print(
        "\nReparem: com K=1 a resposta é Mago (só olha o vizinho colado), com "
        "K=3 e K=5 vira Guerreiro, com K=6 EMPATA 3 a 3 (K par é arriscado "
        "em problema de 2 classes), e com K=7 volta a ser Mago. O valor de "
        "K não é detalhe: ele muda a resposta final."
    )
    return vizinhos


def _explicar_voto_ponderado(vizinhos: list):
    """Mostra o voto ponderado por distância desempatando o K=6, que empatou no voto simples."""
    print("\n--- Desempatando o K=6 com voto ponderado por distância (peso = 1/distância) ---")
    vizinhos_k6 = vizinhos[:6]
    for nome, classe, distancia in vizinhos_k6:
        print(f"  {nome:<10} ({classe:<9}) distância={distancia:.3f}  peso=1/d={1/distancia:.4f}")

    vencedora, pesos = votar(vizinhos_k6, ponderado=True)
    print("\nSoma de peso por classe:")
    for classe, peso in pesos.items():
        print(f"  {classe:<9}: {peso:.4f}")
    print(
        f"\nVoto simples empatava 3 a 3, mas no voto ponderado {vencedora} "
        "vence, porque o vizinho MAIS próximo de todos (Sylvana) pesa mais "
        "do que qualquer vizinho isolado do outro lado."
    )


def _explicar_normalizacao():
    """
    Mostra o Ouro (escala de centenas) atropelando Força e Agilidade
    (escala de 1 a 10) na conta da distância, e como isso muda o vizinho
    mais próximo do Kael de forma que não faz sentido pelas estatísticas.
    """
    print("\n--- Por que precisa normalizar: adicionando o atributo Ouro (sem escalonar) ---")
    consulta = (*KAEL_FORCA_AGILIDADE, KAEL_OURO)
    vizinhos_sem_ouro = calcular_vizinhos_ordenados(KAEL_FORCA_AGILIDADE, _vetor_stats)
    vizinhos_com_ouro = calcular_vizinhos_ordenados(consulta, _vetor_com_ouro)

    print(f"Sem Ouro na conta, o vizinho mais próximo de {KAEL_NOME} era: "
          f"{vizinhos_sem_ouro[0][0]} ({vizinhos_sem_ouro[0][1]}), distância = {vizinhos_sem_ouro[0][2]:.3f}")
    print(f"Adicionando Ouro (na escala de centenas, sem normalizar), o mais próximo passa a ser: "
          f"{vizinhos_com_ouro[0][0]} ({vizinhos_com_ouro[0][1]}), distância = {vizinhos_com_ouro[0][2]:.3f}")
    print(
        "\nThorin tem Ouro=390, bem perto do Ouro=400 do Kael, e isso sozinho "
        "já decide a distância toda, mesmo Thorin tendo Força e Agilidade bem "
        "diferentes do Kael. O atributo de escala maior atropelou os outros "
        "dois. A solução é normalizar (Min-Max ou Z-score) ANTES de calcular "
        "distância, o que o pipeline de dados de verdade já faz com o "
        "StandardScaler (utils/data_utils.py)."
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


def _cores_por_classe() -> dict:
    """Paleta compartilhada entre os gráficos do exemplo de brincadeira."""
    return {"Guerreiro": "#4C72B0", "Mago": "#DD8452"}


def plotar_dispersao_aventureiros(caminho_saida: Path | None = None) -> Path:
    """
    Espalha os 10 aventureiros no plano Força x Agilidade, coloridos por
    classe, com o Kael marcado como estrela: visualiza por que ele fica
    numa região ambígua, na fronteira entre os dois grupos.
    """
    plt = _preparar_pyplot()
    cores = _cores_por_classe()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for classe, cor in cores.items():
        pontos = [linha for linha in DATASET_AVENTUREIROS if linha[4] == classe]
        ax.scatter(
            [p[1] for p in pontos], [p[2] for p in pontos],
            color=cor, s=90, label=classe, edgecolor="black", zorder=3,
        )
        for nome, forca, agilidade, _, _ in pontos:
            ax.annotate(nome, (forca, agilidade), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.scatter(*KAEL_FORCA_AGILIDADE, color="#2ECC71", marker="*", s=350,
               edgecolor="black", label=KAEL_NOME, zorder=4)
    ax.annotate(KAEL_NOME, KAEL_FORCA_AGILIDADE, textcoords="offset points", xytext=(8, -12), fontsize=9, weight="bold")

    ax.set_xlabel("Força")
    ax.set_ylabel("Agilidade")
    ax.set_title("Guilda dos aventureiros: onde o Kael cai no mapa")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "dispersao_aventureiros.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_fronteira_decisao(caminho_saida: Path | None = None) -> Path:
    """
    Desenha a fronteira de decisão do k-NN treinado no toy dataset (Força
    x Agilidade), para K=1, K=3 e K=7 lado a lado: mostra a fronteira
    ficando mais recortada (overfitting) com K pequeno e mais suave
    (underfitting) com K grande.
    """
    plt = _preparar_pyplot()
    import numpy as np

    X = [_vetor_stats(linha) for linha in DATASET_AVENTUREIROS]
    y = [linha[4] for linha in DATASET_AVENTUREIROS]
    cores = _cores_por_classe()

    xx, yy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
    grade = np.c_[xx.ravel(), yy.ravel()]

    fig, eixos = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, k in zip(eixos, (1, 3, 7)):
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X, y)
        previsao = modelo.predict(grade)
        rotulo_para_numero = {"Guerreiro": 0, "Mago": 1}
        zz = np.array([rotulo_para_numero[c] for c in previsao]).reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=[cores["Guerreiro"], cores["Mago"]], alpha=0.25)
        for classe, cor in cores.items():
            pontos = [linha for linha in DATASET_AVENTUREIROS if linha[4] == classe]
            ax.scatter([p[1] for p in pontos], [p[2] for p in pontos], color=cor, edgecolor="black", s=70, zorder=3)
        ax.scatter(*KAEL_FORCA_AGILIDADE, color="#2ECC71", marker="*", s=250, edgecolor="black", zorder=4)
        ax.set_title(f"K={k}")
        ax.set_xlabel("Força")
    eixos[0].set_ylabel("Agilidade")
    fig.suptitle("Fronteira de decisão do k-NN: K pequeno recorta, K grande suaviza")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "fronteira_decisao.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_efeito_normalizacao(caminho_saida: Path | None = None) -> Path:
    """
    Compara, num gráfico de barras, a distância de Kael até Thorin e até
    Sylvana com e sem o atributo Ouro na conta: visualiza como a escala
    grande do Ouro inverte quem é o vizinho mais próximo.
    """
    plt = _preparar_pyplot()

    thorin = next(l for l in DATASET_AVENTUREIROS if l[0] == "Thorin")
    sylvana = next(l for l in DATASET_AVENTUREIROS if l[0] == "Sylvana")
    consulta_com_ouro = (*KAEL_FORCA_AGILIDADE, KAEL_OURO)

    dist_sem_ouro = {
        "Thorin": distancia_euclidiana(_vetor_stats(thorin), KAEL_FORCA_AGILIDADE),
        "Sylvana": distancia_euclidiana(_vetor_stats(sylvana), KAEL_FORCA_AGILIDADE),
    }
    dist_com_ouro = {
        "Thorin": distancia_euclidiana(_vetor_com_ouro(thorin), consulta_com_ouro),
        "Sylvana": distancia_euclidiana(_vetor_com_ouro(sylvana), consulta_com_ouro),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=False)
    for ax, dados, titulo in (
        (ax1, dist_sem_ouro, "Sem Ouro (só Força e Agilidade)"),
        (ax2, dist_com_ouro, "Com Ouro, sem normalizar"),
    ):
        ax.bar(dados.keys(), dados.values(), color=["#4C72B0", "#DD8452"])
        ax.set_title(titulo)
        ax.set_ylabel("distância até o Kael")
        for i, valor in enumerate(dados.values()):
            ax.text(i, valor, f"{valor:.1f}", ha="center", va="bottom")

    fig.suptitle("Um atributo de escala grande (Ouro) atropela a distância")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "efeito_normalizacao.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_projecao_pca(X_train, y_train, caminho_saida: Path | None = None) -> Path:
    """
    Projeta uma amostra do treino de verdade (30 atributos) em 2 dimensões
    via PCA, pra dar uma intuição visual de como as classes fraude/normal
    se distribuem nesse mapa de "vizinhança" reduzido.
    """
    plt = _preparar_pyplot()
    from sklearn.decomposition import PCA

    indices_fraude = y_train[y_train == 1].index
    indices_normais = y_train[y_train == 0].sample(n=3000, random_state=42).index
    amostra = X_train.loc[indices_fraude.union(indices_normais)]
    classes_amostra = y_train.loc[amostra.index]

    coordenadas = PCA(n_components=2, random_state=42).fit_transform(amostra)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        coordenadas[classes_amostra == 0, 0], coordenadas[classes_amostra == 0, 1],
        color="#4C72B0", s=10, alpha=0.5, label="normal",
    )
    ax.scatter(
        coordenadas[classes_amostra == 1, 0], coordenadas[classes_amostra == 1, 1],
        color="#DD8452", s=25, label="fraude", zorder=3,
    )
    ax.set_xlabel("componente principal 1")
    ax.set_ylabel("componente principal 2")
    ax.set_title("Treino projetado em 2D (PCA): tudo que o k-NN\nprecisa separar em 30 dimensões, aqui achatado em 2")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "projecao_pca_fraude.png"
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


def demonstracao_manual():
    """
    Refaz à mão, com o exemplo de brincadeira dos aventureiros, a conta
    que o k-NN faz escondida por trás do `KNeighborsClassifier` do
    scikit-learn: medir distância até todo mundo, ordenar, votar, e por
    que K e a normalização mudam o resultado.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine que você é o taverneiro de uma guilda com 10 "
        "aventureiros cadastrados (Guerreiros e Magos) e chega um novato, "
        "Kael, sem ficha de classe. Cada aventureiro cadastrado já tem "
        "Força e Agilidade medidas, que é o que o k-NN usa pra achar quem "
        "mais se parece com o Kael."
    )

    _explicar_distancia_euclidiana()
    vizinhos = _classificar_kael_por_varios_k()
    _explicar_voto_ponderado(vizinhos)
    _explicar_normalizacao()

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_dispersao_aventureiros()
    plotar_fronteira_decisao()
    plotar_efeito_normalizacao()
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _comparar_k(X_train, X_test, y_train, y_test):
    """Treina um KNeighborsClassifier com K=3 e K=11 (voto simples) e imprime o desempenho."""
    for k in (3, 11):
        modelo = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\n--- K={k}, weights='uniform' ---")
        print(classification_report(y_test, y_pred, digits=4))


def _treinar_com_peso_por_distancia(X_train, X_test, y_train, y_test):
    """Treina com K=11 e voto ponderado por distância, pra ver se ajuda numa classe rara (fraude)."""
    modelo = KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("\n--- K=11, weights='distance' ---")
    print(classification_report(y_test, y_pred, digits=4))
    plotar_projecao_pca(X_train, y_train)


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    _comparar_k(X_train, X_test, y_train, y_test)

    _titulo("VOTO PONDERADO POR DISTÂNCIA: ajuda a classe rara (fraude) a ganhar votação?")
    _treinar_com_peso_por_distancia(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
