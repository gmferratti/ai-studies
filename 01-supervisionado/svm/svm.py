"""
SVM (Support Vector Machine) aplicado ao dataset de detecção de fraude.

Teoria completa (maximização da margem, margem suave, truque do kernel,
problema dual, Teoria do Aprendizado Estatístico) está em
`notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com dois reinos rivais, calculando na mão o
     hiperplano de margem máxima a partir de dois vetores de suporte,
     depois mostrando o efeito de um "infiltrado" (margem suave) e de um
     XOR de brinquedo (truque do kernel), pra sentir o algoritmo
     escolhendo a fronteira antes de ver isso rodando em cima de dados de
     verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.svm import SVC, LinearSVC

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: dois
# reinos rivais posicionados num mapa 2D. Ragnar (Norte) e Aurelia (Sul)
# são as tropas da linha de frente (vetores de suporte); os demais são
# retaguarda, só pra mostrar que mover eles não muda a fronteira.
DATASET_EXERCITO = [
    # (nome, x, y, reino)
    ("Ragnar", -1, -1, "Norte"),
    ("Ymir", -3, -2, "Norte"),
    ("Frost", -2, -4, "Norte"),
    ("Aurelia", 1, 1, "Sul"),
    ("Solenne", 3, 2, "Sul"),
    ("Helios", 2, 4, "Sul"),
]

# Um espião do Norte disfarçado bem no meio do território do Sul: viola a
# margem rígida e força a escolha entre distorcer a fronteira (C alto) ou
# tolerar o erro em troca de uma margem mais larga (C baixo).
INFILTRADO = ("Espiao Corvo", 1, 0.7, "Norte")

# XOR de brinquedo: dois postos do Norte nos cantos (0,0) e (1,1), dois do
# Sul nos cantos (0,1) e (1,0). Nenhuma reta separa essas duas classes,
# não importa a inclinação: é o exemplo clássico de fronteira não linear.
DATASET_XOR = [
    ("Posto A", 0, 0, "Norte"),
    ("Posto B", 1, 1, "Norte"),
    ("Posto C", 0, 1, "Sul"),
    ("Posto D", 1, 0, "Sul"),
]

ROTULO_NUMERICO = {"Norte": -1, "Sul": 1}


# ---------------------------------------------------------------------------
# Matemática do hiperplano de margem máxima
# ---------------------------------------------------------------------------


def _vetor(linha: tuple) -> tuple:
    """Extrai (x, y) de uma linha de DATASET_EXERCITO, INFILTRADO ou DATASET_XOR."""
    return (linha[1], linha[2])


def hiperplano_por_dois_pontos(sv_negativo: tuple, sv_positivo: tuple) -> tuple:
    """
    Calcula (w, b) do hiperplano de margem máxima quando só existem dois
    vetores de suporte, um de cada classe: w = 2d / ||d||^2, com
    d = sv_positivo - sv_negativo, e b = -w . ponto_medio. É a única
    situação em que o SVM tem solução fechada, sem precisar de otimização
    numérica: com mais pontos candidatos a vetor de suporte, o problema
    vira a otimização convexa que o scikit-learn resolve por baixo dos
    panos.
    """
    dx = sv_positivo[0] - sv_negativo[0]
    dy = sv_positivo[1] - sv_negativo[1]
    norma_d_quadrado = dx**2 + dy**2
    k = 2 / norma_d_quadrado
    w = (k * dx, k * dy)

    ponto_medio = ((sv_positivo[0] + sv_negativo[0]) / 2, (sv_positivo[1] + sv_negativo[1]) / 2)
    b = -(w[0] * ponto_medio[0] + w[1] * ponto_medio[1])
    return w, b


def margem(w: tuple) -> float:
    """Largura da margem: 2 / ||w||."""
    return 2 / math.sqrt(w[0] ** 2 + w[1] ** 2)


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: fronteira entre dois reinos (margem rígida)
# ---------------------------------------------------------------------------


def _explicar_hiperplano_a_mao():
    """Calcula o hiperplano de margem máxima na mão e confere com o SVC do scikit-learn."""
    print("\n--- Calculando a fronteira na mão, só com as duas tropas da linha de frente ---")
    ragnar = _vetor(DATASET_EXERCITO[0])
    aurelia = _vetor(DATASET_EXERCITO[3])
    print(f"Ragnar (Norte, vetor de suporte): {ragnar}")
    print(f"Aurelia (Sul, vetor de suporte):  {aurelia}")

    w, b = hiperplano_por_dois_pontos(ragnar, aurelia)
    print(f"\nw = 2*(Aurelia - Ragnar) / ||Aurelia - Ragnar||^2 = ({w[0]:.3f}, {w[1]:.3f})")
    print(f"b = -w . ponto_medio = {b:.3f}")
    print(f"Margem = 2 / ||w|| = {margem(w):.3f}")

    X = [_vetor(linha) for linha in DATASET_EXERCITO]
    y = [ROTULO_NUMERICO[linha[3]] for linha in DATASET_EXERCITO]
    modelo = SVC(kernel="linear", C=1e6).fit(X, y)
    w_sklearn = modelo.coef_[0]
    print(
        f"\nConferindo com o SVC do scikit-learn: w = ({w_sklearn[0]:.3f}, "
        f"{w_sklearn[1]:.3f}), b = {modelo.intercept_[0]:.3f}. Bateu."
    )
    nomes_suporte = [DATASET_EXERCITO[i][0] for i in modelo.support_]
    print(f"Vetores de suporte encontrados pelo scikit-learn: {nomes_suporte}")


def _mostrar_retaguarda_nao_importa():
    """Move duas tropas da retaguarda bem longe e mostra que a fronteira não muda."""
    print("\n--- A retaguarda pode andar à vontade que a fronteira não muda ---")
    X = [_vetor(linha) for linha in DATASET_EXERCITO]
    y = [ROTULO_NUMERICO[linha[3]] for linha in DATASET_EXERCITO]
    modelo_original = SVC(kernel="linear", C=1e6).fit(X, y)

    X_movido = list(X)
    X_movido[1] = (-10, -8)  # Ymir marcha bem mais fundo no território Norte
    X_movido[4] = (8, 9)  # Solenne marcha bem mais fundo no território Sul
    modelo_movido = SVC(kernel="linear", C=1e6).fit(X_movido, y)

    w1, w2 = modelo_original.coef_[0], modelo_movido.coef_[0]
    print(f"w antes de mover a retaguarda:  ({w1[0]:.3f}, {w1[1]:.3f})")
    print(f"w depois de mover a retaguarda: ({w2[0]:.3f}, {w2[1]:.3f})")
    print(
        "Ymir foi de (-3,-2) pra (-10,-8) e Solenne de (3,2) pra (8,9), bem "
        "mais fundo no próprio território, e w não mudou nadinha: só quem "
        "está na linha de frente decide a fronteira."
    )


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: margem suave e o espião infiltrado
# ---------------------------------------------------------------------------


def _demonstrar_margem_suave():
    """Adiciona o infiltrado e compara C alto (quase margem rígida) com C baixo (margem suave)."""
    print("\n--- O Espião Corvo se infiltra no meio do território do Sul ---")
    X = [_vetor(linha) for linha in DATASET_EXERCITO] + [_vetor(INFILTRADO)]
    y = [ROTULO_NUMERICO[linha[3]] for linha in DATASET_EXERCITO] + [ROTULO_NUMERICO[INFILTRADO[3]]]
    nomes = [linha[0] for linha in DATASET_EXERCITO] + [INFILTRADO[0]]
    print(f"{INFILTRADO[0]} (Norte) está em {_vetor(INFILTRADO)}, cercado de tropas do Sul.")

    for C, apelido in ((1e6, "quase rígida, tenta acertar todo mundo"), (1.0, "suave, tolera o infiltrado")):
        modelo = SVC(kernel="linear", C=C).fit(X, y)
        w, b = modelo.coef_[0], modelo.intercept_[0]
        previsao_infiltrado = modelo.predict([_vetor(INFILTRADO)])[0]
        veredito = "acertou (Norte)" if previsao_infiltrado == -1 else "errou, achou que era Sul"
        suportes = [nomes[i] for i in modelo.support_]
        print(f"\nC={C:g} (margem {apelido}):")
        print(f"  w=({w[0]:.3f}, {w[1]:.3f})  margem={margem(w):.3f}")
        print(f"  vetores de suporte: {suportes}")
        print(f"  previsão pro {INFILTRADO[0]}: {veredito}")

    print(
        "\nCom C bem alto, a fronteira se contorce pra acertar até o "
        "infiltrado, e a margem encolhe bastante. Com C baixo, o modelo "
        "prefere errar o infiltrado a estreitar a margem: mais barato "
        "pagar a folga de um único exemplo teimoso do que distorcer a "
        "fronteira inteira por causa dele."
    )


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: o truque do kernel no XOR
# ---------------------------------------------------------------------------


def _demonstrar_truque_do_kernel():
    """Mostra o kernel linear falhando no XOR e o kernel RBF separando certinho."""
    print("\n--- Testando o XOR: nenhuma reta separa Norte de Sul aqui ---")
    X = [_vetor(linha) for linha in DATASET_XOR]
    y = [ROTULO_NUMERICO[linha[3]] for linha in DATASET_XOR]

    for kernel in ("linear", "rbf"):
        modelo = SVC(kernel=kernel, C=10, gamma="scale").fit(X, y)
        previsoes = [int(p) for p in modelo.predict(X)]
        acertos = sum(p == real for p, real in zip(previsoes, y))
        print(f"kernel={kernel:<7} previsões={previsoes}  esperado={y}  acertos={acertos}/{len(y)}")

    print(
        "\nO kernel linear erra a metade: como os dois Norte estão em "
        "cantos opostos (0,0) e (1,1), e os dois Sul também estão em "
        "cantos opostos (0,1) e (1,0), qualquer reta deixa pelo menos um "
        "Norte e um Sul do mesmo lado. O kernel RBF, calculando "
        "similaridade em vez de posição crua, consegue 'levantar' o "
        "problema pra uma forma onde uma fronteira curva separa os dois "
        "grupos perfeitamente."
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


def _desenhar_fronteira_e_margens(ax, w, b, eixo_x, cor="#333333"):
    """Desenha a fronteira (w.x+b=0) e as duas margens (w.x+b=+-1) num eixo já existente."""
    for deslocamento, estilo, largura in ((0, "-", 1.6), (1, "--", 1.0), (-1, "--", 1.0)):
        eixo_y = (deslocamento - b - w[0] * eixo_x) / w[1]
        ax.plot(eixo_x, eixo_y, linestyle=estilo, color=cor, linewidth=largura)


def plotar_fronteira_exercito(caminho_saida: Path | None = None) -> Path:
    """
    Desenha o mapa de batalha com os dois reinos, a fronteira de margem
    máxima calculada a partir de Ragnar e Aurelia, as duas margens em
    volta dela, e destaca os vetores de suporte com estrela.
    """
    plt = _preparar_pyplot()
    import numpy as np

    w, b = hiperplano_por_dois_pontos(_vetor(DATASET_EXERCITO[0]), _vetor(DATASET_EXERCITO[3]))
    cores = {"Norte": "#4C72B0", "Sul": "#C0392B"}
    suportes = {"Ragnar", "Aurelia"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for nome, x, y, reino in DATASET_EXERCITO:
        eh_suporte = nome in suportes
        ax.scatter(
            x, y, color=cores[reino], s=280 if eh_suporte else 140,
            marker="*" if eh_suporte else "o", edgecolor="black", zorder=3,
        )
        ax.annotate(nome, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)

    _desenhar_fronteira_e_margens(ax, w, b, np.linspace(-5, 5, 200))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("posição x no mapa")
    ax.set_ylabel("posição y no mapa")
    ax.set_title("Fronteira de margem máxima entre Norte e Sul\n(estrelas = vetores de suporte)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "fronteira_exercito.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_margem_suave(caminho_saida: Path | None = None) -> Path:
    """
    Compara, lado a lado, a fronteira com C alto (tenta acertar até o
    infiltrado, margem estreita) e C baixo (tolera o infiltrado errado,
    margem larga).
    """
    plt = _preparar_pyplot()
    import numpy as np

    X = [_vetor(linha) for linha in DATASET_EXERCITO] + [_vetor(INFILTRADO)]
    y = [ROTULO_NUMERICO[linha[3]] for linha in DATASET_EXERCITO] + [ROTULO_NUMERICO[INFILTRADO[3]]]
    cores = {-1: "#4C72B0", 1: "#C0392B"}
    eixo_x = np.linspace(-5, 5, 200)

    fig, eixos = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    configuracoes = ((1e6, "C alto: quase margem rígida"), (1.0, "C baixo: margem suave"))
    for ax, (C, titulo) in zip(eixos, configuracoes):
        modelo = SVC(kernel="linear", C=C).fit(X, y)
        w, b = modelo.coef_[0], modelo.intercept_[0]
        for i, (px, py) in enumerate(X):
            eh_suporte = i in modelo.support_
            ax.scatter(
                px, py, color=cores[y[i]], s=280 if eh_suporte else 140,
                marker="*" if eh_suporte else "o", edgecolor="black", zorder=3,
            )
        _desenhar_fronteira_e_margens(ax, w, b, eixo_x)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(f"{titulo}\nmargem = {margem(w):.2f}")
        ax.set_xlabel("posição x")
    eixos[0].set_ylabel("posição y")
    fig.suptitle("Margem suave: o preço de insistir em acertar o infiltrado")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "margem_suave_infiltrado.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_kernel_trick(caminho_saida: Path | None = None) -> Path:
    """
    Desenha a região de decisão do SVM no XOR de brinquedo, com kernel
    linear (falha, corta o mapa ao meio) e kernel RBF (separa certinho),
    lado a lado.
    """
    plt = _preparar_pyplot()
    import numpy as np

    X = np.array([_vetor(linha) for linha in DATASET_XOR], dtype=float)
    y = np.array([ROTULO_NUMERICO[linha[3]] for linha in DATASET_XOR])
    cores = {-1: "#4C72B0", 1: "#C0392B"}

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grade = np.c_[xx.ravel(), yy.ravel()]

    fig, eixos = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, kernel in zip(eixos, ("linear", "rbf")):
        modelo = SVC(kernel=kernel, C=10, gamma="scale").fit(X, y)
        if kernel == "linear":
            # Nesse XOR de 4 pontos o kernel linear não acha direção nenhuma
            # que ajude: a solução ótima é w=0, prever sempre a mesma classe.
            # w.x+b calculado direto evita o ruído numérico que o
            # decision_function() do SVC mostra perto dessa solução degenerada.
            zz = grade @ modelo.coef_[0] + modelo.intercept_[0]
        else:
            zz = modelo.decision_function(grade)
        zz = zz.reshape(xx.shape)
        limite = max(abs(zz.min()), abs(zz.max()), 1e-6)
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", vmin=-limite, vmax=limite, alpha=0.6)
        ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=2)
        for classe, cor in cores.items():
            pontos = X[y == classe]
            ax.scatter(pontos[:, 0], pontos[:, 1], color=cor, s=140, edgecolor="black", zorder=3)
        ax.set_title(f"kernel={kernel}")
        if kernel == "linear":
            ax.set_xlabel("x\n(w ≈ (0, 0): sem direção que separe, prevê sempre a mesma classe)")
        else:
            ax.set_xlabel("x")
    eixos[0].set_ylabel("y")
    fig.suptitle("Truque do kernel: reta não separa o XOR, curva separa")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "kernel_trick_xor.png"
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
    Refaz à mão, com o exemplo de brincadeira dos dois reinos, a lógica
    que o SVM aplica escondida atrás do SVC do scikit-learn: hiperplano de
    margem máxima, por que só a linha de frente importa, o trade-off da
    margem suave, e o truque do kernel num XOR que nenhuma reta resolve.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine dois reinos rivais, Norte e Sul, com tropas espalhadas "
        "num mapa. Alguém precisa desenhar a fronteira mais segura "
        "possível entre eles: a que fica o mais longe possível das tropas "
        "mais avançadas dos dois lados ao mesmo tempo."
    )

    _explicar_hiperplano_a_mao()
    _mostrar_retaguarda_nao_importa()
    _demonstrar_margem_suave()
    _demonstrar_truque_do_kernel()

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_fronteira_exercito()
    plotar_margem_suave()
    plotar_kernel_trick()
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _metricas_fraude(y_test, y_pred) -> tuple:
    """Precisão, recall e F1 da classe fraude (rótulo 1)."""
    return (
        precision_score(y_test, y_pred, pos_label=1),
        recall_score(y_test, y_pred, pos_label=1),
        f1_score(y_test, y_pred, pos_label=1),
    )


def _treinar_linear_svc_completo(X_train, X_test, y_train, y_test) -> tuple:
    """Treina LinearSVC (kernel linear, via liblinear) no treino completo e imprime o desempenho."""
    inicio = time.time()
    modelo = LinearSVC(C=1.0, max_iter=5000)
    modelo.fit(X_train, y_train)
    tempo = time.time() - inicio

    y_pred = modelo.predict(X_test)
    print(f"\n--- LinearSVC, treino completo ({len(X_train)} exemplos), {tempo:.2f}s ---")
    print(classification_report(y_test, y_pred, digits=4))
    return (tempo, *_metricas_fraude(y_test, y_pred))


def _amostra_estratificada(X_train, y_train, n_normais: int, random_state: int = 42):
    """Monta uma amostra com todas as fraudes do treino mais n_normais transações normais."""
    indices_fraude = y_train[y_train == 1].index
    indices_normais = y_train[y_train == 0].sample(n=n_normais, random_state=random_state).index
    idx = indices_fraude.union(indices_normais)
    return X_train.loc[idx], y_train.loc[idx]


def _treinar_svc_rbf_amostra(X_train, X_test, y_train, y_test, n_normais: int = 6000) -> tuple:
    """Treina SVC com kernel RBF numa amostra estratificada (todas as fraudes + n_normais normais)."""
    Xs, ys = _amostra_estratificada(X_train, y_train, n_normais)
    inicio = time.time()
    modelo = SVC(kernel="rbf", C=1.0)
    modelo.fit(Xs, ys)
    tempo = time.time() - inicio

    y_pred = modelo.predict(X_test)
    print(
        f"\n--- SVC kernel RBF, amostra de {len(Xs)} exemplos "
        f"({(ys == 1).sum()} fraudes + {(ys == 0).sum()} normais), {tempo:.2f}s ---"
    )
    print(classification_report(y_test, y_pred, digits=4))
    return (tempo, *_metricas_fraude(y_test, y_pred))


def _comparar_custo_computacional(X_train, y_train):
    """Treina SVC RBF em duas amostras de tamanhos bem diferentes, só pra sentir o custo crescendo mais rápido que os dados."""
    print("\n--- Quanto o tempo de treino cresce junto com o tamanho da amostra? ---")
    for n_normais in (6000, 40000):
        Xs, ys = _amostra_estratificada(X_train, y_train, n_normais)
        inicio = time.time()
        SVC(kernel="rbf", C=1.0).fit(Xs, ys)
        tempo = time.time() - inicio
        print(f"  Amostra de {len(Xs)} exemplos: {tempo:.2f}s de treino")

    print(
        "\nA amostra maior tem uns 6x mais exemplos que a menor, mas o "
        "tempo de treino não cresce só 6x: é o custo entre O(n²) e O(n³) "
        "do problema de otimização do SVM aparecendo na prática. Com o "
        "dataset de fraude inteiro (mais de 200 mil exemplos de treino), "
        "um kernel não linear como o RBF levaria tempo demais pra caber "
        "numa demonstração; por isso a Parte 3 treina o RBF só numa "
        "amostra, e usa o LinearSVC, que escala bem melhor, no treino "
        "completo."
    )


def plotar_comparacao_linear_vs_rbf(resultado_linear: tuple, resultado_rbf: tuple, caminho_saida: Path | None = None) -> Path:
    """
    Compara precisão, recall e F1 (classe fraude) entre o LinearSVC no
    treino completo e o SVC RBF numa amostra, anotando o tempo de treino
    de cada um.
    """
    plt = _preparar_pyplot()
    import numpy as np

    tempo_linear, *metricas_linear = resultado_linear
    tempo_rbf, *metricas_rbf = resultado_rbf
    rotulos = ["Precisão", "Recall", "F1"]
    posicoes = np.arange(len(rotulos))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(posicoes - largura / 2, metricas_linear, largura, label=f"LinearSVC ({tempo_linear:.2f}s, treino completo)", color="#4C72B0")
    ax.bar(posicoes + largura / 2, metricas_rbf, largura, label=f"SVC RBF ({tempo_rbf:.2f}s, amostra)", color="#C0392B")
    ax.set_xticks(posicoes)
    ax.set_xticklabels(rotulos)
    ax.set_ylim(0, 1)
    ax.set_title("Classe fraude: LinearSVC (treino completo) x SVC RBF (amostra)")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "comparacao_linear_vs_rbf.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    resultado_linear = _treinar_linear_svc_completo(X_train, X_test, y_train, y_test)

    _titulo("KERNEL RBF NUMA AMOSTRA: vale o custo extra?")
    resultado_rbf = _treinar_svc_rbf_amostra(X_train, X_test, y_train, y_test)
    plotar_comparacao_linear_vs_rbf(resultado_linear, resultado_rbf)

    _titulo("CUSTO COMPUTACIONAL: o preço de crescer o kernel RBF")
    _comparar_custo_computacional(X_train, y_train)


if __name__ == "__main__":
    main()
