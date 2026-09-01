"""
Análise de componentes principais (PCA) aplicada à redução de dimensionalidade.

Teoria completa (variância, matriz de covariância, autovalores e
autovetores, variância explicada, regra do cotovelo) está em
`notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com fichas de personagem de RPG (Força e
     Resistência), refazendo na mão a centralização dos dados, a matriz de
     covariância e a decomposição em autovalores/autovetores que o PCA faz
     escondido, pra sentir o algoritmo encontrando a direção de maior
     variância antes de ver isso rodando em cima de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.decomposition import PCA

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira: 8 personagens de RPG, cada um com dois atributos
# físicos (Força e Resistência) que costumam crescer juntos, quem é forte
# também aguenta pancada. Só pra "sentir" o PCA achando essa direção comum
# de variação na mão, antes de aplicar em cima de 30 atributos de verdade.
PERSONAGENS = ["Guerreiro", "Bárbaro", "Paladino", "Cavaleiro", "Ladino", "Arqueiro", "Mago", "Bardo"]
ATRIBUTOS_RPG = ["Força", "Resistência"]
DATASET_RPG = np.array(
    [
        [8, 9],
        [9, 8],
        [7, 8],
        [8, 7],
        [4, 5],
        [3, 4],
        [2, 3],
        [3, 3],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Matemática do PCA (centralização, covariância, autovalores/autovetores)
# ---------------------------------------------------------------------------


def centralizar(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtrai de cada coluna a própria média: devolve (X centralizado, vetor de médias)."""
    media = X.mean(axis=0)
    return X - media, media


def matriz_covariancia(X_centralizado: np.ndarray) -> np.ndarray:
    """Matriz de covariância de X (já centralizado): Cov = XᵀX / (n - 1)."""
    n = X_centralizado.shape[0]
    return (X_centralizado.T @ X_centralizado) / (n - 1)


def autovalores_autovetores(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Autovalores e autovetores de uma matriz de covariância (sempre simétrica),
    ordenados do maior autovalor pro menor: o autovetor de maior autovalor É
    o primeiro componente principal.
    """
    autovalores, autovetores = np.linalg.eigh(cov)
    ordem = np.argsort(autovalores)[::-1]
    return autovalores[ordem], autovetores[:, ordem]


def variancia_explicada(autovalores: np.ndarray) -> np.ndarray:
    """Fração da variância total capturada por cada componente (autovalor / soma dos autovalores)."""
    return autovalores / autovalores.sum()


def projetar(X_centralizado: np.ndarray, autovetores: np.ndarray, n_componentes: int) -> np.ndarray:
    """Projeta X (centralizado) nos `n_componentes` primeiros autovetores."""
    return X_centralizado @ autovetores[:, :n_componentes]


# ---------------------------------------------------------------------------
# Gráficos (salvos em images/, dentro da pasta deste script)
# ---------------------------------------------------------------------------


def _preparar_pyplot():
    """Configura o backend sem interface gráfica (necessário em servidor/terminal) e devolve o pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plotar_direcoes_principais(caminho_saida: Path | None = None) -> Path:
    """
    Desenha os 8 personagens no plano Força x Resistência, com os dois
    componentes principais como setas saindo do centro (a média): a seta
    maior (PC1) aponta pra direção onde os pontos mais se espalham, a
    menor (PC2) é perpendicular a ela e capta o resto, bem pouco, da
    variação.
    """
    plt = _preparar_pyplot()

    X_centralizado, media = centralizar(DATASET_RPG)
    cov = matriz_covariancia(X_centralizado)
    autovalores, autovetores = autovalores_autovetores(cov)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(DATASET_RPG[:, 0], DATASET_RPG[:, 1], s=80, color="#4C72B0", zorder=3)
    for nome, (x, y) in zip(PERSONAGENS, DATASET_RPG):
        ax.annotate(nome, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

    cores = ("#C44E52", "#55A868")
    rotulos = ("PC1", "PC2")
    # posição do texto como fração do vetor: mais curta pra PC1 (a seta
    # grande, cuja ponta cai perto do aglomerado Mago/Bardo) do que pra
    # PC2 (a seta pequena, que já sobra em espaço vazio), pra não sobrepor
    # os pontos.
    fracoes_texto = (0.78, 1.3)
    pontas_x, pontas_y = [], []
    for i in range(2):
        escala = np.sqrt(autovalores[i])
        dx, dy = autovetores[:, i] * escala
        ax.annotate(
            "",
            xy=(media[0] + dx, media[1] + dy),
            xytext=(media[0], media[1]),
            arrowprops=dict(arrowstyle="->", color=cores[i], linewidth=2.5),
        )
        ax.text(
            media[0] + dx * fracoes_texto[i], media[1] + dy * fracoes_texto[i], rotulos[i],
            color=cores[i], fontsize=11, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1),
        )
        pontas_x.append(media[0] + dx * 1.3)
        pontas_y.append(media[1] + dy * 1.3)

    # annotate() não entra no autoscale do eixo, então sem isso as setas
    # (em especial a de PC1, bem maior) ficam cortadas fora da área visível.
    todos_x = list(DATASET_RPG[:, 0]) + pontas_x
    todos_y = list(DATASET_RPG[:, 1]) + pontas_y
    margem_x = (max(todos_x) - min(todos_x)) * 0.15
    margem_y = (max(todos_y) - min(todos_y)) * 0.15
    ax.set_xlim(min(todos_x) - margem_x, max(todos_x) + margem_x)
    ax.set_ylim(min(todos_y) - margem_y, max(todos_y) + margem_y)

    ax.scatter(*media, color="black", marker="x", s=100, zorder=4, label="média")
    ax.set_xlabel(ATRIBUTOS_RPG[0])
    ax.set_ylabel(ATRIBUTOS_RPG[1])
    ax.set_title("Personagens de RPG: PC1 segue a diagonal 'físico geral'")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "direcoes_principais_rpg.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_variancia_explicada(
    autovalores: np.ndarray, titulo: str, caminho_saida: Path
) -> Path:
    """
    Gráfico de "cotovelo": barras com a variância explicada de cada
    componente e uma linha com a variância acumulada, pra visualizar em
    quantos componentes a curva acumulada já quase satura perto de 100%.
    """
    plt = _preparar_pyplot()

    fracoes = variancia_explicada(autovalores)
    acumulada = np.cumsum(fracoes)
    eixo_x = np.arange(1, len(fracoes) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(eixo_x, fracoes * 100, color="#4C72B0", label="variância explicada")
    ax1.set_xlabel("componente principal")
    ax1.set_ylabel("variância explicada (%)")
    ax1.set_xticks(eixo_x if len(eixo_x) <= 15 else eixo_x[:: max(1, len(eixo_x) // 15)])

    ax2 = ax1.twinx()
    ax2.plot(eixo_x, acumulada * 100, color="#C44E52", marker="o", markersize=3, label="acumulada")
    ax2.axhline(95, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("variância acumulada (%)")
    ax2.set_ylim(0, 105)

    linhas1, rotulos1 = ax1.get_legend_handles_labels()
    linhas2, rotulos2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc="center right")
    ax1.set_title(titulo)
    fig.tight_layout()

    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_projecao_1d(projecao_pc1: np.ndarray, caminho_saida: Path | None = None) -> Path:
    """
    Achata os 8 personagens numa única reta (o PC1), mostrando que a
    ordem ao longo dessa reta já separa sozinha quem é "tanque" (Força e
    Resistência altas) de quem é "esguio" (ambas baixas), com um número
    só em vez de dois.
    """
    plt = _preparar_pyplot()

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.scatter(projecao_pc1, np.zeros_like(projecao_pc1), s=80, color="#4C72B0", zorder=3)

    # Pontos próximos no PC1 (ex.: Guerreiro e Bárbaro) teriam o rótulo
    # sobreposto na mesma altura; alterna a altura do texto pra separar.
    ordem = np.argsort(projecao_pc1)
    alturas = (14, 30)
    for posicao, i in enumerate(ordem):
        ax.annotate(
            PERSONAGENS[i], (projecao_pc1[i], 0),
            textcoords="offset points", xytext=(0, alturas[posicao % 2]),
            ha="center", fontsize=9,
        )
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("PC1 (uma única coordenada resume Força + Resistência)")
    ax.set_title("Os 8 personagens, achatados na reta do primeiro componente")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "projecao_1d_rpg.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_projecao_2d_fraude(X_pca: np.ndarray, y, caminho_saida: Path | None = None) -> Path:
    """
    Espalha as transações do teste nos dois primeiros componentes
    principais, normais em cinza claro por baixo, fraudes em vermelho por
    cima (bem mais raras, por isso desenhadas depois e maiores), pra ver
    se sobra alguma separação visual mesmo o PCA nunca tendo olhado pro
    rótulo "fraude" durante o cálculo.
    """
    plt = _preparar_pyplot()

    normais = y == 0
    fraudes = y == 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        X_pca[normais, 0], X_pca[normais, 1],
        s=6, alpha=0.15, color="#8C8C8C", label=f"normal ({normais.sum()})",
    )
    ax.scatter(
        X_pca[fraudes, 0], X_pca[fraudes, 1],
        s=14, alpha=0.8, color="#C44E52", label=f"fraude ({fraudes.sum()})",
    )
    # Poucas transações bem extremas em PC1 (ex.: valores muito fora do
    # comum) espremeriam a nuvem inteira num canto se o eixo cobrisse todo
    # o intervalo; recorta a área visível pelos percentis 0,5-99,5% pra
    # focar onde a maioria dos pontos (normais e fraudes) realmente está.
    x_min, x_max = np.percentile(X_pca[:, 0], [0.5, 99.5])
    y_min, y_max = np.percentile(X_pca[:, 1], [0.5, 99.5])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Transações de teste projetadas em PC1 x PC2 (recortado nos percentis 0,5-99,5%)")
    ax.legend(markerscale=3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "projecao_2d_fraude.png"
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


def _mostrar_dados_originais():
    print("\n--- Ficha dos 8 personagens (Força, Resistência) ---")
    for nome, (forca, resistencia) in zip(PERSONAGENS, DATASET_RPG):
        print(f"  {nome:<10} Força={forca:.0f}  Resistência={resistencia:.0f}")


def _centralizar_e_mostrar() -> tuple[np.ndarray, np.ndarray]:
    print("\n--- Passo 1: centralizar (subtrair a média de cada coluna) ---")
    X_centralizado, media = centralizar(DATASET_RPG)
    print(f"Média do grupo: Força={media[0]:.3f}  Resistência={media[1]:.3f}")
    print(
        "PCA sempre trabalha em cima do desvio de cada ponto em relação à "
        "média, não do valor bruto: o que importa é o quanto cada "
        "personagem se afasta do 'personagem médio', não o valor absoluto."
    )
    for nome, (dx, dy) in zip(PERSONAGENS, X_centralizado):
        print(f"  {nome:<10} desvio Força={dx:+.3f}  desvio Resistência={dy:+.3f}")
    return X_centralizado, media


def _covariancia_e_mostrar(X_centralizado: np.ndarray) -> np.ndarray:
    print("\n--- Passo 2: matriz de covariância ---")
    cov = matriz_covariancia(X_centralizado)
    print(f"Cov(Força, Força)             = {cov[0, 0]:.3f}  (variância da Força sozinha)")
    print(f"Cov(Resistência, Resistência) = {cov[1, 1]:.3f}  (variância da Resistência sozinha)")
    print(
        f"Cov(Força, Resistência)       = {cov[0, 1]:.3f}  (positiva e grande: "
        "quem tem Força alta também tende a ter Resistência alta)"
    )
    print(
        "É exatamente essa covariância fora da diagonal que o PCA explora: "
        "se os dois atributos sobem e descem juntos, existe uma direção "
        "'diagonal' que resume os dois ao mesmo tempo, sem perder quase "
        "nada de informação."
    )
    return cov


def _autovalores_e_mostrar(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print("\n--- Passo 3: autovalores e autovetores da matriz de covariância ---")
    autovalores, autovetores = autovalores_autovetores(cov)
    for i, (autovalor, autovetor) in enumerate(zip(autovalores, autovetores.T), start=1):
        print(
            f"  PC{i}: autovalor={autovalor:.3f}  "
            f"direção=({autovetor[0]:+.3f}, {autovetor[1]:+.3f})"
        )
    print(
        "\nO autovetor é a DIREÇÃO (pra onde a seta aponta no plano Força x "
        "Resistência); o autovalor é o TAMANHO dessa seta, quanta variância "
        "sobra quando os dados são espremidos naquela direção. PC1 aponta "
        "quase na diagonal (Força e Resistência crescendo juntas); PC2 é "
        "perpendicular a PC1 e sobra bem pouca variância pra ele."
    )
    return autovalores, autovetores


def _variancia_explicada_e_mostrar(autovalores: np.ndarray):
    print("\n--- Passo 4: quanto cada componente explica da variância total ---")
    fracoes = variancia_explicada(autovalores)
    for i, fracao in enumerate(fracoes, start=1):
        print(f"  PC{i}: {fracao:.1%} da variância total")
    print(
        f"\nPC1 sozinho já explica {fracoes[0]:.1%}: trocar as duas colunas "
        "(Força, Resistência) por essa única coordenada perde muito pouca "
        "informação. É o cotovelo mais óbvio possível: 2 atributos bem "
        "correlacionados viram 1 sem dor."
    )


def _projetar_e_mostrar(X_centralizado: np.ndarray, autovetores: np.ndarray) -> np.ndarray:
    print("\n--- Passo 5: projetar cada personagem só no PC1 ---")
    projecao_pc1 = projetar(X_centralizado, autovetores, n_componentes=1)[:, 0]
    ordem = np.argsort(projecao_pc1)
    for i in ordem:
        print(f"  {PERSONAGENS[i]:<10} PC1={projecao_pc1[i]:+.3f}")
    print(
        "\nRepara na ordem: tanques (Guerreiro, Bárbaro, Paladino, "
        "Cavaleiro) ficam de um lado, magos e afins do outro, com só UM "
        "número por personagem em vez de dois."
    )
    return projecao_pc1


def demonstracao_manual():
    """
    Refaz à mão, com o exemplo de brincadeira dos 8 personagens de RPG, a
    conta que o PCA faz escondido: centralizar os dados, montar a matriz
    de covariância, extrair autovalores e autovetores, medir a variância
    explicada de cada componente e projetar os pontos no componente
    principal.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine uma mesa de RPG com 8 fichas de personagem, cada uma "
        "com dois atributos físicos (Força e Resistência). Os dois "
        "andam bem juntos: quem é forte também aguenta pancada. O PCA "
        "vai procurar a 'diagonal' que resume os dois atributos num só."
    )

    _mostrar_dados_originais()
    X_centralizado, _ = _centralizar_e_mostrar()
    cov = _covariancia_e_mostrar(X_centralizado)
    autovalores, autovetores = _autovalores_e_mostrar(cov)
    _variancia_explicada_e_mostrar(autovalores)
    projecao_pc1 = _projetar_e_mostrar(X_centralizado, autovetores)

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_direcoes_principais()
    plotar_variancia_explicada(
        autovalores, "Variância explicada: Força x Resistência (RPG)",
        IMAGES_DIR / "variancia_explicada_rpg.png",
    )
    plotar_projecao_1d(projecao_pc1)
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _aplicar_pca_real(X_train, X_test, limiar_variancia: float = 0.95):
    """
    Ajusta o PCA com TODOS os componentes possíveis (só pra medir a
    variância explicada de cada um), descobre quantos componentes bastam
    pra reter `limiar_variancia` da variância total, e devolve o treino e
    o teste já projetados nesse número reduzido de componentes.
    """
    pca_completo = PCA(n_components=None, random_state=42)
    pca_completo.fit(X_train)

    acumulada = np.cumsum(pca_completo.explained_variance_ratio_)
    n_componentes = int(np.searchsorted(acumulada, limiar_variancia) + 1)
    print(
        f"Com todos os {X_train.shape[1]} componentes disponíveis, "
        f"{n_componentes} já bastam pra reter {limiar_variancia:.0%} da "
        f"variância total (acumulada = {acumulada[n_componentes - 1]:.4f})."
    )

    pca_reduzido = PCA(n_components=n_componentes, random_state=42)
    X_train_pca = pca_reduzido.fit_transform(X_train)
    X_test_pca = pca_reduzido.transform(X_test)
    return pca_completo, pca_reduzido, X_train_pca, X_test_pca


def _mostrar_maiores_pesos_pc1(pca_reduzido: PCA, nomes_atributos):
    """Mostra os atributos originais com maior peso (em módulo) no primeiro componente."""
    pesos = pca_reduzido.components_[0]
    ordem = np.argsort(np.abs(pesos))[::-1][:5]
    print("\nAtributos originais com mais peso no PC1 (em módulo):")
    for i in ordem:
        print(f"  {nomes_atributos[i]:<8} peso={pesos[i]:+.3f}")


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    print(
        "\nCuriosidade sobre esse dataset: as colunas V1 a V28 já são o "
        "resultado de um PCA que o próprio Kaggle aplicou antes de "
        "publicar os dados, pra anonimizar as transações sem revelar os "
        "atributos originais do banco. Só 'Time' e 'Amount' escaparam "
        "dessa transformação. Aplicar PCA de novo aqui é redundante pra "
        "esconder informação (isso já foi feito), mas serve pra mostrar "
        "a técnica reduzindo 30 atributos a poucos, igual faria em "
        "qualquer outro dataset com atributos correlacionados."
    )

    pca_completo, pca_reduzido, X_train_pca, X_test_pca = _aplicar_pca_real(X_train, X_test)
    _mostrar_maiores_pesos_pc1(pca_reduzido, list(X_train.columns))

    plotar_variancia_explicada(
        pca_completo.explained_variance_, "Variância explicada: dataset de fraude (30 atributos)",
        IMAGES_DIR / "variancia_explicada_fraude.png",
    )
    plotar_projecao_2d_fraude(X_test_pca, y_test.to_numpy())


if __name__ == "__main__":
    main()
