"""
Random Forest aplicado ao dataset de detecção de fraude.

Teoria completa (bagging + sorteio de atributos por divisão, correlação
entre árvores, o trade-off de `max_features`, importância de atributos)
está em `notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com um torneio de sobrevivência de 16
     participantes, onde um atributo é campeão disparado: mostra bagging
     puro (árvores quase idênticas, todas usando o mesmo atributo
     primeiro) contra random forest (árvores mais variadas entre si,
     porque às vezes o atributo campeão nem está disponível pra
     escolher), pra sentir a decorrelação antes de ver isso rodando em
     cima de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito,
     comparando `max_features` e olhando a importância dos atributos.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 16
# sobreviventes de uma batalha royale numa floresta, com três atributos
# (armadura, agilidade, sorte). "Armadura" é um atributo campeão
# disparado: sozinho já separa os 16 sem nenhum erro. "Agilidade" tem
# sinal de verdade, mas nitidamente mais fraco (três sobreviventes fogem
# do padrão, de propósito, pra ela nunca competir de igual pra igual com
# armadura). "Sorte" é fraca, mal ajuda a separar os dois grupos.
SOBREVIVENTES = [
    # (nome, armadura, agilidade, sorte, sobrevive)
    ("Aki", 20, 60, 30, "Não"),  # agilidade alta, foge do padrão de propósito
    ("Bo", 25, 35, 50, "Não"),
    ("Cass", 30, 20, 70, "Não"),
    ("Dex", 35, 65, 20, "Não"),  # agilidade alta, foge do padrão de propósito
    ("Eryn", 40, 45, 60, "Não"),
    ("Fen", 45, 30, 40, "Não"),
    ("Groh", 48, 50, 55, "Não"),
    ("Hoku", 55, 40, 35, "Sim"),  # agilidade baixa, foge do padrão de propósito
    ("Ilse", 60, 60, 65, "Sim"),
    ("Jax", 65, 70, 30, "Sim"),
    ("Kira", 70, 65, 75, "Sim"),
    ("Lior", 75, 75, 45, "Sim"),
    ("Mox", 80, 60, 20, "Sim"),
    ("Nyra", 85, 80, 80, "Sim"),
    ("Oz", 90, 70, 50, "Sim"),
    ("Pia", 95, 85, 40, "Sim"),
]

# Recruta novo: armadura baixa (diria "Não" com confiança) mas agilidade
# alta (diria "Sim" com confiança), o ponto exato onde árvores enraizadas
# em armadura e árvores enraizadas em agilidade discordam de verdade.
NOVO_RECRUTA = ("Recruta", 25, 75, 50)

ROTULO_NUMERICO = {"Não": 0, "Sim": 1}
NOMES_ATRIBUTOS = ["armadura", "agilidade", "sorte"]
B_EXIBICAO = 9


def _X_y():
    """Extrai X (armadura, agilidade, sorte) e y (0/1) de SOBREVIVENTES."""
    X = [(linha[1], linha[2], linha[3]) for linha in SOBREVIVENTES]
    y = [ROTULO_NUMERICO[linha[4]] for linha in SOBREVIVENTES]
    return X, y


# ---------------------------------------------------------------------------
# Bootstrap + sorteio de atributos por divisão (bagging x random forest)
# ---------------------------------------------------------------------------


def _sortear_indices_bootstrap(n: int, rng: random.Random) -> list:
    """Sorteia n índices COM reposição entre 0 e n-1: a amostra bootstrap."""
    return [rng.randrange(n) for _ in range(n)]


def treinar_arvores(X: list, y: list, n_arvores: int, max_features, seed_base: int = 0) -> list:
    """
    Treina n_arvores, cada uma na sua própria amostra bootstrap.
    max_features=None reproduz bagging puro (cada árvore considera todos
    os atributos em toda divisão); max_features=2 reproduz random forest
    (cada divisão sorteia só 2 dos 3 atributos como candidatos).
    """
    n = len(X)
    arvores = []
    for b in range(n_arvores):
        rng = random.Random(seed_base + b)
        indices = _sortear_indices_bootstrap(n, rng)
        X_amostra = [X[i] for i in indices]
        y_amostra = [y[i] for i in indices]
        modelo = DecisionTreeClassifier(max_depth=3, max_features=max_features, random_state=seed_base + b)
        modelo.fit(X_amostra, y_amostra)
        arvores.append(modelo)
    return arvores


def _feature_raiz(modelo: DecisionTreeClassifier) -> str:
    """Nome do atributo usado na divisão da raiz da árvore."""
    indice = modelo.tree_.feature[0]
    return NOMES_ATRIBUTOS[indice]


def _matriz_concordancia(arvores: list, X: list) -> list:
    """Matriz B x B com a fração de exemplos em que cada par de árvores concorda na previsão."""
    previsoes = [modelo.predict(X) for modelo in arvores]
    n = len(arvores)
    matriz = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matriz[i][j] = sum(a == b for a, b in zip(previsoes[i], previsoes[j])) / len(X)
    return matriz


def _concordancia_media_entre_arvores(matriz: list) -> float:
    """Média da concordância par a par, ignorando a diagonal (uma árvore sempre concorda consigo mesma)."""
    n = len(matriz)
    pares = [matriz[i][j] for i in range(n) for j in range(n) if i != j]
    return sum(pares) / len(pares)


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: bagging puro x random forest
# ---------------------------------------------------------------------------


def _mostrar_raizes(nome_metodo: str, arvores: list):
    """Mostra qual atributo cada árvore escolheu pra raiz, e a contagem consolidada."""
    raizes = [_feature_raiz(modelo) for modelo in arvores]
    print(f"\n--- Raiz de cada árvore ({nome_metodo}) ---")
    for b, raiz in enumerate(raizes, start=1):
        print(f"  Árvore {b}: raiz = {raiz}")
    print(f"  Consolidado: {dict(Counter(raizes))}")
    return raizes


def _mostrar_concordancia(nome_metodo: str, arvores: list, X: list) -> float:
    """Calcula e mostra a concordância média par a par entre as árvores de um método."""
    matriz = _matriz_concordancia(arvores, X)
    media = _concordancia_media_entre_arvores(matriz)
    print(f"Concordância média entre pares de árvores ({nome_metodo}): {media:.1%}")
    return matriz


def _mostrar_previsao_recruta(nome_metodo: str, arvores: list):
    """Mostra o voto de cada árvore do método pro Recruta Novo, e o veredito por maioria."""
    nome, armadura, agilidade, sorte = NOVO_RECRUTA
    votos = [modelo.predict([[armadura, agilidade, sorte]])[0] for modelo in arvores]
    contagem = Counter(votos)
    veredito = "Sim" if contagem[1] >= contagem[0] else "Não"
    print(
        f"{nome_metodo}: {contagem[1]} árvore(s) vota(m) 'Sim', {contagem[0]} vota(m) 'Não' "
        f"-> veredito = {veredito}"
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


def plotar_raizes(raizes_bagging: list, raizes_rf: list, caminho_saida: Path | None = None) -> Path:
    """Barras lado a lado: quantas árvores de cada método escolheram cada atributo pra raiz."""
    plt = _preparar_pyplot()
    import numpy as np

    contagens_bagging = Counter(raizes_bagging)
    contagens_rf = Counter(raizes_rf)
    posicoes = np.arange(len(NOMES_ATRIBUTOS))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(
        posicoes - largura / 2, [contagens_bagging[a] for a in NOMES_ATRIBUTOS], largura,
        label="bagging puro (max_features=None)", color="#C0392B",
    )
    ax.bar(
        posicoes + largura / 2, [contagens_rf[a] for a in NOMES_ATRIBUTOS], largura,
        label="random forest (max_features=2)", color="#4C72B0",
    )
    ax.set_xticks(posicoes)
    ax.set_xticklabels(NOMES_ATRIBUTOS)
    ax.set_ylabel("número de árvores com esse atributo na raiz")
    ax.set_title(f"Atributo escolhido pra raiz, em {len(raizes_bagging)} árvores de cada método")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "raizes_bagging_vs_rf.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_concordancia(matriz_bagging: list, matriz_rf: list, caminho_saida: Path | None = None) -> Path:
    """Dois mapas de calor lado a lado: concordância par a par entre árvores, bagging x random forest."""
    plt = _preparar_pyplot()
    import numpy as np

    fig, eixos = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, matriz, titulo in zip(
        eixos, (matriz_bagging, matriz_rf), ("bagging puro (max_features=None)", "random forest (max_features=2)")
    ):
        mapa = ax.imshow(np.array(matriz), vmin=0.5, vmax=1.0, cmap="RdYlBu_r")
        ax.set_xlabel("árvore")
        ax.set_ylabel("árvore")
        media = _concordancia_media_entre_arvores(matriz)
        ax.set_title(f"{titulo}\nconcordância média = {media:.1%}")
        fig.colorbar(mapa, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Concordância par a par entre árvores: quanto mais amarelo, mais parecidas")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "concordancia_bagging_vs_rf.png"
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
    Refaz à mão, com o torneio de sobrevivência de 16 participantes, a
    diferença entre bagging puro e random forest: mesma amostra bootstrap
    por árvore nos dois métodos, só variando se o atributo campeão
    disparado (armadura) pode ou não ser escondido em cada divisão.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine uma batalha royale numa floresta cheia de baús: existe "
        "uma arma campeã disparada (armadura), e se todo grupo pudesse "
        "sempre escolher livremente, quase todo grupo ia correr atrás "
        "dela primeiro. O random forest limita cada baú a um sorteio de "
        "só algumas armas disponíveis, forçando estratégias mais "
        "variadas entre os grupos."
    )

    X, y = _X_y()
    arvores_bagging = treinar_arvores(X, y, B_EXIBICAO, max_features=None, seed_base=0)
    arvores_rf = treinar_arvores(X, y, B_EXIBICAO, max_features=2, seed_base=0)

    raizes_bagging = _mostrar_raizes("bagging puro", arvores_bagging)
    raizes_rf = _mostrar_raizes("random forest", arvores_rf)

    print("\n--- Concordância entre as árvores de cada método ---")
    matriz_bagging = _mostrar_concordancia("bagging puro", arvores_bagging, X)
    matriz_rf = _mostrar_concordancia("random forest", arvores_rf, X)
    print(
        "\nAs duas florestas viram a mesma amostra bootstrap em cada árvore "
        "(a única diferença é max_features), então essa queda de "
        "concordância vem só de esconder o atributo campeão às vezes."
    )

    print(f"\n--- Prevendo o {NOVO_RECRUTA[0]} (armadura=25, agilidade=75, sorte=50) ---")
    _mostrar_previsao_recruta("bagging puro", arvores_bagging)
    _mostrar_previsao_recruta("random forest", arvores_rf)
    print(
        "\nNo bagging puro, praticamente toda árvore usa armadura de raiz, "
        "então o comitê inteiro concorda: armadura baixa, veredito 'Não'. "
        "No random forest, as árvores que tiveram armadura escondida "
        "sortearam agilidade como substituta, e agilidade alta aponta "
        "'Sim': o comitê passa a ter opiniões de verdade divergentes, "
        "mesmo quando a resposta final da maioria não muda."
    )

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_raizes(raizes_bagging, raizes_rf)
    plotar_concordancia(matriz_bagging, matriz_rf)
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _comparar_max_features(X_train, X_test, y_train, y_test, configuracoes=(None, "sqrt", 3)) -> dict:
    """Treina um RandomForestClassifier pra cada valor de max_features e compara F1 (classe fraude) e OOB."""
    print("\n--- Comparando max_features, no dataset de fraude ---")
    resultados = {}
    for config in configuracoes:
        modelo = RandomForestClassifier(
            n_estimators=100, max_features=config, oob_score=True, random_state=42, n_jobs=-1,
        ).fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        resultados[config] = {"modelo": modelo, "f1": f1, "oob": modelo.oob_score_}
        print(f"  max_features={config!r:<6}: F1={f1:.4f}   acurácia OOB={modelo.oob_score_:.4f}")
    return resultados


def _treinar_final_e_reportar(resultado: dict, X_test, y_test):
    """Imprime o classification_report completo do modelo escolhido como final (max_features='sqrt')."""
    y_pred = resultado["modelo"].predict(X_test)
    print(f"\n--- RandomForestClassifier final, max_features='sqrt', 100 árvores ---")
    print(classification_report(y_test, y_pred, digits=4))


def plotar_f1_por_max_features(resultados: dict, caminho_saida: Path | None = None) -> Path:
    """Barras de F1 (classe fraude) por valor de max_features testado."""
    plt = _preparar_pyplot()

    rotulos = [str(config) for config in resultados]
    valores_f1 = [resultados[config]["f1"] for config in resultados]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(rotulos, valores_f1, color="#4C72B0")
    ax.set_xlabel("max_features")
    ax.set_ylabel("F1 (classe fraude)")
    ax.set_title("F1 conforme max_features restringe os atributos por divisão")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "f1_por_max_features.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_importancia_atributos(modelo: RandomForestClassifier, nomes_atributos, top_n: int = 15, caminho_saida: Path | None = None) -> Path:
    """Ranking dos atributos mais importantes (redução de impureza acumulada) do modelo final."""
    plt = _preparar_pyplot()

    pares = sorted(zip(nomes_atributos, modelo.feature_importances_), key=lambda par: par[1], reverse=True)[:top_n]
    nomes, importancias = zip(*pares)

    fig, ax = plt.subplots(figsize=(7, 6))
    posicoes = range(len(nomes))
    ax.barh(posicoes, importancias[::-1], color="#4C72B0")
    ax.set_yticks(posicoes)
    ax.set_yticklabels(nomes[::-1])
    ax.set_xlabel("importância (redução de impureza acumulada)")
    ax.set_title(f"Top {top_n} atributos mais importantes pro random forest")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "importancia_atributos.png"
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
    resultados = _comparar_max_features(X_train, X_test, y_train, y_test)
    plotar_f1_por_max_features(resultados)

    _treinar_final_e_reportar(resultados["sqrt"], X_test, y_test)
    plotar_importancia_atributos(resultados["sqrt"]["modelo"], list(X_train.columns))


if __name__ == "__main__":
    main()
