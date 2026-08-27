"""
Árvore de decisão aplicada ao dataset de detecção de fraude.

Teoria completa (indução top-down, critérios de divisão como entropia,
ganho de informação e índice Gini, poda) está em `notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com Pokémon, refazendo na mão a conta de
     entropia, Gini e ganho de informação que a árvore faz escondida,
     pra sentir o algoritmo escolhendo a pergunta antes de ver isso
     rodando em cima de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_utils import get_train_test_split, load_raw_data, build_preprocessing_pipeline

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 14
# Pokémons, cada um com Tipo e se já evoluiu, marcados como "vale a pena
# treinar pra um Ginásio" (Sim) ou não (Não). Só pra "sentir" entropia,
# Gini e ganho de informação na mão, com mais de um atributo em jogo.
DATASET_POKEMON = [
    # (tipo, ja_evoluiu, bom_pra_ginasio)
    ("Fogo", "Não", "Não"),
    ("Fogo", "Não", "Não"),
    ("Fogo", "Sim", "Sim"),
    ("Fogo", "Sim", "Sim"),
    ("Fogo", "Não", "Não"),
    ("Água", "Sim", "Sim"),
    ("Água", "Não", "Sim"),
    ("Água", "Sim", "Sim"),
    ("Água", "Não", "Sim"),
    ("Grama", "Sim", "Sim"),
    ("Grama", "Sim", "Sim"),
    ("Grama", "Sim", "Sim"),
    ("Grama", "Não", "Não"),
    ("Grama", "Não", "Não"),
]


# ---------------------------------------------------------------------------
# Matemática de impureza (entropia, surpresa, Gini)
# ---------------------------------------------------------------------------


def surpresa(p: float) -> float:
    """
    'Surpresa' de um evento com probabilidade p, em bits: -log2(p).
    Quanto mais raro o evento (p pequeno), maior a surpresa ao ele
    ocorrer; um evento certo (p=1) tem surpresa 0.
    """
    return max(0.0, -math.log2(p))  # evita "-0.0" quando p=1


def entropia(rotulos: list) -> float:
    """
    Entropia de Shannon de uma lista de rótulos: a MÉDIA da surpresa de
    cada classe, ponderada pela própria probabilidade da classe.
    """
    total = len(rotulos)
    contagens = Counter(rotulos)
    probabilidades = [n / total for n in contagens.values()]
    valor = sum(p * surpresa(p) for p in probabilidades)
    return max(0.0, valor)  # evita "-0.0" quando o conjunto já é puro


def indice_gini(rotulos: list) -> float:
    """Índice de impureza Gini de uma lista de rótulos: 1 - Σ p_i²."""
    total = len(rotulos)
    contagens = Counter(rotulos)
    return 1 - sum((n / total) ** 2 for n in contagens.values())


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: árvore dos Pokémons
# ---------------------------------------------------------------------------


def _filtrar_pokemon(tipo: str | None = None, ja_evoluiu: str | None = None) -> list:
    """Filtra DATASET_POKEMON por Tipo e/ou 'Já evoluiu?'."""
    linhas = DATASET_POKEMON
    if tipo is not None:
        linhas = [linha for linha in linhas if linha[0] == tipo]
    if ja_evoluiu is not None:
        linhas = [linha for linha in linhas if linha[1] == ja_evoluiu]
    return linhas


def _rotulos_pokemon(tipo: str | None = None, ja_evoluiu: str | None = None) -> list:
    """Rótulos ('Sim'/'Não' de vale a captura) do subgrupo filtrado por Tipo e/ou evolução."""
    return [linha[2] for linha in _filtrar_pokemon(tipo, ja_evoluiu)]


def desenhar_arvore_ascii():
    """
    Desenha em texto a arvorezinha completa dos Pokémons, com as duas
    perguntas em sequência: primeiro Tipo, depois (só onde ainda estava
    misturado) Já evoluiu?
    """
    print(f"\n[ Todos os Pokémons: {dict(Counter(_rotulos_pokemon()))} ]")
    print('  Pergunta 1: "Qual o Tipo?"')
    print("   |")
    for tipo in ("Fogo", "Água", "Grama"):
        rotulos_tipo = _rotulos_pokemon(tipo=tipo)
        pura = len(set(rotulos_tipo)) == 1
        marca = "(folha, já é pura)" if pura else "(ainda misturado)"
        print(f"   +-- Tipo={tipo:<6} {dict(Counter(rotulos_tipo))}  {marca}")
        if pura:
            continue
        print('   |     Pergunta 2: "Já evoluiu?"')
        for ja_evoluiu in ("Sim", "Não"):
            rotulos_evoluiu = _rotulos_pokemon(tipo=tipo, ja_evoluiu=ja_evoluiu)
            if rotulos_evoluiu:
                print(
                    f"   |     +-- Evoluiu={ja_evoluiu:<4} "
                    f"{dict(Counter(rotulos_evoluiu))}  (folha, já é pura)"
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


def plotar_arvore_pokemon(caminho_saida: Path | None = None) -> Path:
    """
    Desenha a árvore dos Pokémons como um diagrama de verdade (caixas e
    linhas), com as duas perguntas em sequência: Tipo e, dentro dos
    galhos ainda misturados, Já evoluiu?
    """
    plt = _preparar_pyplot()

    caixa_interna = dict(boxstyle="round,pad=0.5", facecolor="#FFF3CD", edgecolor="#333333")
    caixa_folha = dict(boxstyle="round,pad=0.5", facecolor="#C8E6C9", edgecolor="#333333")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def no(x, y, titulo, tipo=None, ja_evoluiu=None):
        rotulos = _rotulos_pokemon(tipo, ja_evoluiu)
        contagem, h = dict(Counter(rotulos)), entropia(rotulos)
        # a raiz nunca é folha; os nós de "Já evoluiu?" sempre são (fim da recursão neste exemplo)
        folha = ja_evoluiu is not None or (tipo is not None and h == 0.0)
        ax.text(
            x, y, f"{titulo}\n{contagem}\nH = {h:.3f}",
            ha="center", va="center", fontsize=9,
            bbox=caixa_folha if folha else caixa_interna,
        )

    def aresta(x1, y1, x2, y2, rotulo):
        ax.plot([x1, x2], [y1 - 0.06, y2 + 0.08], color="#777777", linewidth=1.2, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, rotulo, fontsize=8, ha="center", color="#444444")

    no(0.5, 0.92, "Todos os Pokémons")

    pos_tipo = {"Fogo": 0.15, "Água": 0.5, "Grama": 0.85}
    for tipo, x in pos_tipo.items():
        no(x, 0.58, f"Tipo = {tipo}", tipo=tipo)
        aresta(0.5, 0.92, x, 0.58, tipo)

    pos_evoluiu = {
        "Fogo": {"Não": 0.03, "Sim": 0.27},
        "Grama": {"Sim": 0.73, "Não": 0.97},
    }
    for tipo, filhos in pos_evoluiu.items():
        for ja_evoluiu, x in filhos.items():
            no(x, 0.20, f"Evoluiu? {ja_evoluiu}", tipo=tipo, ja_evoluiu=ja_evoluiu)
            aresta(pos_tipo[tipo], 0.58, x, 0.20, ja_evoluiu)

    ax.set_title("Árvore dos Pokémons: duas perguntas, Tipo e Já evoluiu?", fontsize=12)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "arvore_pokemon.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Árvore desenhada em: {caminho_saida}")
    return caminho_saida


def plotar_contagem_por_tipo(caminho_saida: Path | None = None) -> Path:
    """
    Gráfico de barras: quantos Pokémons de cada Tipo valem (Sim) ou não
    (Não) a captura. Mostra visualmente por que 'Água' é um grupo "puro"
    (uma cor só) enquanto 'Fogo' e 'Grama' vêm misturados.
    """
    plt = _preparar_pyplot()

    tipos = sorted(set(linha[0] for linha in DATASET_POKEMON))
    contagens_sim = [Counter(_rotulos_pokemon(tipo=tipo))["Sim"] for tipo in tipos]
    contagens_nao = [Counter(_rotulos_pokemon(tipo=tipo))["Não"] for tipo in tipos]

    fig, ax = plt.subplots(figsize=(6, 4))
    posicoes = range(len(tipos))
    ax.bar(posicoes, contagens_sim, label="Sim, vale a captura", color="#4CAF50")
    ax.bar(posicoes, contagens_nao, bottom=contagens_sim, label="Não vale", color="#E57373")
    ax.set_xticks(list(posicoes))
    ax.set_xticklabels(tipos)
    ax.set_ylabel("Quantidade de Pokémons")
    ax.set_title("Pokémons por Tipo: quão 'misturado' cada grupo é")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "pokemons_por_tipo.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_curvas_impureza(caminho_saida: Path | None = None) -> Path:
    """
    Plota Entropia(p) e Gini(p) para o caso binário, com p variando de 0 a
    1: visualiza por que as duas valem 0 nos extremos (grupo puro) e são
    máximas em p=0.5 (máxima "bagunça"), só que em escalas diferentes
    (entropia chega a 1, Gini chega a 0,5).
    """
    plt = _preparar_pyplot()
    import numpy as np

    p = np.linspace(1e-3, 1 - 1e-3, 500)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    gini = 1 - (p**2 + (1 - p) ** 2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p, h, label="Entropia(p) = -p·log2(p) - (1-p)·log2(1-p)", linewidth=2)
    ax.plot(p, gini, label="Gini(p) = 1 - p² - (1-p)²", linewidth=2)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.scatter([0, 0.5, 1], [0, 1, 0], color="black", zorder=5)
    ax.set_xlabel("p (proporção da classe positiva)")
    ax.set_ylabel("impureza (bagunça do grupo)")
    ax.set_title("Entropia x Gini: duas réguas pra medir a mesma bagunça")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "curvas_impureza.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_arvore_treinada(modelo, nomes_atributos, caminho_saida: Path | None = None) -> Path:
    """
    Desenha a árvore já treinada pelo scikit-learn (nós coloridos pela
    classe majoritária, mostrando o atributo e o limiar de cada divisão).
    Complementa o `export_text`, que é preciso mas ilegível para quem não
    tem prática com árvores em texto.
    """
    plt = _preparar_pyplot()
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(
        modelo,
        feature_names=nomes_atributos,
        class_names=["normal", "fraude"],
        filled=True,
        rounded=True,
        fontsize=6,
        ax=ax,
    )
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "arvore_fraude_podada.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Árvore treinada desenhada em: {caminho_saida}")
    return caminho_saida


# ---------------------------------------------------------------------------
# Demonstração manual (Parte 1 e 2)
# ---------------------------------------------------------------------------


def _titulo(texto: str):
    """Imprime um cabeçalho de seção padronizado no terminal."""
    print("=" * 78)
    print(texto)
    print("=" * 78)


def _explicar_surpresa_em_bits():
    """Aquecimento: mostra a surpresa caindo pela metade a cada bit, com números concretos."""
    print("\n--- Aquecimento: 'surpresa', em bits ---")
    print(
        "Surpresa é 'quão bizarra seria essa notícia'. Um evento certo "
        "(100% de chance) não surpreende ninguém. Cada vez que a chance "
        "cai pela metade, a surpresa sobe exatamente 1 bit:"
    )
    fracoes = {1: "100%", 0.5: "50%", 0.25: "25%", 0.125: "12,5%"}
    for p in (1, 0.5, 0.25, 0.125):
        print(f"  chance={fracoes[p]:<6} -> surpresa = {surpresa(p):.3f} bit(s)")


def _explicar_extremos_da_entropia():
    """Aquecimento: entropia de um grupo puro (0) versus um grupo 50/50 (máxima)."""
    print("\n--- Aquecimento: os dois extremos da entropia ---")
    potinho_uma_cor = ["Sim", "Sim", "Sim", "Sim"]
    potinho_meio_a_meio = ["Sim", "Não"]
    print(
        f"Potinho só de 'Sim'       {potinho_uma_cor} "
        f"-> Entropia = {entropia(potinho_uma_cor):.3f}  "
        "(zero dúvida, nem preciso perguntar)"
    )
    print(
        f"Potinho 50/50             {potinho_meio_a_meio}          "
        f"-> Entropia = {entropia(potinho_meio_a_meio):.3f}  "
        "(dúvida máxima, 1 pergunta boa resolve)"
    )


def _dividir_por_tipo():
    """Pergunta 1: divide os 14 Pokémons por Tipo e mede quanto isso reduz a dúvida."""
    print("\n--- Pergunta 1: 14 Pokémons, agrupados por Tipo ---")
    rotulos_totais = _rotulos_pokemon()
    print(f"Contagem geral: {dict(Counter(rotulos_totais))}  (9 valem a captura, 5 não valem)")

    h_pai = entropia(rotulos_totais)
    g_pai = indice_gini(rotulos_totais)
    print(f"\nEntropia do grupo inteiro = {h_pai:.3f}  (bem misturado, dúvida alta)")
    print(f"Gini do grupo inteiro     = {g_pai:.3f}")

    print("\nSerá que a pergunta 'Qual o Tipo?' ajuda a diminuir essa dúvida?")
    tipos = sorted(set(linha[0] for linha in DATASET_POKEMON))
    h_filhos_ponderada = 0.0
    g_filhos_ponderada = 0.0
    for tipo in tipos:
        sub = _rotulos_pokemon(tipo=tipo)
        peso = len(sub) / len(DATASET_POKEMON)
        h_sub = entropia(sub)
        g_sub = indice_gini(sub)
        h_filhos_ponderada += peso * h_sub
        g_filhos_ponderada += peso * g_sub
        print(
            f"  Tipo={tipo:<6} {dict(Counter(sub))!s:<20} "
            f"({peso:.0%} dos Pokémons)  Entropia={h_sub:.3f}  Gini={g_sub:.3f}"
        )

    print(f"\nEntropia média dos subgrupos = {h_filhos_ponderada:.3f}")
    print(
        f"Ganho de informação = {h_pai:.3f} - {h_filhos_ponderada:.3f} = "
        f"{h_pai - h_filhos_ponderada:.3f}  <- a dúvida caiu bastante!"
    )
    print(f"\nGini médio dos subgrupos    = {g_filhos_ponderada:.3f}")
    print(f"Redução de impureza (Gini)   = {g_pai - g_filhos_ponderada:.3f}")

    print(
        "\n'Água' já virou uma folha pura, mas 'Fogo' e 'Grama' continuam "
        "misturados. É aqui que a recursão entra: a árvore repete a MESMA "
        "conta dentro de cada um desses dois galhos, testando o próximo "
        "atributo disponível, 'Já evoluiu?'."
    )


def _dividir_por_evolucao():
    """Pergunta 2: repete a mesma conta dentro dos galhos que ainda ficaram misturados (Fogo, Grama)."""
    print("\n--- Pergunta 2 (só dentro dos galhos ainda misturados) ---")
    for tipo in ("Fogo", "Grama"):
        sub_tipo = _filtrar_pokemon(tipo=tipo)
        rotulos_tipo = _rotulos_pokemon(tipo=tipo)
        h_no = entropia(rotulos_tipo)
        print(f"\nDentro de Tipo={tipo}: {dict(Counter(rotulos_tipo))}  Entropia = {h_no:.3f}")

        h_filhos_no = 0.0
        for ja_evoluiu in sorted(set(linha[1] for linha in sub_tipo)):
            sub2 = _rotulos_pokemon(tipo=tipo, ja_evoluiu=ja_evoluiu)
            peso2 = len(sub2) / len(sub_tipo)
            h_sub2 = entropia(sub2)
            h_filhos_no += peso2 * h_sub2
            print(
                f"  Evoluiu={ja_evoluiu:<4} {dict(Counter(sub2))!s:<15} "
                f"({peso2:.0%})  Entropia={h_sub2:.3f}"
            )

        print(
            f"  Ganho de 'Já evoluiu?' dentro de {tipo} = {h_no:.3f} - "
            f"{h_filhos_no:.3f} = {h_no - h_filhos_no:.3f}"
        )

    print(
        "\nCom duas perguntas (Tipo e Já evoluiu?), a árvore chega a folhas "
        "100% puras: dentro de Fogo e de Grama, saber se o Pokémon já "
        "evoluiu resolve toda a dúvida que sobrou. Numa árvore de "
        "verdade, com dezenas de atributos, esse processo continuaria "
        "até não sobrar mais dúvida pra resolver, ou até bater num limite "
        "de profundidade (a pré-poda que vem mais adiante)."
    )


def demonstracao_manual():
    """
    Refaz à mão, com o exemplo de brincadeira dos Pokémons, a conta que
    uma árvore de decisão faz escondida por trás do
    `DecisionTreeClassifier` do scikit-learn ao escolher qual pergunta
    fazer em cada nó, incluindo a segunda rodada de perguntas dentro dos
    galhos que ainda ficaram misturados.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine que você é um treinador Pokémon com 14 Pokémons na "
        "mochila e quer decidir: 'vale a pena treinar esse aqui pra um "
        "Ginásio, ou não?' Cada um já tem essa etiqueta colada (Sim/Não), "
        "que é o que uma árvore de decisão aprenderia a prever em novos "
        "Pokémons."
    )

    _explicar_surpresa_em_bits()
    _explicar_extremos_da_entropia()
    _dividir_por_tipo()
    _dividir_por_evolucao()
    desenhar_arvore_ascii()

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_arvore_pokemon()
    plotar_contagem_por_tipo()
    plotar_curvas_impureza()
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _comparar_criterios(X_train, X_test, y_train, y_test):
    """Treina um DecisionTreeClassifier com cada critério de divisão e imprime o desempenho."""
    for criterio in ("gini", "entropy"):
        modelo = DecisionTreeClassifier(criterion=criterio, random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\n--- criterion='{criterio}' (profundidade da árvore: {modelo.get_depth()}) ---")
        print(classification_report(y_test, y_pred, digits=4))


def _treinar_com_pre_poda(X_train, X_test, y_train, y_test):
    """Treina uma árvore rasa (max_depth=4) para inspecionar as regras aprendidas."""
    modelo_podado = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    modelo_podado.fit(X_train, y_train)
    y_pred_podado = modelo_podado.predict(X_test)

    print("\n--- max_depth=4 ---")
    print(classification_report(y_test, y_pred_podado, digits=4))
    print("\nRegras aprendidas (árvore limitada a profundidade 4):\n")
    print(export_text(modelo_podado, feature_names=list(X_train.columns)))
    plotar_arvore_treinada(modelo_podado, list(X_train.columns))


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    _comparar_criterios(X_train, X_test, y_train, y_test)

    _titulo("PRÉ-PODA (max_depth): árvore mais simples, para inspecionar as regras")
    _treinar_com_pre_poda(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
