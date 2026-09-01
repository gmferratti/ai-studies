"""
Apriori (regras de associação) aplicado ao dataset de detecção de fraude.

Teoria completa (suporte, confiança, lift, propriedade Apriori de poda por
anti-monotonicidade) está em `notes/anotacoes.md`.

Este arquivo tem três partes:
  1) Exemplo de brincadeira com decks de um card game, refazendo na mão,
     nível por nível, a geração de candidatos e a poda que o Apriori faz
     escondida, até chegar nas regras de associação finais.
  2) Visualização desse mesmo exemplo (suporte de cada itemset avaliado e
     o grafo das regras encontradas).
  3) Mineração de regras de verdade no dataset de fraude de cartão de
     crédito, discretizando os atributos contínuos em faixas pra virarem
     "itens" de uma cesta.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from utils.data_utils import load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

MIN_SUPORTE = 0.4
MIN_CONFIANCA = 0.6

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 10
# jogadores contam quais cartas entraram no deck que montaram pra um card
# game. "Grimório" aparece só numa vez de propósito, pra já ser podado na
# primeira rodada por suporte baixo.
DATASET_DECKS = [
    {"Dragão", "Espada", "Escudo"},
    {"Dragão", "Espada"},
    {"Dragão", "Escudo", "Poção"},
    {"Espada", "Escudo"},
    {"Dragão", "Espada", "Escudo", "Poção", "Grimório"},
    {"Escudo", "Poção"},
    {"Dragão", "Poção"},
    {"Dragão", "Espada", "Poção"},
    {"Espada", "Poção"},
    {"Dragão", "Espada", "Escudo"},
]


# ---------------------------------------------------------------------------
# Matemática do Apriori (suporte, geração de candidatos, poda, regras)
# ---------------------------------------------------------------------------


def suporte(itemset: frozenset, decks: list = DATASET_DECKS) -> float:
    """Fração dos decks que contêm TODAS as cartas do itemset."""
    return sum(1 for deck in decks if itemset <= deck) / len(decks)


def _itens_frequentes_nivel_1(decks: list, min_suporte: float) -> tuple[dict, dict]:
    """Conta cada carta sozinha e descarta as que não batem o suporte mínimo."""
    todas_cartas = sorted(set().union(*decks))
    avaliados, frequentes = {}, {}
    for carta in todas_cartas:
        itemset = frozenset({carta})
        s = suporte(itemset, decks)
        avaliados[itemset] = s
        estado = "OK" if s >= min_suporte else "podada (suporte baixo)"
        print(f"  {{{carta}}}: suporte = {s:.1f}  -> {estado}")
        if s >= min_suporte:
            frequentes[itemset] = s
    return frequentes, avaliados


def _gerar_candidatos(frequentes_anteriores: dict, tamanho: int) -> list:
    """Junta as cartas que já apareceram nos itemsets frequentes do nível
    anterior em candidatos do tamanho pedido (passo de junção do Apriori)."""
    cartas = sorted(set().union(*frequentes_anteriores.keys()))
    return [frozenset(c) for c in combinations(cartas, tamanho)]


def _podar_por_subconjuntos(candidatos: list, frequentes_anteriores: dict) -> list:
    """
    Propriedade Apriori (anti-monotonicidade): um itemset só pode ser
    frequente se TODOS os seus subconjuntos também forem. Elimina, sem
    nem olhar os decks de novo, qualquer candidato que já carregue um
    pedaço que não sobreviveu ao nível anterior.
    """
    sobreviventes = []
    for candidato in candidatos:
        subconjuntos = combinations(candidato, len(candidato) - 1)
        if all(frozenset(sub) in frequentes_anteriores for sub in subconjuntos):
            sobreviventes.append(candidato)
        else:
            print(f"  {set(candidato)}: podado antes de contar (tem subconjunto que não é frequente)")
    return sobreviventes


def _itens_frequentes_proximo_nivel(
    frequentes_anteriores: dict, tamanho: int, decks: list, min_suporte: float
) -> tuple[dict, dict]:
    """Um nível completo do Apriori: gera candidatos, poda pela propriedade
    Apriori e só depois conta de verdade quem sobrou."""
    candidatos = _gerar_candidatos(frequentes_anteriores, tamanho)
    print(f"\nCandidatos de tamanho {tamanho} gerados por junção: {[set(c) for c in candidatos]}")

    sobreviventes = _podar_por_subconjuntos(candidatos, frequentes_anteriores)
    print(f"Sobreviveram à poda por subconjunto: {[set(c) for c in sobreviventes]}")

    avaliados, frequentes = {}, {}
    for candidato in sobreviventes:
        s = suporte(candidato, decks)
        avaliados[candidato] = s
        estado = "OK" if s >= min_suporte else "podado (suporte baixo na contagem real)"
        print(f"  {set(candidato)}: suporte = {s:.1f}  -> {estado}")
        if s >= min_suporte:
            frequentes[candidato] = s
    return frequentes, avaliados


def _gerar_regras(frequentes: dict, min_confianca: float) -> list:
    """Pra cada itemset frequente com 2+ cartas, testa todo jeito de
    separar antecedente/consequente e mede confiança e lift."""
    regras = []
    for itemset, sup_itemset in frequentes.items():
        if len(itemset) < 2:
            continue
        for tamanho_antecedente in range(1, len(itemset)):
            for combinacao in combinations(itemset, tamanho_antecedente):
                antecedente = frozenset(combinacao)
                consequente = itemset - antecedente
                confianca = sup_itemset / frequentes[antecedente]
                lift = confianca / frequentes[consequente]
                if confianca >= min_confianca:
                    regras.append((antecedente, consequente, sup_itemset, confianca, lift))
    return sorted(regras, key=lambda regra: -regra[4])


# ---------------------------------------------------------------------------
# Gráficos (salvos em images/, dentro da pasta deste script)
# ---------------------------------------------------------------------------


def _preparar_pyplot():
    """Configura o backend sem interface gráfica (necessário em servidor/terminal) e devolve o pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plotar_suporte_itemsets(avaliados: dict, min_suporte: float, caminho_saida: Path | None = None) -> Path:
    """
    Barras com o suporte de cada itemset que o Apriori de fato contou
    (nível 1 e nível 2), verde quando passou do suporte mínimo, vermelho
    quando foi podado. Mostra de cara por que o Grimório nem chega perto
    de virar item frequente.
    """
    plt = _preparar_pyplot()

    itens_ordenados = sorted(avaliados.items(), key=lambda kv: (len(kv[0]), -kv[1]))
    rotulos = [" + ".join(sorted(chave)) for chave, _ in itens_ordenados]
    suportes = [valor for _, valor in itens_ordenados]
    cores = ["#4CAF50" if s >= min_suporte else "#E57373" for s in suportes]

    fig, ax = plt.subplots(figsize=(11, 5))
    posicoes = range(len(rotulos))
    ax.bar(posicoes, suportes, color=cores)
    ax.axhline(min_suporte, color="#333333", linestyle="--", linewidth=1, label=f"suporte mínimo = {min_suporte}")
    ax.set_xticks(list(posicoes))
    ax.set_xticklabels(rotulos, rotation=40, ha="right")
    ax.set_ylabel("suporte")
    ax.set_title("Suporte de cada itemset avaliado: verde passou, vermelho foi podado")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "suporte_itemsets.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico de suporte salvo em: {caminho_saida}")
    return caminho_saida


def plotar_grafo_regras(regras: list, caminho_saida: Path | None = None) -> Path:
    """
    Desenha as 4 cartas como nós e cada regra como uma seta entre elas:
    verde e contínua quando o lift é maior que 1 (associação real), cinza
    e tracejada quando o lift é menor ou igual a 1 (só popularidade).
    """
    plt = _preparar_pyplot()

    posicoes = {
        "Dragão": (0.5, 0.92),
        "Espada": (0.92, 0.35),
        "Escudo": (0.5, 0.05),
        "Poção": (0.08, 0.35),
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    for carta, (x, y) in posicoes.items():
        ax.scatter([x], [y], s=2200, color="#FFF3CD", edgecolor="#333333", zorder=3)
        ax.text(x, y, carta, ha="center", va="center", fontsize=10, zorder=4)

    contagem_pares = Counter()
    for antecedente, consequente, _sup, conf, lift in regras:
        (origem,) = antecedente
        (destino,) = consequente
        par = tuple(sorted((origem, destino)))
        contagem_pares[par] += 1
        curvatura = 0.15 if contagem_pares[par] == 1 else -0.15

        cor = "#2E7D32" if lift > 1 else "#90A4AE"
        estilo = "-" if lift > 1 else "--"
        x0, y0 = posicoes[origem]
        x1, y1 = posicoes[destino]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=cor,
                linestyle=estilo,
                linewidth=1.8,
                connectionstyle=f"arc3,rad={curvatura}",
                shrinkA=30,
                shrinkB=30,
            ),
            zorder=2,
        )
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2 + curvatura * 0.35
        ax.text(xm, ym, f"conf={conf:.2f}\nlift={lift:.2f}", fontsize=7.5, color=cor, ha="center")

    ax.set_title("Regras de associação entre as cartas\n(verde = associação real, cinza = só popularidade)")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "grafo_regras.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Grafo de regras salvo em: {caminho_saida}")
    return caminho_saida


def plotar_regras_fraude(regras_fraude: pd.DataFrame, caminho_saida: Path | None = None) -> Path:
    """Barras com o lift de cada regra encontrada no dataset de fraude, da mais fraca pra mais forte."""
    plt = _preparar_pyplot()

    regras_fraude = regras_fraude.sort_values("lift")
    rotulos = [" + ".join(sorted(item.replace("_", "=") for item in ante)) for ante in regras_fraude["antecedents"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(rotulos, regras_fraude["lift"], color="#C62828")
    ax.axvline(1, color="#333333", linestyle="--", linewidth=1, label="lift = 1 (nenhuma associação)")
    ax.set_xlabel("lift (quantas vezes mais provável que o acaso)")
    ax.set_title("Combinações de faixas mais associadas a fraude")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "lift_regras_fraude.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico de lift salvo em: {caminho_saida}")
    return caminho_saida


# ---------------------------------------------------------------------------
# Demonstração manual (Parte 1 e 2)
# ---------------------------------------------------------------------------


def _titulo(texto: str):
    """Imprime um cabeçalho de seção padronizado no terminal."""
    print("=" * 78)
    print(texto)
    print("=" * 78)


def _explicar_e_imprimir_regras(regras: list):
    print(f"\n--- Regras de associação (confiança >= {MIN_CONFIANCA:.0%}) ---")
    for antecedente, consequente, sup, conf, lift in regras:
        seta = f"{sorted(antecedente)} -> {sorted(consequente)}"
        tipo = "associação real" if lift > 1 else "só popularidade, sem associação de verdade"
        print(f"  {seta:<28} suporte={sup:.2f}  confiança={conf:.3f}  lift={lift:.3f}  ({tipo})")

    print(
        "\nRepara na pegadinha: 'Escudo -> Dragão' tem confiança de 0,667, "
        "quase 67% dos decks com Escudo também têm Dragão, parece uma regra "
        "e tanto. Mas o lift é 0,952, abaixo de 1: Dragão sozinho já aparece "
        "em 70% de TODOS os decks, então achar Dragão junto de Escudo não é "
        "surpresa nenhuma, é só reflexo de Dragão ser popular demais. "
        "Confiança alta sozinha engana; o lift é quem mostra se a segunda "
        "carta aparece MAIS do que o normal quando a primeira está no deck, "
        "ou se é só coincidência de duas cartas populares."
    )


def demonstracao_manual():
    """
    Refaz à mão, com o exemplo de brincadeira dos decks, a geração de
    candidatos e a poda que o Apriori faz escondido nível por nível, até
    não sobrar mais nenhum itemset frequente novo pra combinar.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine 10 jogadores contando quais cartas entraram no deck que "
        "cada um montou pra um card game. Cada deck é uma 'cesta de "
        "compras', cada carta é um 'item': a mesma lógica usada pra "
        "descobrir, num mercado de verdade, que quem compra fralda também "
        "costuma comprar cerveja."
    )
    for i, deck in enumerate(DATASET_DECKS, start=1):
        print(f"  Deck {i:>2}: {sorted(deck)}")

    print("\n--- Nível 1: contando cada carta sozinha ---")
    nivel1, avaliados1 = _itens_frequentes_nivel_1(DATASET_DECKS, MIN_SUPORTE)

    print("\n--- Nível 2: juntando pares a partir das cartas que sobraram ---")
    nivel2, avaliados2 = _itens_frequentes_proximo_nivel(nivel1, 2, DATASET_DECKS, MIN_SUPORTE)

    print("\n--- Nível 3: juntando trincas a partir dos pares que sobraram ---")
    nivel3, _avaliados3 = _itens_frequentes_proximo_nivel(nivel2, 3, DATASET_DECKS, MIN_SUPORTE)
    if not nivel3:
        print(
            "\nNenhuma trinca sobreviveu (mesmo a única que passou pela poda "
            "por subconjunto caiu na contagem real): o Apriori para por "
            "aqui, nível 4 nem chega a ser tentado."
        )

    frequentes = {**nivel1, **nivel2, **nivel3}
    regras = _gerar_regras(frequentes, MIN_CONFIANCA)
    _explicar_e_imprimir_regras(regras)

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_suporte_itemsets({**avaliados1, **avaliados2}, MIN_SUPORTE)
    plotar_grafo_regras(regras)
    print("=" * 78 + "\n")

    return regras


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _selecionar_atributos_mais_correlacionados(df: pd.DataFrame, n: int = 3) -> list:
    """Escolhe os n atributos V (componentes de PCA) mais correlacionados, em módulo, com Class."""
    correlacoes = df.corr(numeric_only=True)["Class"].drop("Class")
    correlacoes_v = correlacoes.filter(like="V").abs().sort_values(ascending=False)
    escolhidos = list(correlacoes_v.head(n).index)
    print(f"Atributos V mais correlacionados com fraude: {escolhidos}")
    return escolhidos


def _discretizar_em_cesta(df: pd.DataFrame, atributos_v: list) -> pd.DataFrame:
    """
    Transforma o dataset contínuo numa 'cesta de itens': cada transação
    (linha) vira um conjunto de faixas (valor da compra, período do dia,
    os atributos V escolhidos) mais a própria classe, tudo em colunas
    booleanas. Apriori não sabe lidar com número contínuo, só com
    presença/ausência de item, então essa discretização é obrigatória.
    """
    cesta = pd.get_dummies(pd.qcut(df["Amount"], q=3, labels=["Valor_baixo", "Valor_médio", "Valor_alto"]))

    # 'Time' é o número de segundos desde a primeira transação do dataset
    # (só 2 dias de dados), não um relógio de parede de verdade; tratamos
    # o resto da divisão por 24h como uma aproximação de "hora do dia".
    hora_do_dia = (df["Time"] % 86400) // 3600
    periodo = pd.cut(hora_do_dia, bins=[-1, 6, 12, 18, 24], labels=["Madrugada", "Manhã", "Tarde", "Noite"])
    cesta = cesta.join(pd.get_dummies(periodo))

    for atributo in atributos_v:
        mediana = df[atributo].median()
        cesta[f"{atributo}_alto"] = df[atributo] > mediana
        cesta[f"{atributo}_baixo"] = df[atributo] <= mediana

    cesta["Classe_fraude"] = df["Class"] == 1
    cesta["Classe_normal"] = df["Class"] == 0

    return cesta.astype(bool)


def _minerar_regras_fraude(cesta: pd.DataFrame, min_suporte: float = 0.001) -> pd.DataFrame:
    """
    Roda o Apriori de verdade (via mlxtend, o scikit-learn não implementa
    regras de associação) na cesta discretizada e filtra só as regras cujo
    consequente é 'a transação é fraude'.
    """
    frequentes = apriori(cesta, min_support=min_suporte, use_colnames=True, max_len=4, low_memory=True)
    print(f"Itemsets frequentes encontrados (suporte mínimo {min_suporte}): {len(frequentes)}")

    regras = association_rules(frequentes, metric="confidence", min_threshold=0.0)
    regras_fraude = regras[regras["consequents"] == frozenset({"Classe_fraude"})].sort_values(
        "lift", ascending=False
    )

    print(f"\nRegras que apontam pra fraude: {len(regras_fraude)}")
    for _, regra in regras_fraude.iterrows():
        antecedente = sorted(regra["antecedents"])
        print(
            f"  {antecedente}  suporte={regra['support']:.4f}  "
            f"confiança={regra['confidence']:.4f}  lift={regra['lift']:.3f}"
        )

    return regras_fraude


def main():
    demonstracao_manual()

    df = load_raw_data().dropna()

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    atributos_v = _selecionar_atributos_mais_correlacionados(df)
    cesta = _discretizar_em_cesta(df, atributos_v)
    regras_fraude = _minerar_regras_fraude(cesta)

    print(
        "\nRepare que a confiança de cada regra sozinha é baixíssima (a "
        "maior passa perto de 1,5%), porque fraude é rara: só 0,17% de "
        "TODAS as transações. O que importa aqui é o lift, não a confiança "
        "em valor absoluto: combinar as três faixas baixas de V17, V14 e "
        "V12 num único itemset multiplica a chance de fraude por mais de 8 "
        "vezes em relação ao acaso, bem mais forte do que qualquer um dos "
        "três atributos usado sozinho."
    )
    plotar_regras_fraude(regras_fraude)


if __name__ == "__main__":
    main()
