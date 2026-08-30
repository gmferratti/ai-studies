"""
Bagging (Bootstrap Aggregating) aplicado ao dataset de detecção de fraude.

Teoria completa (amostragem bootstrap, agregação por voto, decomposição
viés/variância, erro out-of-bag) está em `notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com um "júri" de 16 suspeitos e vários
     investigadores, cada um vendo uma amostra bootstrap diferente,
     votando de forma divergente e depois convergindo por agregação, pra
     sentir o algoritmo reduzindo variância antes de ver isso rodando em
     cima de dados de verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito,
     incluindo uma comparação de variância entre uma árvore única e um
     comitê de bagging em várias sementes aleatórias.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 16
# suspeitos num "julgamento de classe", com dois atributos (nervosismo e
# número de contradições no depoimento) e o veredito real (Sim = culpado).
# Os suspeitos G, I e J ficam propositalmente numa zona de sobreposição
# entre os dois grupos (nenhum corte reto nos separa direito), pra fazer
# os investigadores discordarem entre si dependendo da amostra bootstrap
# que sortearam.
SUSPEITOS = [
    # (nome, nervosismo, contradicoes, culpado)
    ("Aluno A", 1.0, 0.5, "Não"),
    ("Aluno B", 1.5, 1.0, "Não"),
    ("Aluno C", 2.0, 1.5, "Não"),
    ("Aluno D", 2.5, 0.5, "Não"),
    ("Aluno E", 3.5, 2.5, "Não"),
    ("Aluno F", 4.0, 3.0, "Não"),
    ("Aluno G", 5.0, 3.5, "Não"),  # zona de sobreposição
    ("Aluno H", 6.5, 1.5, "Não"),
    ("Aluno I", 4.5, 3.5, "Sim"),  # zona de sobreposição
    ("Aluno J", 5.5, 3.0, "Sim"),  # zona de sobreposição
    ("Aluno K", 6.0, 4.0, "Sim"),
    ("Aluno L", 7.0, 4.0, "Sim"),
    ("Aluno M", 7.5, 4.5, "Sim"),
    ("Aluno N", 8.0, 4.0, "Sim"),
    ("Aluno O", 8.5, 5.0, "Sim"),
    ("Aluno P", 9.0, 4.5, "Sim"),
]

# Suspeito novo, caído bem na zona cinzenta entre os dois grupos: é
# justamente aí que os investigadores tendem a discordar entre si.
ALUNO_NOVO = ("Aluno Novo", 6.0, 3.0)

ROTULO_NUMERICO = {"Não": 0, "Sim": 1}
B_EXIBICAO = 9  # investigadores mostrados no terminal e nos gráficos individuais
B_TOTAL = 51  # tamanho do "júri completo", usado na curva de estabilização


def _X_y():
    """Extrai X (nervosismo, contradições) e y (0/1) de SUSPEITOS."""
    X = [(linha[1], linha[2]) for linha in SUSPEITOS]
    y = [ROTULO_NUMERICO[linha[3]] for linha in SUSPEITOS]
    return X, y


# ---------------------------------------------------------------------------
# Bootstrap e agregação (a receita do bagging, na mão)
# ---------------------------------------------------------------------------


def _sortear_indices_bootstrap(n: int, rng: random.Random) -> list:
    """Sorteia n índices COM reposição entre 0 e n-1: a amostra bootstrap."""
    return [rng.randrange(n) for _ in range(n)]


def _indices_out_of_bag(indices_amostra: list, n: int) -> list:
    """Índices de 0 a n-1 que não saíram no sorteio da amostra bootstrap."""
    sorteados = set(indices_amostra)
    return [i for i in range(n) if i not in sorteados]


def treinar_investigadores(X: list, y: list, n_investigadores: int, seed_base: int = 0) -> list:
    """
    Treina um comitê de investigadores (stumps rasos), cada um numa amostra
    bootstrap diferente do mesmo dataset, guardando também quem cada um
    deixou de fora (out-of-bag) pra uso posterior.
    """
    n = len(X)
    investigadores = []
    for b in range(n_investigadores):
        rng = random.Random(seed_base + b)
        indices_amostra = _sortear_indices_bootstrap(n, rng)
        indices_oob = _indices_out_of_bag(indices_amostra, n)

        X_amostra = [X[i] for i in indices_amostra]
        y_amostra = [y[i] for i in indices_amostra]
        modelo = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_amostra, y_amostra)

        investigadores.append(
            {"modelo": modelo, "indices_amostra": indices_amostra, "indices_oob": indices_oob}
        )
    return investigadores


def _votar_maioria(votos: list) -> int:
    """Classe mais votada entre uma lista de votos (0/1)."""
    return Counter(votos).most_common(1)[0][0]


def _calcular_erro_oob(investigadores: list, X: list, y: list) -> float:
    """
    Erro out-of-bag do comitê: para cada exemplo, agrega só o voto dos
    investigadores que NÃO o tinham na própria amostra bootstrap, e
    compara com o rótulo verdadeiro. Exemplos que por azar entraram na
    amostra de TODOS os investigadores do comitê (raro, mas possível com
    poucos investigadores) ficam de fora da conta, por falta de voto.
    """
    erros, avaliados = 0, 0
    for i in range(len(X)):
        votos = [inv["modelo"].predict([X[i]])[0] for inv in investigadores if i in inv["indices_oob"]]
        if not votos:
            continue
        avaliados += 1
        if _votar_maioria(votos) != y[i]:
            erros += 1
    return erros / avaliados


def _simular_fracao_out_of_bag(n: int, tentativas: int, seed: int = 42) -> float:
    """Simula muitas amostras bootstrap de tamanho n e mede a fração média que fica out-of-bag."""
    rng = random.Random(seed)
    fracoes = []
    for _ in range(tentativas):
        indices_amostra = _sortear_indices_bootstrap(n, rng)
        fracoes.append(len(_indices_out_of_bag(indices_amostra, n)) / n)
    return sum(fracoes) / len(fracoes)


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: o júri de 16 suspeitos
# ---------------------------------------------------------------------------


def _mostrar_amostras_bootstrap(investigadores: list):
    """Mostra, pros primeiros investigadores, quem eles sortearam (com repetição) e quem ficou de fora."""
    print(f"\n--- Sorteio bootstrap dos {len(investigadores)} primeiros investigadores ---")
    for b, inv in enumerate(investigadores, start=1):
        nomes_amostra = [SUSPEITOS[i][0] for i in inv["indices_amostra"]]
        repetidos = {nome: c for nome, c in Counter(nomes_amostra).items() if c > 1}
        nomes_oob = [SUSPEITOS[i][0] for i in inv["indices_oob"]]
        print(f"\nInvestigador {b}: viu {len(set(inv['indices_amostra']))} suspeitos distintos")
        if repetidos:
            print(f"  repetiu no sorteio: {repetidos}")
        print(f"  ficaram de fora (out-of-bag): {nomes_oob}")


def _mostrar_regra_de_um_investigador(investigadores: list):
    """Imprime a regra aprendida pelo primeiro investigador, só pra lembrar que por baixo é uma árvore rasa."""
    print("\n--- Por baixo, cada investigador é uma arvorezinha rasa (profundidade 2) ---")
    print("Regra aprendida pelo Investigador 1, na amostra bootstrap dele:\n")
    print(export_text(investigadores[0]["modelo"], feature_names=["nervosismo", "contradições"]))


def _mostrar_votos_para_aluno_novo(investigadores: list):
    """Mostra o voto individual de cada investigador pro Aluno Novo, e o veredito por maioria."""
    nome, nervosismo, contradicoes = ALUNO_NOVO
    print(f"\n--- Julgando o {nome} (nervosismo={nervosismo}, contradições={contradicoes}) ---")
    votos = []
    for b, inv in enumerate(investigadores, start=1):
        voto = inv["modelo"].predict([[nervosismo, contradicoes]])[0]
        votos.append(voto)
        veredito = "Sim, culpado" if voto == 1 else "Não, inocente"
        print(f"  Investigador {b:>2}: vota '{veredito}'")

    contagem = Counter(votos)
    veredito_final = "Sim, culpado" if _votar_maioria(votos) == 1 else "Não, inocente"
    print(
        f"\nVotos: {contagem[1]} por 'culpado', {contagem[0]} por 'inocente'. "
        f"Cada investigador viu uma amostra diferente e alguns discordam entre si, "
        f"mas a turma inteira vota, e vence a maioria: veredito final = {veredito_final}."
    )
    return votos


def _mostrar_erro_out_of_bag(investigadores_completo: list, X: list, y: list):
    """Calcula o erro OOB do júri completo e mostra a fração empírica x teórica (1/e) de out-of-bag."""
    erro_oob = _calcular_erro_oob(investigadores_completo, X, y)
    print(f"\n--- Erro out-of-bag do júri completo ({len(investigadores_completo)} investigadores) ---")
    print(
        f"Erro OOB = {erro_oob:.1%}: usando, pra cada suspeito, só o voto de quem "
        "NÃO o tinha na própria amostra de treino. É uma estimativa de erro de "
        "generalização sem separar nenhum suspeito à parte pra validação."
    )

    n = len(X)
    fracao_empirica = _simular_fracao_out_of_bag(n, tentativas=20000)
    fracao_formula = (1 - 1 / n) ** n
    print(
        f"\nFração média de suspeitos out-of-bag por investigador (n={n}): "
        f"empírica={fracao_empirica:.3f}, fórmula (1-1/n)^n={fracao_formula:.3f}, "
        f"limite teórico 1/e={1 / 2.718281828:.3f} (quanto maior o n, mais perto do limite)."
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


def plotar_fronteiras_individuais_vs_comite(investigadores: list, caminho_saida: Path | None = None) -> Path:
    """
    Desenha, num mesmo mapa, a fronteira de decisão de cada investigador
    individual (linhas finas e cinzas, uma por amostra bootstrap) e a
    fronteira do comitê inteiro por voto majoritário (linha grossa preta):
    o "espaguete" de fronteiras instáveis que se resolve numa fronteira
    única e mais suave depois da agregação.
    """
    plt = _preparar_pyplot()
    import numpy as np

    X, y = _X_y()
    cores = {0: "#4C72B0", 1: "#C0392B"}
    xx, yy = np.meshgrid(np.linspace(0, 10, 300), np.linspace(0, 6, 300))
    grade = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots(figsize=(8, 6))
    votos_grade = []
    for inv in investigadores:
        zz = inv["modelo"].predict(grade).reshape(xx.shape)
        votos_grade.append(zz)
        ax.contour(xx, yy, zz, levels=[0.5], colors="gray", alpha=0.5, linewidths=1)

    zz_comite = (np.mean(votos_grade, axis=0) >= 0.5).astype(int)
    ax.contour(xx, yy, zz_comite, levels=[0.5], colors="black", linewidths=2.8)

    for classe, cor in cores.items():
        pontos = [(linha[1], linha[2]) for linha, rotulo in zip(SUSPEITOS, y) if rotulo == classe]
        xs, ys = zip(*pontos)
        rotulo_legenda = "Culpado" if classe == 1 else "Inocente"
        ax.scatter(xs, ys, color=cor, s=120, edgecolor="black", zorder=3, label=rotulo_legenda)

    ax.scatter(*ALUNO_NOVO[1:], color="gold", s=260, marker="*", edgecolor="black", zorder=4, label=ALUNO_NOVO[0])
    ax.set_xlabel("nervosismo")
    ax.set_ylabel("contradições no depoimento")
    ax.set_title(
        f"{len(investigadores)} fronteiras individuais (cinza) x fronteira do comitê (preta)"
    )
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "fronteiras_individuais_vs_comite.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_votos_investigador(votos: list, caminho_saida: Path | None = None) -> Path:
    """Barras com o voto de cada investigador pro Aluno Novo, e a linha de corte da maioria."""
    plt = _preparar_pyplot()

    # Barra nasce da linha de corte (0,5), não do zero: senão um voto "0"
    # vira uma barra de altura zero, invisível no gráfico.
    cores = ["#C0392B" if voto == 1 else "#4C72B0" for voto in votos]
    alturas = [voto - 0.5 for voto in votos]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(votos) + 1), alturas, bottom=0.5, color=cores, edgecolor="black")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="linha de corte da maioria")
    ax.set_xlabel("investigador")
    ax.set_ylabel("voto (0 = inocente, 1 = culpado)")
    ax.set_yticks([0, 1])
    ax.set_title(f"Voto de cada investigador pro {ALUNO_NOVO[0]}")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "votos_aluno_novo.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_curva_estabilizacao(caminho_saida: Path | None = None, seeds_base=(0, 100, 200, 300, 400)) -> Path:
    """
    Erro out-of-bag médio em função de quantos investigadores já foram
    ouvidos, repetindo o sorteio do júri inteiro em 5 sementes-base
    diferentes e tirando a média das 5 curvas: uma única sequência de
    sorteios (só 16 suspeitos no exemplo de brinquedo) fica cheia de
    zigue-zague só por sorte de qual investigador entrou quando, então a
    média de vários júris independentes mostra a tendência de verdade
    sem depender de uma sequência de sementes específica.
    """
    plt = _preparar_pyplot()

    X, y = _X_y()
    tamanhos = list(range(1, B_TOTAL + 1, 2))
    curvas = []
    for seed_base in seeds_base:
        investigadores = treinar_investigadores(X, y, B_TOTAL, seed_base=seed_base)
        curvas.append([_calcular_erro_oob(investigadores[:k], X, y) for k in tamanhos])
    erros_medios = [sum(pontos) / len(pontos) for pontos in zip(*curvas)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curva in curvas:
        ax.plot(tamanhos, curva, color="#4C72B0", alpha=0.25, linewidth=1)
    ax.plot(tamanhos, erros_medios, marker="o", color="#4C72B0", linewidth=2.5, label="média de 5 júris")
    ax.set_xlabel("número de investigadores no júri")
    ax.set_ylabel("erro out-of-bag")
    ax.set_title("Erro OOB conforme o júri cresce: cada júri (fino) oscila, a média (grossa) se assenta")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "curva_estabilizacao.png"
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
    Refaz à mão, com o júri de 16 suspeitos, a receita escondida atrás do
    bagging: sorteio bootstrap, treino de vários investigadores rasos,
    votos divergindo, agregação por maioria e erro out-of-bag.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine um julgamento de classe: 16 suspeitos, cada um com dois "
        "atributos (nervosismo e número de contradições no depoimento), e "
        "vários investigadores que só enxergam uma amostra sorteada com "
        "reposição do baralho de suspeitos, podendo repetir suspeito e "
        "deixar outros de fora."
    )

    X, y = _X_y()
    investigadores_exibicao = treinar_investigadores(X, y, B_EXIBICAO, seed_base=0)
    _mostrar_amostras_bootstrap(investigadores_exibicao)
    _mostrar_regra_de_um_investigador(investigadores_exibicao)
    votos = _mostrar_votos_para_aluno_novo(investigadores_exibicao)

    investigadores_completo = treinar_investigadores(X, y, B_TOTAL, seed_base=0)
    _mostrar_erro_out_of_bag(investigadores_completo, X, y)

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_fronteiras_individuais_vs_comite(investigadores_exibicao)
    plotar_votos_investigador(votos)
    plotar_curva_estabilizacao()
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _amostra_bootstrap_treino(X_train, y_train, semente: int) -> tuple:
    """Sorteia, com reposição, uma amostra bootstrap do treino inteiro: o que UM investigador sozinho veria."""
    combo = X_train.copy()
    combo["_alvo"] = y_train.values
    combo_boot = combo.sample(n=len(combo), replace=True, random_state=semente)
    return combo_boot.drop(columns="_alvo"), combo_boot["_alvo"]


def _comparar_variancia_single_vs_bagging(X_train, X_test, y_train, y_test, sementes=(0, 1, 2, 3, 4)) -> tuple:
    """
    Treina, pra cada semente, UMA árvore só numa amostra bootstrap do
    treino (o equivalente a um único investigador sozinho) e um comitê de
    bagging inteiro (que agrega vários investigadores), comparando o F1
    (classe fraude) de cada abordagem: o objetivo não é o melhor modelo
    isolado, é ver o quanto o resultado MUDA de semente pra semente, ou
    seja, o quanto cada abordagem depende da sorte de qual amostra caiu.
    """
    print("\n--- Um investigador sozinho x o comitê inteiro, em várias sementes aleatórias ---")
    f1_arvore, f1_bagging = [], []
    for semente in sementes:
        X_boot, y_boot = _amostra_bootstrap_treino(X_train, y_train, semente)
        arvore = DecisionTreeClassifier(random_state=semente).fit(X_boot, y_boot)
        f1_a = f1_score(y_test, arvore.predict(X_test), pos_label=1)

        comite = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=semente),
            n_estimators=15,
            random_state=semente,
            n_jobs=-1,
        ).fit(X_train, y_train)
        f1_b = f1_score(y_test, comite.predict(X_test), pos_label=1)

        f1_arvore.append(f1_a)
        f1_bagging.append(f1_b)
        print(f"  semente={semente}: F1 investigador sozinho={f1_a:.4f}   F1 comitê (15 árvores)={f1_b:.4f}")

    media_a, media_b = sum(f1_arvore) / len(f1_arvore), sum(f1_bagging) / len(f1_bagging)
    desvio_a = (sum((v - media_a) ** 2 for v in f1_arvore) / len(f1_arvore)) ** 0.5
    desvio_b = (sum((v - media_b) ** 2 for v in f1_bagging) / len(f1_bagging)) ** 0.5
    print(
        f"\nInvestigador sozinho:  média F1={media_a:.4f}  desvio-padrão={desvio_a:.4f}\n"
        f"Comitê (bagging):      média F1={media_b:.4f}  desvio-padrão={desvio_b:.4f}\n"
        "O desvio-padrão do comitê entre sementes é bem menor: é a variância "
        "caindo na prática, o mesmo efeito visto na curva de estabilização do "
        "exemplo de brinquedo, agora com dados de verdade. Repare que a "
        "comparação não é 'árvore com semente diferente na mesma base', que "
        "quase não muda nada com atributos contínuos (a árvore de decisão "
        "quase não tem empate pra desempatar); a instabilidade de verdade "
        "vem de treinar em amostras bootstrap DIFERENTES, exatamente o que "
        "cada investigador do comitê vive sozinho."
    )
    return f1_arvore, f1_bagging


def _treinar_bagging_com_oob(X_train, X_test, y_train, y_test, n_estimators: int = 100) -> BaggingClassifier:
    """Treina o comitê de bagging final, comparando a acurácia OOB (de graça) com a acurácia real de teste."""
    modelo = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_estimators,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acuracia_teste = modelo.score(X_test, y_test)

    print(f"\n--- BaggingClassifier final, {n_estimators} árvores ---")
    print(
        f"Acurácia estimada por OOB (sem tocar no conjunto de teste) = {modelo.oob_score_:.4f}\n"
        f"Acurácia medida de fato no conjunto de teste                = {acuracia_teste:.4f}\n"
        "As duas ficam bem próximas: o erro OOB realmente funciona como uma "
        "prévia gratuita do desempenho em dados novos."
    )
    print(classification_report(y_test, y_pred, digits=4))
    return modelo


def plotar_comparacao_variancia(f1_arvore: list, f1_bagging: list, caminho_saida: Path | None = None) -> Path:
    """Dispersão do F1 (classe fraude) por semente, investigador sozinho x comitê: mostra a nuvem de pontos do comitê bem mais apertada."""
    plt = _preparar_pyplot()

    fig, ax = plt.subplots(figsize=(7, 5))
    sementes = list(range(len(f1_arvore)))
    ax.scatter(sementes, f1_arvore, color="#C0392B", s=100, label="investigador sozinho", zorder=3)
    ax.scatter(sementes, f1_bagging, color="#4C72B0", s=100, label="comitê (bagging, 15 árvores)", zorder=3)
    ax.axhline(sum(f1_arvore) / len(f1_arvore), color="#C0392B", linestyle="--", alpha=0.5)
    ax.axhline(sum(f1_bagging) / len(f1_bagging), color="#4C72B0", linestyle="--", alpha=0.5)
    ax.set_xticks(sementes)
    ax.set_xlabel("semente aleatória")
    ax.set_ylabel("F1 (classe fraude)")
    ax.set_title("F1 por semente: o comitê oscila bem menos que um investigador sozinho")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "comparacao_variancia.png"
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
    f1_arvore, f1_bagging = _comparar_variancia_single_vs_bagging(X_train, X_test, y_train, y_test)
    plotar_comparacao_variancia(f1_arvore, f1_bagging)

    _titulo("COMITÊ FINAL: erro out-of-bag x erro de teste de verdade")
    _treinar_bagging_com_oob(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
