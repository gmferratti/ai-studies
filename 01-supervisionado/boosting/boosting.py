"""
Boosting (AdaBoost) aplicado ao dataset de detecção de fraude.

Teoria completa (peso de exemplo, erro ponderado, peso de voto do
estimador, atualização dos pesos, contraste com bagging) está em
`notes/anotacoes.md`.

Este arquivo tem duas partes:
  1) Exemplo de brincadeira com um torneio de 10 lutadores, refazendo na
     mão as contas do AdaBoost rodada a rodada: peso ponderado, erro,
     peso de voto de cada especialista, e o peso dos lutadores mais
     difíceis crescendo a cada rodada, pra sentir o algoritmo corrigindo
     seus próprios erros antes de ver isso rodando em cima de dados de
     verdade.
  2) O treino de verdade, no dataset de fraude de cartão de crédito,
     incluindo o desempenho do comitê em função do número de rodadas.

Rode o arquivo e leia o terminal de cima pra baixo.
"""

import math
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 10
# lutadores tentando montar a estratégia certa contra o campeão invicto,
# com dois atributos (força e velocidade) e o resultado real (Vence = +1,
# Perde = -1). Nenhum corte reto num atributo só separa os dois grupos
# direito: é justamente essa dificuldade que faz falta uma sequência de
# especialistas, cada um corrigindo o que o anterior errou.
LUTADORES = [
    # (nome, força, velocidade, resultado)
    ("Kael", 20, 30, "Perde"),
    ("Rho", 25, 65, "Perde"),
    ("Tam", 55, 35, "Perde"),
    ("Iva", 35, 55, "Perde"),
    ("Gwen", 58, 58, "Perde"),
    ("Hale", 45, 45, "Perde"),
    ("Bron", 60, 80, "Vence"),
    ("Doran", 70, 75, "Vence"),
    ("Elka", 65, 60, "Vence"),
    ("Fenn", 38, 82, "Vence"),
]

# Desafiante novo, caído numa zona ambígua do mapa (força e velocidade
# medianas): é aí que os especialistas mais discordam entre si.
DESAFIANTE = ("Desafiante", 50, 63)

ROTULO_NUMERICO = {"Perde": -1, "Vence": 1}
N_RODADAS = 3


def _X_y():
    """Extrai X (força, velocidade) e y (+1/-1) de LUTADORES."""
    X = [(linha[1], linha[2]) for linha in LUTADORES]
    y = [ROTULO_NUMERICO[linha[3]] for linha in LUTADORES]
    return X, y


# ---------------------------------------------------------------------------
# A matemática do AdaBoost, na mão
# ---------------------------------------------------------------------------


def _erro_ponderado(y: list, preds: list, pesos: list) -> float:
    """Fração do PESO total (não da contagem crua) que o estimador errou."""
    erro = sum(peso for peso, real, pred in zip(pesos, y, preds) if real != pred)
    return erro / sum(pesos)


def _calcular_alpha(erro: float) -> float:
    """Peso de voto do estimador: alfa = (1/2) ln((1-erro)/erro)."""
    erro_seguro = min(max(erro, 1e-10), 1 - 1e-10)  # evita log(0) se o estimador acertar tudo
    return 0.5 * math.log((1 - erro_seguro) / erro_seguro)


def _atualizar_pesos(pesos: list, y: list, preds: list, alpha: float) -> list:
    """Sobe o peso de quem foi errado, desce o peso de quem foi acertado, e normaliza pra somar 1."""
    novos = [peso * math.exp(-alpha * real * pred) for peso, real, pred in zip(pesos, y, preds)]
    soma = sum(novos)
    return [peso / soma for peso in novos]


def treinar_adaboost_manual(X: list, y: list, n_rodadas: int) -> list:
    """
    Roda o AdaBoost à mão por n_rodadas: a cada rodada, treina um stump
    (profundidade 1) ponderado pelos pesos atuais, mede o erro ponderado,
    calcula o peso de voto do estimador e atualiza os pesos pra próxima
    rodada. Guarda tudo (estimador, alfa, pesos usados) pra inspeção.
    """
    pesos = [1 / len(X)] * len(X)
    rodadas = []
    for _ in range(n_rodadas):
        modelo = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=pesos)
        preds = list(modelo.predict(X))
        erro = _erro_ponderado(y, preds, pesos)
        alpha = _calcular_alpha(erro)
        rodadas.append({"modelo": modelo, "alpha": alpha, "erro": erro, "pesos": list(pesos), "preds": preds})
        pesos = _atualizar_pesos(pesos, y, preds, alpha)
    return rodadas


def _prever_combinado(rodadas: list, x) -> int:
    """Previsão final: sinal da soma dos votos de cada estimador, ponderados por alfa."""
    soma = sum(r["alpha"] * r["modelo"].predict([x])[0] for r in rodadas)
    return 1 if soma >= 0 else -1


def _erro_cumulativo_por_rodada(rodadas: list, X: list, y: list) -> list:
    """Erro do comitê (só com os primeiros t estimadores) em função de t, de 1 até o total de rodadas."""
    erros = []
    for t in range(1, len(rodadas) + 1):
        preds_finais = [_prever_combinado(rodadas[:t], x) for x in X]
        erro = sum(1 for real, pred in zip(y, preds_finais) if real != pred) / len(y)
        erros.append(erro)
    return erros


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: o torneio de 10 lutadores
# ---------------------------------------------------------------------------


def _mostrar_rodada(t: int, rodada: dict, y: list):
    """Mostra a regra aprendida, quem foi errado, o erro ponderado e o peso de voto de uma rodada."""
    print(f"\n--- Rodada {t}: especialista focado no que sobrou difícil ---")
    tabela_pesos = ", ".join(f"{LUTADORES[i][0]}={rodada['pesos'][i]:.3f}" for i in range(len(y)))
    print(f"Pesos usados pra treinar esta rodada: {tabela_pesos}")
    print(export_text(rodada["modelo"], feature_names=["força", "velocidade"]))

    nomes_errados = [LUTADORES[i][0] for i in range(len(y)) if rodada["preds"][i] != y[i]]
    print(f"Errou: {nomes_errados}")
    print(f"Erro ponderado ε_{t} = {rodada['erro']:.4f}")
    print(f"Peso de voto α_{t} = {rodada['alpha']:.4f}")


def _mostrar_previsao_desafiante(rodadas: list):
    """Mostra o voto (ponderado por alfa) de cada especialista pro Desafiante, e o veredito final."""
    nome, forca, velocidade = DESAFIANTE
    print(f"\n--- Prevendo o {nome} (força={forca}, velocidade={velocidade}) ---")
    soma = 0.0
    for t, rodada in enumerate(rodadas, start=1):
        voto = rodada["modelo"].predict([[forca, velocidade]])[0]
        contribuicao = rodada["alpha"] * voto
        soma += contribuicao
        veredito = "Vence" if voto == 1 else "Perde"
        print(f"  Rodada {t}: especialista vota '{veredito}'  (α_{t}={rodada['alpha']:.4f} de peso no voto)")

    veredito_final = "Vence" if soma >= 0 else "Perde"
    print(f"\nSoma ponderada dos votos = {soma:.4f}  ->  veredito final = {veredito_final}")


def _mostrar_erro_cumulativo(rodadas: list, X: list, y: list):
    """Mostra como o erro do comitê (no próprio torneio) cai conforme mais rodadas entram."""
    erros = _erro_cumulativo_por_rodada(rodadas, X, y)
    print("\n--- Erro do comitê conforme as rodadas se acumulam ---")
    for t, erro in enumerate(erros, start=1):
        print(f"  com {t} especialista(s): erro = {erro:.1%}")
    return erros


# ---------------------------------------------------------------------------
# Gráficos (salvos em images/, dentro da pasta deste script)
# ---------------------------------------------------------------------------


def _preparar_pyplot():
    """Configura o backend sem interface gráfica (necessário em servidor/terminal) e devolve o pyplot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plotar_evolucao_pesos(rodadas: list, caminho_saida: Path | None = None) -> Path:
    """
    Um painel por rodada, com o tamanho de cada lutador proporcional ao
    peso que ele tinha NAQUELA rodada: mostra visualmente o peso indo
    parar nos lutadores que os especialistas anteriores erraram.
    """
    plt = _preparar_pyplot()

    X, y = _X_y()
    cores = {-1: "#4C72B0", 1: "#C0392B"}
    fig, eixos = plt.subplots(1, len(rodadas), figsize=(6 * len(rodadas), 5), sharey=True)
    for t, (ax, rodada) in enumerate(zip(eixos, rodadas), start=1):
        for i, (forca, velocidade) in enumerate(X):
            ax.scatter(
                forca, velocidade, color=cores[y[i]], s=rodada["pesos"][i] * 4000,
                edgecolor="black", zorder=3, alpha=0.85,
            )
            ax.annotate(LUTADORES[i][0], (forca, velocidade), fontsize=7, ha="center", va="center")
        ax.set_title(f"Pesos usados na rodada {t}\n(ε_{t}={rodada['erro']:.3f}, α_{t}={rodada['alpha']:.3f})")
        ax.set_xlabel("força")
    eixos[0].set_ylabel("velocidade")
    fig.suptitle("Tamanho da bolinha = peso do lutador naquela rodada")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "evolucao_pesos.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_fronteiras_boosting(rodadas: list, caminho_saida: Path | None = None) -> Path:
    """
    Desenha a fronteira de cada especialista (uma reta simples, cor por
    rodada) e a fronteira do comitê ponderado inteiro (linha grossa
    preta): mostra como somar retas simples, cada uma focada num pedaço
    diferente do problema, produz uma fronteira final bem mais elaborada.
    """
    plt = _preparar_pyplot()
    import numpy as np

    X, y = _X_y()
    cores_classe = {-1: "#4C72B0", 1: "#C0392B"}
    cores_rodada = ["#8DA0CB", "#FC8D62", "#66C2A5"]
    xx, yy = np.meshgrid(np.linspace(0, 90, 300), np.linspace(0, 100, 300))
    grade = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots(figsize=(8, 6))
    for t, (rodada, cor) in enumerate(zip(rodadas, cores_rodada), start=1):
        zz = rodada["modelo"].predict(grade).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=[0], colors=cor, linewidths=1.8, linestyles="dashed")
        ax.plot([], [], color=cor, linestyle="dashed", label=f"especialista da rodada {t}")

    zz_comite = np.array([_prever_combinado(rodadas, ponto) for ponto in grade]).reshape(xx.shape)
    ax.contour(xx, yy, zz_comite, levels=[0], colors="black", linewidths=2.8)
    ax.plot([], [], color="black", linewidth=2.8, label="comitê ponderado (final)")

    for classe, cor in cores_classe.items():
        pontos = [(linha[1], linha[2]) for linha, rotulo in zip(LUTADORES, y) if rotulo == classe]
        xs, ys = zip(*pontos)
        rotulo_legenda = "Vence" if classe == 1 else "Perde"
        ax.scatter(xs, ys, color=cor, s=120, edgecolor="black", zorder=3, label=rotulo_legenda)

    ax.scatter(*DESAFIANTE[1:], color="gold", s=260, marker="*", edgecolor="black", zorder=4, label=DESAFIANTE[0])
    ax.set_xlabel("força")
    ax.set_ylabel("velocidade")
    ax.set_title("Especialistas individuais (tracejado) x comitê ponderado (preto)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "fronteiras_boosting.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_erro_por_rodada(erros: list, caminho_saida: Path | None = None) -> Path:
    """Erro do comitê no torneio de brinquedo, conforme mais especialistas entram na votação."""
    plt = _preparar_pyplot()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(erros) + 1), erros, marker="o", color="#4C72B0", linewidth=2)
    ax.set_xlabel("número de especialistas no comitê")
    ax.set_ylabel("erro (no próprio torneio)")
    ax.set_xticks(range(1, len(erros) + 1))
    ax.set_title("Erro caindo a cada especialista que entra, focado no que sobrou difícil")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "erro_por_rodada.png"
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
    Refaz à mão, com o torneio de 10 lutadores, a receita escondida atrás
    do AdaBoost: treino sequencial de especialistas fracos, erro
    ponderado, peso de voto de cada um, e peso dos exemplos difíceis
    crescendo rodada a rodada.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine uma escola de artes marciais tentando montar a estratégia "
        "certa contra o campeão invicto: manda um especialista fraco por "
        "vez pra treinar, e cada novo especialista foca justamente nos "
        "golpes que os anteriores não conseguiram resolver."
    )

    X, y = _X_y()
    rodadas = treinar_adaboost_manual(X, y, N_RODADAS)
    for t, rodada in enumerate(rodadas, start=1):
        _mostrar_rodada(t, rodada, y)

    _mostrar_previsao_desafiante(rodadas)
    erros = _mostrar_erro_cumulativo(rodadas, X, y)

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_evolucao_pesos(rodadas)
    plotar_fronteiras_boosting(rodadas)
    plotar_erro_por_rodada(erros)
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _treinar_adaboost_final(X_train, X_test, y_train, y_test, n_estimators: int = 150) -> AdaBoostClassifier:
    """Treina o AdaBoost final, com stumps de profundidade 1 como especialista fraco."""
    modelo = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(f"\n--- AdaBoostClassifier final, {n_estimators} rodadas (stumps de profundidade 1) ---")
    print(classification_report(y_test, y_pred, digits=4))
    return modelo


def _curva_f1_treino_teste(modelo: AdaBoostClassifier, X_train, y_train, X_test, y_test) -> tuple:
    """F1 (classe fraude) em treino e teste, rodada a rodada, usando o comitê parcial de cada estágio."""
    # staged_predict, por baixo dos panos, repassa X pros estimadores internos
    # sem preservar os nomes das colunas: um warning de sklearn sem
    # consequência prática (a ordem das colunas continua a mesma do fit).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        f1_treino = [f1_score(y_train, pred, pos_label=1) for pred in modelo.staged_predict(X_train)]
        f1_teste = [f1_score(y_test, pred, pos_label=1) for pred in modelo.staged_predict(X_test)]
    print(
        f"\n--- F1 (classe fraude) por rodada: 1ª rodada treino={f1_treino[0]:.4f} teste={f1_teste[0]:.4f} | "
        f"última rodada treino={f1_treino[-1]:.4f} teste={f1_teste[-1]:.4f} ---"
    )
    print(
        "Um stump sozinho (1ª rodada) rende um F1 baixo, como esperado de um "
        "estimador fraco. O F1 sobe rápido nas primeiras rodadas e depois "
        "desacelera; se a curva de treino continuar subindo enquanto a de "
        "teste estaciona ou cai, é a assinatura de overfitting do boosting."
    )
    return f1_treino, f1_teste


def plotar_curva_f1_rodada(f1_treino: list, f1_teste: list, caminho_saida: Path | None = None) -> Path:
    """F1 (classe fraude) em treino e teste, em função do número de rodadas do AdaBoost."""
    plt = _preparar_pyplot()

    fig, ax = plt.subplots(figsize=(8, 5))
    rodadas = range(1, len(f1_treino) + 1)
    ax.plot(rodadas, f1_treino, color="#C0392B", label="treino")
    ax.plot(rodadas, f1_teste, color="#4C72B0", label="teste")
    ax.set_xlabel("número de rodadas (n_estimators)")
    ax.set_ylabel("F1 (classe fraude)")
    ax.set_title("F1 conforme o comitê cresce: treino x teste, de olho no overfitting")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "curva_f1_rodada.png"
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
    modelo = _treinar_adaboost_final(X_train, X_test, y_train, y_test)
    f1_treino, f1_teste = _curva_f1_treino_teste(modelo, X_train, y_train, X_test, y_test)
    plotar_curva_f1_rodada(f1_treino, f1_teste)


if __name__ == "__main__":
    main()
