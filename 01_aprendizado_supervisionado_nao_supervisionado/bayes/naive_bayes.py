"""
Naive Bayes aplicado ao dataset de detecção de fraude.

Conceitos revisados: Teorema de Bayes, suposição de independência
condicional ("ingenuidade"), critério MAP, o problema do produto de zero
e a suavização de Laplace, as variantes Gaussian, Multinomial e Bernoulli.

Este arquivo tem duas partes:
  1) Uma aula "mastigada", com analogia de sistema anti-cheat de jogo
     online e um exemplo de brincadeira detectando bot numa partida, pra
     entender a ideia ANTES de ver a matemática rodando em cima de dados
     de verdade. Não exige conhecimento prévio de programação ou
     estatística.
  2) O treino de verdade, no dataset de fraude de cartão de crédito.

Rode o arquivo e leia o terminal de cima pra baixo.

---------------------------------------------------------------------------
COMO FUNCIONA O NAIVE BAYES
---------------------------------------------------------------------------

PENSE NUM SISTEMA ANTI-CHEAT DE JOGO ONLINE.

Ele não segue um fluxograma de perguntas tipo "jogou mais de 20 horas
seguidas? então É bot" (isso seria mais parecido com uma árvore de
decisão). Ele faz outra coisa: junta várias pistas de comportamento
(tempo de reação, horário que joga, tipo de mensagem no chat) e calcula
uma PROBABILIDADE de a conta ser bot, combinando cada pista como se ela
não tivesse nenhuma relação com as outras. No fim, escolhe o rótulo (bot
ou humano) que ficou com a maior probabilidade. É esse "combinar pistas
como se fossem independentes" que dá nome ao algoritmo: ele é "ingênuo"
(naive) de propósito.

No vocabulário "oficial":

    - Classe          -> o rótulo que se quer prever (bot ou humano).
    - Atributo (pista) -> uma característica observável da conta (reação
      robótica? online 24h seguidas? chat só com frases prontas?).
    - Prior            -> a chance de ser bot ANTES de olhar pra qualquer
      pista.
    - Verossimilhança  -> a chance de observar uma pista específica, SE a
      conta já fosse bot (ou já fosse humana).
    - Posterior        -> a chance de ser bot DEPOIS de já ter olhado as
      pistas, o que o algoritmo realmente quer calcular.

1) TEOREMA DE BAYES: A IDEIA CENTRAL

   O nome do algoritmo vem do Teorema de Bayes, que relaciona duas
   probabilidades "de trás pra frente":

       P(classe | pista) = P(pista | classe) * P(classe) / P(pista)

   Traduzindo cada pedaço com o exemplo do anti-cheat:

       - P(bot | pista)  (posterior): o que se quer saber, a chance de
         ser bot depois de observar a pista.
       - P(pista | bot)  (verossimilhança): entre as contas que JÁ SE
         SABE que são bot (histórico analisado por um moderador), quantas
         tinham essa pista.
       - P(bot)          (prior): a chance de qualquer conta aleatória
         ser bot, sem olhar pra nenhuma pista. Se 3 em cada 10 contas
         analisadas eram bot, P(bot) = 0,3.
       - P(pista)        (evidência): a chance de ver essa pista somando
         bot e humano juntos. Funciona só como "normalizador", pra as
         probabilidades de todas as classes somarem 1, e por isso, como
         vamos ver adiante, dá pra ignorar ela na hora de decidir.

   A ideia chave: a probabilidade de ser bot não depende só de quão
   suspeita é a pista (verossimilhança), depende também de quão comum já
   era ser bot ANTES de qualquer pista (o prior). Se bot for raríssimo no
   servidor (prior bem baixo), mesmo uma pista bem suspeita não é
   necessariamente o bastante pra concluir "é bot com certeza": é a mesma
   lógica de "exame médico raro dá falso positivo raro, mesmo sendo um
   bom exame".

2) A SUPOSIÇÃO "INGÊNUA": INDEPENDÊNCIA CONDICIONAL

   Na vida real, jogar 24 horas seguidas e ter reação robótica não são
   pistas totalmente soltas uma da outra: quem joga sem parar tende a
   ficar cansado e reagir PIOR, não igual todo santo dia. Mesmo assim, o
   Naive Bayes finge que, sabendo já se a conta é bot ou não, uma pista
   não diz NADA sobre a outra. Essa suposição quase nunca é 100%
   verdadeira, mas o algoritmo funciona bem mesmo assim, porque pra
   decidir o rótulo só importa QUAL classe fica com a maior probabilidade
   no fim, não o valor exato dela. Mesmo que o número saia meio torto pela
   suposição errada, a ORDEM entre "chance de ser bot" e "chance de ser
   humano" costuma se manter certa.

3) VÁRIAS PISTAS AO MESMO TEMPO: O PRODUTÓRIO E O CRITÉRIO MAP

   Na prática, uma conta tem várias pistas ao mesmo tempo, não só uma.
   Com a suposição de independência, a verossimilhança conjunta de todas
   as pistas vira simplesmente o PRODUTO das verossimilhanças de cada
   pista isolada:

       P(bot | pistas) ∝ P(bot) * P(pista1|bot) * P(pista2|bot) * P(pista3|bot)

   O símbolo ∝ quer dizer "proporcional a": dá pra ignorar o denominador
   P(pistas) do Teorema de Bayes, porque ele é igual pra bot e pra humano
   (não muda qual classe vence a comparação), então só interessa comparar
   os numeradores.

   Calcula-se esse produto pra "bot" e pra "humano" e escolhe-se o maior.
   Esse critério tem nome, MAP (Maximum A Posteriori, "máximo a
   posteriori"): maximiza a probabilidade calculada DEPOIS de olhar as
   pistas, em contraste com maximizar só a verossimilhança sozinha,
   ignorando o prior.

4) O PROBLEMA DO PRODUTO DE ZERO E A SUAVIZAÇÃO DE LAPLACE

   E se uma pista nunca apareceu junto de "bot" em nenhuma conta do
   histórico analisado? Então P(pista|bot) = 0, e como é um produtório,
   isso zera a chance de ser bot inteira, não importa quão suspeitas sejam
   as OUTRAS pistas. Uma única combinação nunca vista antes derruba a
   conclusão inteira pra zero, mesmo que todo o resto grite "bot".

   A correção clássica é a suavização de Laplace: em vez de contar direto,
   soma-se 1 em cada contagem antes de dividir:

       P(pista|classe) = (contagem(pista, classe) + 1) / (total(classe) + k)

   onde k é o número de valores possíveis daquela pista (2, no caso de uma
   pista Sim/Não). Isso garante que nenhuma probabilidade fica exatamente
   zero, então uma pista rara deixa de ter poder de veto total sobre as
   demais.

5) AS VARIANTES: O QUE MUDA É SÓ COMO P(pista|classe) É CALCULADO

   - BernoulliNB: pista é presença/ausência (tipo as do nosso anti-cheat),
     calculada por contagem direta, exatamente como fizemos acima.
   - MultinomialNB: pista é uma CONTAGEM (quantas vezes uma palavra
     aparece num texto, por exemplo), clássico em filtro de spam.
   - GaussianNB: pista é um NÚMERO CONTÍNUO (tipo valor de uma compra em
     reais). Em vez de contar quantas vezes um valor exato apareceu (o que
     quase nunca se repete pra número real), assume-se que os valores
     seguem uma curva de sino (distribuição normal) dentro de cada classe,
     usando a média e o desvio padrão da própria classe pra calcular a
     "densidade" naquele ponto. Quanto mais perto da média da classe,
     maior a densidade, e vice-versa.

   O dataset de fraude do módulo 01 tem atributos numéricos contínuos
   (valor da compra, componentes de PCA), então a Parte 3 usa GaussianNB.

   Detalhe de implementação: multiplicar dezenas de números pequenos entre
   0 e 1 pode fazer o resultado ficar tão perto de zero que o computador
   arredonda pra zero (erro de underflow). Por isso, na prática, os
   algoritmos trabalham com o LOGARITMO das probabilidades, transformando
   o produtório numa soma, numericamente mais estável e com o mesmo
   resultado na comparação final entre classes.

6) VANTAGENS x DESVANTAGENS
   (+) Rápido de treinar e de prever: só calcula frequências, médias e
       desvios padrão, sem otimização iterativa.
   (+) Funciona bem com poucos dados de treino.
   (+) Lida naturalmente com muitos atributos (tipo milhares de palavras
       num filtro de spam), porque a suposição de independência evita
       calcular probabilidades conjuntas complexas demais pra qualquer
       dataset cobrir.
   (+) Boa baseline: mesmo com a suposição "ingênua" claramente errada em
       muitos casos reais, costuma ter desempenho competitivo.
   (-) A suposição de independência condicional raramente é verdadeira,
       prejudica a qualidade das probabilidades estimadas (mesmo que a
       classificação final ainda saia certa na maioria das vezes).
   (-) Sensível a pistas redundantes: duas pistas correlacionadas são
       contadas como se fossem duas evidências independentes, o que pode
       enviesar a decisão numa direção sem motivo real.
   (-) GaussianNB assume distribuição normal dos atributos contínuos, que
       pode não bater com a distribuição real dos dados.
---------------------------------------------------------------------------
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from utils.data_utils import build_preprocessing_pipeline, get_train_test_split, load_raw_data

IMAGES_DIR = Path(__file__).resolve().parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Exemplo de brincadeira, sem nenhuma relação com o dataset de fraude: 10
# contas de um servidor de jogo online, já analisadas manualmente por um
# moderador (3 bots, 7 humanos), com 3 pistas de comportamento binárias
# (Sim/Não). Só pra "sentir" prior, verossimilhança e o critério MAP na
# mão, com mais de uma pista em jogo.
DATASET_CONTAS = [
    # (nome, reacao_robotica, online_24h, chat_padrao, rotulo)
    ("BotFarm_XX9", "Sim", "Sim", "Sim", "bot"),
    ("ScriptKiddie404", "Sim", "Sim", "Não", "bot"),
    ("MacroLordZR", "Sim", "Não", "Sim", "bot"),
    ("LagadoDoWifi", "Não", "Não", "Não", "humano"),
    ("NoobMaster69", "Não", "Não", "Não", "humano"),
    ("CansadoDePlantao", "Não", "Sim", "Não", "humano"),
    ("InsoniaGamer", "Não", "Não", "Sim", "humano"),
    ("FarmeiroDeXP", "Não", "Sim", "Sim", "humano"),
    ("ChefinDoRole", "Não", "Não", "Não", "humano"),
    ("ReflexoDeOnca", "Sim", "Não", "Não", "humano"),
]

# Índice de cada atributo dentro de uma linha de DATASET_CONTAS (0 = nome, 4 = rótulo).
ATRIBUTOS = ("reacao_robotica", "online_24h", "chat_padrao")


# ---------------------------------------------------------------------------
# Matemática de Bayes (prior, verossimilhança, posterior proporcional, MAP)
# ---------------------------------------------------------------------------


def _filtrar_contas(rotulo: str | None = None) -> list:
    """Filtra DATASET_CONTAS por rótulo (bot/humano), ou devolve tudo se rotulo=None."""
    if rotulo is None:
        return DATASET_CONTAS
    return [linha for linha in DATASET_CONTAS if linha[4] == rotulo]


def _valores_atributo(indice: int, rotulo: str | None = None) -> list:
    """Lista os valores (Sim/Não) do atributo `indice`, dentro do subgrupo filtrado por rótulo."""
    return [linha[indice] for linha in _filtrar_contas(rotulo)]


def priori(rotulo: str) -> float:
    """P(classe): fração das contas do histórico que têm esse rótulo."""
    return len(_filtrar_contas(rotulo)) / len(DATASET_CONTAS)


def verossimilhanca(indice_atributo: int, valor: str, rotulo: str, suavizar: bool = False) -> float:
    """
    P(pista|classe): fração das contas da classe `rotulo` que têm `valor`
    no atributo de índice `indice_atributo`. Com suavizar=True, aplica a
    suavização de Laplace (soma 1 na contagem, soma k=2 no total, já que
    cada pista aqui só tem 2 valores possíveis, Sim ou Não).
    """
    valores_classe = _valores_atributo(indice_atributo, rotulo)
    contagem = valores_classe.count(valor)
    total = len(valores_classe)
    if suavizar:
        return (contagem + 1) / (total + 2)
    return contagem / total


def posterior_proporcional(caso: tuple, rotulo: str, suavizar: bool = False) -> float:
    """
    P(classe) * produtório de P(pista_i|classe): proporcional ao
    posterior de verdade (falta dividir pela evidência P(pistas), que é
    igual pras duas classes e não muda qual delas vence).
    """
    produto = priori(rotulo)
    for indice, valor in enumerate(caso, start=1):
        produto *= verossimilhanca(indice, valor, rotulo, suavizar)
    return produto


def classificar(caso: tuple, suavizar: bool = False) -> tuple:
    """Aplica o critério MAP: calcula o posterior proporcional pra bot e humano e devolve o vencedor."""
    posteriores = {rotulo: posterior_proporcional(caso, rotulo, suavizar) for rotulo in ("bot", "humano")}
    vencedora = max(posteriores, key=posteriores.get)
    return vencedora, posteriores


# ---------------------------------------------------------------------------
# Exemplo de brincadeira: detector de bot
# ---------------------------------------------------------------------------


def _explicar_teorema_com_uma_pista():
    """Aquecimento: aplica o Teorema de Bayes com uma pista só (online 24h seguidas), nomeando os 4 termos."""
    print("\n--- Aquecimento: Teorema de Bayes com uma pista só (online 24h seguidas) ---")
    p_bot = priori("bot")
    p_online_dado_bot = verossimilhanca(2, "Sim", "bot")
    p_online = _valores_atributo(2).count("Sim") / len(DATASET_CONTAS)
    posterior = p_online_dado_bot * p_bot / p_online

    print(f"P(bot) = {p_bot:.3f}  (prior: fração de bots no histórico)")
    print(f"P(online 24h = Sim | bot) = {p_online_dado_bot:.3f}  (verossimilhança)")
    print(f"P(online 24h = Sim) = {p_online:.3f}  (evidência, bot e humano juntos)")
    print(
        f"P(bot | online 24h = Sim) = {p_online_dado_bot:.3f} * {p_bot:.3f} / {p_online:.3f} "
        f"= {posterior:.3f}"
    )
    print(
        f"\nOu seja, {posterior:.0%} de chance de ser bot só com essa pista. Repare que "
        f"mesmo online 24h sendo bem mais comum entre bots do que entre humanos, o "
        "resultado não vira 100%: o prior de bot já era baixo pra começar, e isso pesa "
        "na conta."
    )


def _imprimir_tabela_verossimilhancas():
    """Mostra, lado a lado, a verossimilhança bruta de cada pista em bot e em humano."""
    print("\n--- Verossimilhanças brutas: P(pista = Sim | classe) ---")
    for indice, nome_atributo in enumerate(ATRIBUTOS, start=1):
        p_bot = verossimilhanca(indice, "Sim", "bot")
        p_humano = verossimilhanca(indice, "Sim", "humano")
        print(f"  {nome_atributo:<18} bot={p_bot:.3f}   humano={p_humano:.3f}")


def _classificar_caso_suspeito():
    """Classifica uma conta nova com duas pistas fortes de bot e uma neutra, mostrando o produtório completo."""
    print("\n--- Classificando uma conta nova: reação robótica=Sim, online 24h=Sim, chat pronto=Não ---")
    caso = ("Sim", "Sim", "Não")
    vencedora, posteriores = classificar(caso)
    for rotulo, valor in posteriores.items():
        print(f"  P({rotulo}) * produtório das verossimilhanças = {valor:.5f}")
    print(
        f"\nCritério MAP: '{vencedora}' vence, mesmo o chat não sendo do tipo 'só frases "
        "prontas', porque as outras duas pistas (reação robótica e online 24h seguidas) "
        "pesam mais na multiplicação."
    )


def _demonstrar_problema_zero():
    """Mostra o produtório zerando com uma pista nunca vista em bot, e a suavização de Laplace corrigindo."""
    print(
        "\n--- Testando uma pista nunca vista: reação robótica=Não, com online 24h=Sim, "
        "chat pronto=Sim ---"
    )
    caso = ("Não", "Sim", "Sim")

    _, posteriores_bruta = classificar(caso, suavizar=False)
    for rotulo, valor in posteriores_bruta.items():
        print(f"  [bruta]     P({rotulo}) * produtório = {valor:.5f}")
    print(
        "\nNenhum bot do histórico jogou com reação NÃO robótica, então P(reação=Não | "
        "bot) = 0/3 = 0, e o produtório inteiro zera. As outras duas pistas (online "
        "24h=Sim, chat pronto=Sim) até apontavam pra bot, mas isso deixa de importar: "
        "zero vezes qualquer coisa é zero, e o veredito vira 'humano' só por essa conta "
        "matemática, não porque as evidências de verdade pesem pra lá."
    )

    vencedora_suave, posteriores_suave = classificar(caso, suavizar=True)
    for rotulo, valor in posteriores_suave.items():
        print(f"  [suavizada] P({rotulo}) * produtório = {valor:.5f}")
    print(
        f"\nCom a suavização de Laplace, nenhuma probabilidade fica em zero: agora dá pra "
        f"comparar bot e humano de verdade, e '{vencedora_suave}' vence numa disputa "
        "honesta entre as três pistas, não por um acidente de contagem."
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
    return {"bot": "#C0392B", "humano": "#2E86C1"}


def plotar_probabilidades_por_atributo(caminho_saida: Path | None = None) -> Path:
    """
    Gráfico de barras: P(pista=Sim | classe), já suavizada, lado a lado
    pra bot e humano, nas 3 pistas do exemplo de brincadeira. Visualiza
    por que reação robótica é a pista mais forte (barra de bot bem mais
    alta que a de humano).
    """
    plt = _preparar_pyplot()
    import numpy as np

    cores = _cores_por_classe()
    posicoes = np.arange(len(ATRIBUTOS))
    largura = 0.35

    valores_bot = [verossimilhanca(i, "Sim", "bot", suavizar=True) for i in range(1, len(ATRIBUTOS) + 1)]
    valores_humano = [verossimilhanca(i, "Sim", "humano", suavizar=True) for i in range(1, len(ATRIBUTOS) + 1)]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(posicoes - largura / 2, valores_bot, largura, label="bot", color=cores["bot"])
    ax.bar(posicoes + largura / 2, valores_humano, largura, label="humano", color=cores["humano"])
    ax.set_xticks(posicoes)
    ax.set_xticklabels(ATRIBUTOS, rotation=10)
    ax.set_ylabel("P(pista = Sim | classe), suavizada")
    ax.set_title("Quão forte é cada pista, bot x humano")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "probabilidades_por_pista.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_problema_zero(caminho_saida: Path | None = None) -> Path:
    """
    Compara, em dois gráficos de barras lado a lado, o posterior
    proporcional de bot e humano pro caso 'reação robótica=Não' antes e
    depois da suavização de Laplace: visualiza a barra de bot sumindo de
    vez (zero) na versão bruta e reaparecendo na versão suavizada.
    """
    plt = _preparar_pyplot()
    cores = _cores_por_classe()
    caso = ("Não", "Sim", "Sim")

    _, posteriores_bruta = classificar(caso, suavizar=False)
    _, posteriores_suave = classificar(caso, suavizar=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, posteriores, titulo in (
        (ax1, posteriores_bruta, "Bruta (sem suavização)"),
        (ax2, posteriores_suave, "Suavizada (Laplace)"),
    ):
        rotulos = list(posteriores.keys())
        valores = [posteriores[r] for r in rotulos]
        ax.bar(rotulos, valores, color=[cores[r] for r in rotulos])
        ax.set_title(titulo)
        ax.set_ylabel("posterior proporcional")
        for i, valor in enumerate(valores):
            ax.text(i, valor, f"{valor:.4f}", ha="center", va="bottom")

    fig.suptitle("O produto de zero mata a chance de 'bot' antes mesmo de comparar")
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "problema_produto_zero.png"
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_saida}")
    return caminho_saida


def plotar_densidades_gaussianas(X_train, y_train, atributo: str, caminho_saida: Path | None = None) -> Path:
    """
    Desenha as curvas de sino (Gaussianas) que o GaussianNB ajusta pra
    fraude e pra normal no atributo mais discriminante, sobrepostas ao
    histograma real dos valores: mostra a suposição da variante Gaussiana
    em ação, a mesma ideia da Parte 1, só que com atributo contínuo em vez
    de Sim/Não.
    """
    plt = _preparar_pyplot()
    import numpy as np

    cores = {"normal": "#2E86C1", "fraude": "#C0392B"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for rotulo, nome, cor in ((0, "normal", cores["normal"]), (1, "fraude", cores["fraude"])):
        valores = X_train.loc[y_train == rotulo, atributo]
        ax.hist(valores, bins=40, density=True, alpha=0.35, color=cor, label=f"{nome} (histograma)")

        media, desvio = valores.mean(), valores.std()
        eixo_x = np.linspace(valores.min(), valores.max(), 300)
        densidade = (1 / (desvio * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((eixo_x - media) / desvio) ** 2)
        ax.plot(eixo_x, densidade, color=cor, linewidth=2, label=f"{nome} (curva ajustada pelo GaussianNB)")

    ax.set_xlabel(atributo)
    ax.set_ylabel("densidade")
    ax.set_title(f"GaussianNB assume essa curva de sino pra cada classe em '{atributo}'")
    ax.legend()
    fig.tight_layout()

    caminho_saida = caminho_saida or IMAGES_DIR / "densidades_gaussianas.png"
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
    Refaz à mão, com o exemplo de brincadeira do anti-cheat, a conta que o
    Naive Bayes faz escondida por trás do `GaussianNB` do scikit-learn:
    prior, verossimilhança, produtório, critério MAP, e por que o produto
    de zero é um problema resolvido pela suavização de Laplace.
    """
    _titulo("PARTE 1: ENTENDENDO A IDEIA COM UM EXEMPLO DE BRINCADEIRA")
    print(
        "\nImagine que você modera um servidor de jogo online e tem o histórico de 10 "
        "contas já analisadas manualmente (3 bots, 7 humanos), cada uma com 3 pistas de "
        "comportamento: reação robótica, se fica online 24h seguidas e se só manda "
        "mensagem pronta no chat. Uma conta nova chega, sem rótulo, e o sistema "
        "anti-cheat precisa decidir: bot ou humano?"
    )

    _explicar_teorema_com_uma_pista()
    _imprimir_tabela_verossimilhancas()
    _classificar_caso_suspeito()
    _demonstrar_problema_zero()

    print()
    _titulo("PARTE 2: AS MESMAS IDEIAS, AGORA EM DESENHO")
    plotar_probabilidades_por_atributo()
    plotar_problema_zero()
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Treino de verdade, no dataset de fraude (Parte 3)
# ---------------------------------------------------------------------------


def _escolher_atributo_mais_discriminante(X_train, y_train) -> str:
    """Escolhe, entre as colunas de X_train, a que tem maior diferença de média entre fraude e normal."""
    medias_fraude = X_train[y_train == 1].mean()
    medias_normais = X_train[y_train == 0].mean()
    diferenca = (medias_fraude - medias_normais).abs()
    return diferenca.idxmax()


def _comparar_var_smoothing(X_train, X_test, y_train, y_test):
    """
    Treina GaussianNB com dois valores de var_smoothing: a versão do
    GaussianNB pra suavização de Laplace, em vez de somar 1 numa
    contagem, soma uma fração da variância geral em cada variância de
    classe, pra evitar densidade explodindo quando a variância de um
    atributo dentro de uma classe é quase zero.
    """
    for var_smoothing in (1e-9, 1e-2):
        modelo = GaussianNB(var_smoothing=var_smoothing)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\n--- var_smoothing={var_smoothing} ---")
        print(classification_report(y_test, y_pred, digits=4))


def main():
    demonstracao_manual()

    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    _titulo("PARTE 3: AGORA COM DADOS DE VERDADE (fraude em cartão de crédito)")
    _comparar_var_smoothing(X_train, X_test, y_train, y_test)

    atributo = _escolher_atributo_mais_discriminante(X_train, y_train)
    _titulo(f"VISUALIZANDO A SUPOSIÇÃO GAUSSIANA no atributo mais discriminante ({atributo})")
    plotar_densidades_gaussianas(X_train, y_train, atributo)


if __name__ == "__main__":
    main()
