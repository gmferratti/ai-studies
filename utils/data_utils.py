"""
Funções compartilhadas de carga e pré-processamento de dados.

Usadas por todos os scripts do módulo 01 para garantir que os algoritmos
sejam comparados sobre exatamente o mesmo split de treino/teste.

Dataset: Credit Card Fraud Detection (Kaggle)
https://www.kaggle.com/mlg-ulb/creditcardfraud

Para baixar:
    1. Crie uma conta/API token no Kaggle (https://www.kaggle.com/settings)
    2. kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
RANDOM_STATE = 42


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega o dataset bruto a partir do CSV local."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{path}'. Baixe-o com:\n"
            "    kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip\n"
            "(veja as instruções de configuração da API do Kaggle no topo deste arquivo)."
        )
    return pd.read_csv(path)


def build_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica normalização e trata eventuais valores faltantes."""
    df = df.dropna()
    colunas_numericas = df.columns.drop("Class")
    df[colunas_numericas] = StandardScaler().fit_transform(df[colunas_numericas])
    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Retorna X_train, X_test, y_train, y_test com random_state fixo,
    para que todos os algoritmos do módulo sejam comparáveis entre si.
    """
    X = df.drop(columns="Class")
    y = df["Class"]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
