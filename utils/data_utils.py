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
    # TODO: validar existência do arquivo e orientar o download se ausente
    raise NotImplementedError


def build_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica normalização e trata eventuais valores faltantes."""
    # TODO: normalizar colunas numéricas com StandardScaler
    raise NotImplementedError


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Retorna X_train, X_test, y_train, y_test com random_state fixo,
    para que todos os algoritmos do módulo sejam comparáveis entre si.
    """
    # TODO: separar features e target, aplicar train_test_split com
    # stratify=y (dataset é fortemente desbalanceado)
    raise NotImplementedError
