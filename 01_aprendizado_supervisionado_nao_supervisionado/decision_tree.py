"""
Árvore de decisão aplicada ao dataset de detecção de fraude.

Conceitos revisados: indução top-down, critérios de divisão (entropia,
ganho de informação), poda.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from utils.data_utils import get_train_test_split, load_raw_data, build_preprocessing_pipeline


def main():
    df = load_raw_data()
    df = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    # TODO: treinar DecisionTreeClassifier, testar critérios ("gini" vs "entropy")
    # TODO: imprimir classification_report(y_test, y_pred)
    # TODO: opcionalmente exportar a árvore treinada (plot_tree ou export_text)


if __name__ == "__main__":
    main()
