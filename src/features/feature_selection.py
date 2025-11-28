# src/features/feature_selection.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


def anova_feature_selection(
    X: pd.DataFrame, y: pd.Series, top_k: int = 5
) -> list[str]:
    """
    Perform ANOVA F-test feature selection.

    Args:
        X: DataFrame of features
        y: binary target (0/1)
        top_k: number of features to keep

    Returns:
        List of selected feature names
    """
    # F-test → (F-scores, p-values)
    F, pvals = f_classif(X, y)

    # Score dataframe
    scores_df = pd.DataFrame({
        "feature": X.columns,
        "F": F,
        "pval": pvals
    })

    # Sort by F-score (descending)
    scores_df = scores_df.sort_values("F", ascending=False)

    # Keep top-k features
    selected = scores_df["feature"].iloc[:top_k].tolist()
    return selected
