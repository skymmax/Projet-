# src/features/feature_selection.py
# anova feature selection helper

import pandas as pd
from sklearn.feature_selection import f_classif


def anova_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 5,
) -> list[str]:
    # anova f-test feature selection, returns top_k feature names
    F, pvals = f_classif(X, y)

    scores_df = pd.DataFrame(
        {
            "feature": X.columns,
            "F": F,
            "pval": pvals,
        }
    )

    scores_df = scores_df.sort_values("F", ascending=False)
    selected = scores_df["feature"].iloc[:top_k].tolist()
    return selected
