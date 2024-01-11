"""This is a script for the backend of the TreeModel.eval() method."""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

import aaanalysis.utils as ut

# II Main methods
def eval_feature_selections(X, labels=None, list_is_feature=None, names_feature_selections=None, n_cv=5,
                            list_metrics=None, list_model_classes=None, list_model_kwargs=None):
    """Evaluate the performance of different feature selections for multiple models,
    and compute the average score across all models and rounds for each scoring metric."""
    n_feature_sets = len(list_is_feature)
    n_metrics = len(list_metrics)
    list_evals = np.empty((n_feature_sets, n_metrics))
    for i, is_feature_rounds in enumerate(list_is_feature):
        results_round = np.empty((len(is_feature_rounds), n_metrics))
        for j, is_feature in enumerate(is_feature_rounds):
            X_selected = X[:, is_feature]
            for k, eval_score in enumerate(list_metrics):
                model_scores = [
                    cross_val_score(model_class(**model_kwargs), X_selected, y=labels, cv=n_cv, scoring=eval_score).mean()
                    for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs)
                ]
                results_round[j, k] = np.mean(model_scores)
        # Average the scores across all rounds for each scoring metric
        list_evals[i, :] = results_round.mean(axis=0)
    # Create the DataFrame
    cols_eval = [metric if isinstance(metric, str) else metric.__name__ for metric in list_metrics]
    df_eval = pd.DataFrame(list_evals, columns=cols_eval).round(4)
    df_eval = ut.add_names_to_df_eval(df_eval=df_eval, names=names_feature_selections)
    return df_eval
