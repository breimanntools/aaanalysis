"""This is a script for the backend of the TreeModel.eval() method."""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

import aaanalysis.utils as ut


# II Main methods
@ut.catch_undefined_metric_warning()
def eval_feature_selections(X, labels=None, list_is_selected=None, names_feature_selections=None, n_cv=5,
                            list_metrics=None, list_model_classes=None, list_model_kwargs=None, verbose=True):
    """Evaluate the performance of different feature selections for multiple models,
    and compute the average score across all models and rounds for each scoring metric."""
    n_feature_sets = len(list_is_selected)
    n_metrics = len(list_metrics)
    list_evals = np.empty((n_feature_sets, n_metrics))
    if verbose:
        ut.print_start_progress(start_message=f"Tree Model starts evaluation of {n_feature_sets} feature sets.")
    for i, is_selected_rounds in enumerate(list_is_selected):
        n_rounds = len(is_selected_rounds)
        results_round = np.empty((n_rounds, n_metrics))
        for j, is_selected in enumerate(is_selected_rounds):
            X_selected = X[:, is_selected]
            for k, eval_score in enumerate(list_metrics):
                if verbose:
                    pct_progress = (j + k/n_metrics)/n_rounds
                    ut.print_progress(i=i+pct_progress, n_total=n_feature_sets)
                model_scores = [
                    cross_val_score(model_class(**model_kwargs), X_selected, y=labels, cv=n_cv, scoring=eval_score).mean()
                    for model_class, model_kwargs in zip(list_model_classes, list_model_kwargs)
                ]
                results_round[j, k] = np.mean(model_scores)
        # Average the scores across all rounds for each scoring metric
        list_evals[i, :] = results_round.mean(axis=0)
    if verbose:
        ut.print_end_progress(end_message=f"Tree Model finished evaluation and saves results.")
    # Create the DataFrame
    cols_eval = [metric if isinstance(metric, str) else metric.__name__ for metric in list_metrics]
    df_eval = pd.DataFrame(list_evals, columns=cols_eval).round(4)
    df_eval = ut.add_names_to_df_eval(df_eval=df_eval, names=names_feature_selections)
    return df_eval
