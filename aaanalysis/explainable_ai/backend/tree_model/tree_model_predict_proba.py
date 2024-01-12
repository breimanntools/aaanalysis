"""This is a script for the backend of the TreeModel.predict_proba() method."""
import numpy as np

# II Main methods
def monte_carlo_predict_proba(X=None, list_models=None, is_selected=None):
    """Obtain Monta Carlo estimate of prediction probability of positive class"""
    # Initialize array to store predictions from all rounds
    all_predictions = np.zeros((len(list_models), X.shape[0]))
    # Iterate through each round and each model to make predictions
    for round_idx, models in enumerate(list_models):
        round_predictions = np.zeros((len(models), X.shape[0]))
        for model_idx, model in enumerate(models):
            # Select features for the current round and model
            selected_features = is_selected[round_idx, :]
            X_selected = X[:, selected_features]
            # Make predictions with the selected model and features
            # We are interested only in the second column (probability of positive class)
            round_predictions[model_idx] = model.predict_proba(X_selected)[:, 1]
        # Aggregate predictions for this round
        all_predictions[round_idx] = np.mean(round_predictions, axis=0)
    # Aggregate predictions across all rounds and compute standard deviation
    pred = np.mean(all_predictions, axis=0)
    pred_std = np.std(all_predictions, axis=0)
    return pred, pred_std
