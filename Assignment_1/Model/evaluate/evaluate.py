from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score


def evaluate_model(y_true, y_pred, classification=False, threshold=0.5):
    """
    Evaluate the model using MAE, MSE, R-squared, and accuracy metrics.

    Parameters:
        y_true (array-like): The ground truth target values.
        y_pred (array-like): The predicted values from the model.
        classification (bool): Whether the task is classification or not. Default is False.
        threshold (float): Threshold to classify as positive class for classification tasks. Default is 0.5.

    Returns:
        dict: A dictionary containing the MAE, MSE, R-squared, and possibly accuracy values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'R-squared': r2,
    }

    if classification:
        # Convert probabilistic outputs to class labels based on threshold
        y_pred_class = (y_pred > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_class)
        metrics['Accuracy'] = accuracy

    return metrics
