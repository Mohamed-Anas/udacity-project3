import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from .data import process_data

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = DecisionTreeClassifier(min_samples_split=10)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return np.round(precision, 2), np.round(recall, 2), np.round(fbeta, 2)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def slice_census(df, feature, model, label, cat_features, encoder, lb):
    """
    Function for calculating slice-wise statistics on a feature
    """
    label = "salary"
    result_dict = {}
    print(f'Running slice test for feature column: {feature}\n')
    print(f'Categories in {feature}: {df[feature].value_counts()}\n')
    for cat in df[feature].unique():
        df_temp = df[df[feature] == cat]
        X_test, y_test, _, _ = process_data(
            df_temp, cat_features, label=label,
            training=False, encoder=encoder, lb=lb)
        test_preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, test_preds)
        result_dict[cat] = {}
        result_dict[cat]['precision'] = precision
        result_dict[cat]['recall'] = recall
        result_dict[cat]['fbeta'] = fbeta
    return result_dict
