import numpy as np
import math
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics


def test_process_data():
    data_dict = {'sex': ['male', 'female', 'male'], 'age': [34, 45, 24],
                 'salary': ['A', 'B', 'B']}
    df = pd.DataFrame.from_dict(data_dict)
    cat_feats = ['sex']
    x, y, enc, lb = process_data(df, categorical_features=cat_feats,
                                 label='salary', training=True)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(enc, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)


def test_train_model():
    X = np.array([[0, 0, 1], [1, 0, 1],
                  [1, 1, 1], [1, 1, 0]])
    y = np.array([1, 0, 0, 0])

    md = train_model(X, y)
    assert isinstance(md, DecisionTreeClassifier)


def test_compute_model_metrics():
    y_true = np.array([1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1])
    prec = precision_score(y_true, y_pred, zero_division=1)
    rec = recall_score(y_true, y_pred, zero_division=1)
    fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)

    prec_rd, rec_rd, fb_rd = compute_model_metrics(y_true, y_pred)
    assert math.isclose(prec, prec_rd)
    assert math.isclose(rec, rec_rd)
    assert math.isclose(fbeta, fb_rd)
