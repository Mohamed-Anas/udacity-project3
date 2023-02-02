# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference
from ml.model import compute_model_metrics, slice_census

model_location = '../model/ml_model.joblib'
encoder_location = '../model/encoder.joblib'
lb_location = '../model/label_binarizer.joblib'
metrics_location = '../model/slice_output.txt'

df = pd.read_csv('../data/census.csv')
print('Data read successfully')

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb)

# Train and save a model.
ml_model = train_model(X_train, y_train)
print('Model Trained Successfully')

# Saving model, encoder, label binarizer
dump(ml_model, model_location)
dump(encoder, encoder_location)
dump(lb, lb_location)
print('Model Stored Successfully')

# Predictions on test data
test_predictions = inference(ml_model, X_test)
prec, rec, fbeta = compute_model_metrics(y_test, test_predictions)
print(
    f'Model Metrics \n precision: {prec},recall: {rec}, fbeta: {fbeta}')

# Metrics on slices of categorical feature on test data

print('Running slice tests')
feature = "education"  # Any categorical feature

result_dict = slice_census(
    test,
    feature,
    ml_model,
    "salary",
    cat_features,
    encoder,
    lb)
with open(metrics_location, 'w') as f:
    for key, val in result_dict.items():
        print(f'{key}:', file=f)
        for key2, val2 in result_dict[key].items():
            print(f'\t {key2} : {val2}', file=f)
print('Slice tests completed')
