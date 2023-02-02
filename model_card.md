# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Mohamed Anas created this model. The model is Decision Tree Classifier using default 
hyperparameters in scikit-learn 1.0.0 except for the hyperparameters min_samples_split, which 
has been set a value of 10.

## Intended Use

This model should be used to predit whether the possible salary for a person is >50K or <=50K dollars  anually.

## Training Data

The training data was obtained from the UCI Machine Learning Repository(https://archive.ics.uci.edu/ml/datasets/census+income). The targets to be classified into are >50K  and <=50K. The target is
in the 'salary' column. The dataset is slightly imbalanced with approx. 25000 samples for <=50K and 8000 samples for >50K.

The original dataset has 32561 rows. A 80-20 split was used to split the dataset into training and 
validation set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

20% of the original data was used for evaluation. K-fold cross validation was not used during training.

## Metrics

Precision, Recall and Fbeta scores were calculated. 

Precision: 0.65   Recall: 0.62   Fbeta: 0.64

## Ethical Considerations

The dataset uses features such as race, sex, age and other personal details of an individual. There is a high possiblity of inherent unfairness in the data. There is no information regarding how the dataset was collected. Due to unfairness in the data, the model developed from the data will also be biased.

## Caveats and Recommendations

Users can try other ml models and also it is recommended to train after stratification since the dataset is imbalanced.
