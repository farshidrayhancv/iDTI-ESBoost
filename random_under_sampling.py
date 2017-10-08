import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

dataset = ''  # path to dataset

print("dataset : ", dataset)
df = pd.read_csv(dataset, header=None)

###################################
# Converting dataset to X,Y format
df['label'] = df[df.shape[1] - 1]
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])
X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])
#############################

skf = StratifiedKFold(n_splits=5, shuffle=True)
normalization_object = Normalizer()

# setting up parameters
depth = 15
estimators = 50

sampler = RandomUnderSampler()


for train_index, test_index in skf.split(X, y):

    X_train = X[train_index]
    X_test = X[test_index]

    X_train = normalization_object.fit_transform(X_train)

    y_train = y[train_index]
    y_test = y[test_index]

    X_sampled, y_sampled = sampler.fit_sample(X_train, y_train)

    classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=depth),
        n_estimators=estimators,
        learning_rate=1, algorithm='SAMME')

    classifier.fit(X_sampled, y_sampled)
    
    X_test = normalization_object.transform(X_test)

    predictions = classifier.predict_proba(X_test)

    auc = roc_auc_score(y_test, predictions[:, 1])

    aupr = average_precision_score(y_test, predictions[:, 1])

