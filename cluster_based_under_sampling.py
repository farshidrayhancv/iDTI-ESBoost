import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import interp
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

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
number_of_clusters = 23
percentage_to_choose_from_each_cluster = 0.5


for train_index, test_index in skf.split(X, y):

    X_train = X[train_index]
    X_test = X[test_index]

    X_train = normalization_object.fit_transform(X_train)

    y_train = y[train_index]
    y_test = y[test_index]

    # Clustered Under sampling
    value, counts = np.unique(y_train, return_counts=True)
    minority_class = value[np.argmin(counts)]
    majority_class = value[np.argmax(counts)]

    idx_min = np.where(y_train == minority_class)[0]
    idx_maj = np.where(y_train == majority_class)[0]

    majority_class_instances = X_train[idx_maj]
    majority_class_labels = y_train[idx_maj]




    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(majority_class_instances)

    X_maj = []
    y_maj = []

    points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    for key in points_under_each_cluster.keys():
        points_under_this_cluster = np.array(points_under_each_cluster[key])
        number_of_points_to_choose_from_this_cluster = math.ceil(
            len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)
        selected_points = np.random.choice(points_under_this_cluster, size=number_of_points_to_choose_from_this_cluster)
        X_maj.extend(majority_class_instances[selected_points])
        y_maj.extend(majority_class_labels[selected_points])

    X_sampled = np.concatenate((X_train[idx_min], np.array(X_maj)))
    y_sampled = np.concatenate((y_train[idx_min], np.array(y_maj)))

    classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=depth),
        n_estimators=estimators,
        learning_rate=1, algorithm='SAMME')

    classifier.fit(X_sampled, y_sampled)

    predictions = classifier.predict_proba(X_test)

    auc = roc_auc_score(y_test, predictions[:, 1])

    aupr = average_precision_score(y_test, predictions[:, 1])

