import re
import random

import numpy
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import cross_validation
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
# from sklearn import svm, metrics
# from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.model_selection import cross_val_predict, GridSearchCV
# from sklearn.model_selection import cross_val_predict, cross_val_score, ShuffleSplit
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler, normalize
from sympy.functions.special.gamma_functions import gamma
import standardalize

total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = 291920
# Total_data_number = 5000

data = []  # this list is to generate index value for k fold validation
count_for_number_of_instances = 0
i = 0
cout = 0
z = 1
print("Starting To read From Text ...  ")
with open('enzyme_dataset_feature_reduced.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
        l = list(map(float, l))
        total_matrix.append(l)
        i += 1
        if i == Total_data_number:
            break

# total_matrix = standardalize.std(total_matrix, 882, 412)
c = 0

print("Total instances ", len(total_matrix))

for l in total_matrix:
    last_index = len(l) - 1
    feature_list_of_all_instances.append(l[0:last_index])
    class_list_of_all_instances.append(l[last_index])
# c=0
for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Total features ", len(feature_list_of_all_instances[0]))
print("Positive data  ", c)

number_of_folds = 5
kf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)
under_sample = RandomUnderSampler()

temp_test_class_list = []
best_roc = 0;
best_C = 0
best_G = 0
best_aupr = 0
fold_number = 0
for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):

    fold_number += 1
    temp_train_feature_list = []
    temp_train_class_list = []
    for index in train_set_indexes:
        temp_train_feature_list.append(feature_list_of_all_instances[index])
        temp_train_class_list.append(class_list_of_all_instances[index])

    temp_test_feature_list = []
    temp_test_class_list = []
    for index in test_set_indexes:
        temp_test_feature_list.append(feature_list_of_all_instances[index])
        temp_test_class_list.append(class_list_of_all_instances[index])

    counter_for_positive_class = 0

    temp_train_feature_list, temp_train_class_list = under_sample.fit_sample(temp_train_feature_list,
                                                                             temp_train_class_list)

    cou1 = 0
    cou2 = 0
    for h in temp_train_class_list:
        if h == 1:
            cou1 += 1
        if h == 0:
            cou2 += 1
    print("positive in train list ", cou1)
    print("negative in train list ", cou2)

    cou1 = 0
    cou2 = 0
    for h in temp_test_class_list:
        if h == 1:
            cou1 += 1
        if h == 0:
            cou2 += 1
    print("positive in test list ", cou1)
    print("negative in test list ", cou2)

    for g in numpy.arange(.0001, .01, .0005):
        for c in range(2400, 3000, 10):
            # print(c, g)

            clf = sklearn.svm.SVC(C=c, gamma=g, kernel='rbf')

            clf.fit(temp_train_feature_list, temp_train_class_list)

            predicted = []
            j = 0
            for j in temp_test_feature_list:
                val = [j]
                predicted.append(clf.predict_proba(val))

            if best_roc < sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:1]):
                best_roc = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:1])
                best_aupr = average_precision_score(temp_test_class_list, predicted[:1])
                best_C = c
                best_G = g
                report = classification_report(np.asarray(temp_test_class_list), np.asarray(predicted[:1]))
                print("working on C = ", c, " and gamma = ", g, " for fold number = ", fold_number, " out of ",
                      number_of_folds)
                print("best auc :- ", best_roc, " with C = ", best_C, ' and gamma = ', best_G, " with aupr = ",
                      best_aupr, "\n ", report)
                # print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
                # print("# Average precision score :- ", end='')
                # print(average_precision_score(temp_test_class_list, predicted))

                # print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))
