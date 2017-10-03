import re
import random
import numpy as np
from sklearn import cross_validation

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from   sklearn.ensemble import RandomForestClassifier
# from brew.base import Ensemble
from sklearn.model_selection import StratifiedKFold
import sklearn
from sklearn.model_selection import cross_val_predict, GridSearchCV
# from sklearn.model_selection import cross_val_predict, cross_val_score, ShuffleSplit
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier
from sympy.functions.special.gamma_functions import gamma
import standardalize

total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = 291920
# Total_data_number = 5000
data = []  # this list is to generate index value for k fold validation

print("Opening  Text ...  ")
count_for_number_of_instances = 0
i = 0
cout = 0
print("Starting To read From Text ...  ")
with open('feature_reduced_dataset.txt', 'r') as file_read:
    for x in file_read:
        if len(x) <= 10:
            break
        l = x.rstrip('\n').split(',')
        l = list(map(float, l))
        total_matrix.append(l)
        # feature_list_of_all_instances.append(l[0:519])
        # class_list_of_all_instances.append(int(l[519]))
        i += 1
        #
        # if i == Total_data_number:
        #     break

c = 0

print("Total instances ", len(total_matrix))

for l in total_matrix:
    index = len(l) -1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)

number_of_folds = 5
kf = StratifiedKFold(n_splits=number_of_folds,shuffle=True)
under_sample = RandomUnderSampler()

print("Starting K fold data to Classifier   ...   ")
avg_roc = 0
for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):

    # temp_train_feature_list = feature_list_of_all_instances[train_set_indexes]
    # temp_train_class_list = class_list_of_all_instances[train_set_indexes]
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

    print("Creating Training dataset")
    #
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

    print("Training ...  ")

    #
    est = DecisionTreeClassifier(max_depth=18, min_samples_split=4)   # roc avg result  = 0.895951148384 cross fold = 5 l_rate = 1 n_estimator = 150
    est1 =  DecisionTreeClassifier(max_depth=16, min_samples_split=2)

    # est2 = DecisionTreeClassifier(max_depth=15, min_samples_split=3)
    # est3 = DecisionTreeClassifier(max_depth=15, min_samples_split=4)
    # est4 = DecisionTreeClassifier(max_depth=15, min_samples_split=2)


    estimator_list = []
    for d in range(14,20,1):
        for s in range(2,5):
            estimator_list.append(DecisionTreeClassifier(max_depth=d,min_samples_split=s))

    tuned_parameters = [{
        'base_estimator': estimator_list,
        'n_estimators': [150],
        'learning_rate': [1],  # 'entropy'],
        'algorithm': ['SAMME'],
    }, ]
    #
    scores = [ 'roc_auc' , 'recall_macro','precision_macro' , 'average_precision']
    scores = [ 'roc_auc' ]

    #
    #
    for score in scores:
        print("Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5,
                           scoring='%s' % score)
        # clf = AdaBoostClassifier(estimator, n_estimators=50, learning_rate=1, algorithm='SAMME')

        clf.fit(temp_train_feature_list, temp_train_class_list)

        print("Best Score  ", clf.best_score_)
        print("Best Estimator   ", clf.best_estimator_)

        print("Predicting ")
        predicted = clf.predict(temp_test_feature_list)


        predicted_with_threshold = []
        j = 0
        for j in temp_test_feature_list:
            val = [j]
            val1 = clf.predict_proba(val)
            # print(val1)
            if val1[0][1] > .55:
                predicted_with_threshold.append(1)
            else:
                predicted_with_threshold.append(0)

        # print(" Confusion matrix ")
        # print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
        # print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted_with_threshold))

        print("# Roc auc score                :- ", end='')
        print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
        print("# Average precision score      :- ", end='')
        print(average_precision_score(temp_test_class_list, predicted))
        print("# Roc auc score with threshold :- ", end='')
        print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted_with_threshold))
        # fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
        # print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
        # confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
        # print(" -> Specificity ", end='')
        # specificity = float(confusion_matrix[0][0]) / (float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
        # print(specificity)
        # print(" -> Sensitivity ", end='')
        # sensitivity = float(confusion_matrix[1][1]) / float((confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
        # print(sensitivity)
        print("Actual Report ")
        print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))
        print("Report with threshold ")
        print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted_with_threshold)))
        avg_roc += sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)

print("Average roc = " ,avg_roc/number_of_folds)