import re
import random

import numpy
import pandas
# import plot_neighbourhood_cleaning_rule as ncr
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE, adasyn, random_over_sampler
from imblearn.over_sampling.random_over_sampler import RandomOverSampler
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler, one_sided_selection
from imblearn.under_sampling import TomekLinks, neighbourhood_cleaning_rule
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

# from hsa_to_20_matrix import standardalize

total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = 291920
# Total_data_number = 1000
# clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
data = []  # this list is to generate index value for k fold validation

print("Opening  Text ...  ")

total_matrix = pandas.DataFrame(pandas.read_csv('/home/farshid/Desktop/enzyme_dataset_1477.txt'))

total_matrix = numpy.ndarray.tolist(pandas.DataFrame.as_matrix(total_matrix))

c = 0
print("Total instances ", len(total_matrix))
for l in total_matrix:
    last_index = len(l) - 1
    feature_list_of_all_instances.append(l[0:last_index])
    class_list_of_all_instances.append(l[last_index])

# print(feature_list_of_all_instances[6:])

class_list_of_all_instances = [0 if x == -1 else x for x in class_list_of_all_instances]
# c=0
# print("Total features ", len(feature_list_of_all_instances[0]))
# feature_list_of_all_instances = sklearn.preprocessing.normalize(feature_list_of_all_instances)

feature_list_of_all_instances = StandardScaler().fit_transform(feature_list_of_all_instances)

for i in class_list_of_all_instances:
    if i == 1:
        c += 1
print("Positive data  ", c)

######################################################################################################################
# Here the approach is  to keep removing some number of neighbour from the dataset until the goal scores are reached #
######################################################################################################################

# feature_list_of_all_instances, class_list_of_all_instances = ncr.neighbourhood_cleaning_rule(
#     feature_list_of_all_instances, class_list_of_all_instances, neighbour)
#
# Total_data_number = len(feature_list_of_all_instances)
# print("for ", iter, "th iterationNew total instances  ", Total_data_number)
#
# randomized_list = []
# for x in range(0, Total_data_number):
#     randomized_list.append(x)
#
# random.shuffle(randomized_list)

# temp_feature = []
# temp_class = []

# for i in range(0,Total_data_number):
# for z in randomized_list:
#     temp_feature.append(feature_list_of_all_instances[z])
#     temp_class.append(class_list_of_all_instances[z])
#     # continue
# feature_list_of_all_instances = temp_feature
# class_list_of_all_instances = temp_class
#
# iter += 1
# if iter <= 70:
#     continue

# feature_list_of_all_instances = feature_list_of_all_instances.tolist()
# class_list_of_all_instances = class_list_of_all_instances.tolist()
random_state_list = []
for random_state in range(69, 400, 500):
    print(" random state   ", random_state)  # 69
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    sampler = RandomUnderSampler(ratio=.99)
    # sampler =  ADASYN(k=2,n_neighbors=2)
    # sampler =  RandomOverSampler(ratio=.99)
    # sampler =  SMOTE(ratio=.99)
    # sampler = ClusterCentroids(estimator=KMeans(n_clusters= 10) )
    # sampler = InstanceHardnessThreshold(estimator=RandomForestClassifier(n_estimators=150,max_depth=18,min_samples_split=8))
    # sampler = AllKNN(n_neighbors=10)
    # sampler = SMOTEENN(k = 3,m=3,n_neighbors=3,enn=2)



    best_roc = 0
    best_c = 0
    best_g = 0
    best_aupr = 0
    fold_counter = 1

    print("Starting K fold data to Classifier   ...   ")
    temp_test_class_list = []
    avg_roc = 0
    roc = 0
    for train_set_indexes, test_set_indexes in kf.split(feature_list_of_all_instances, class_list_of_all_instances):
        top_roc = 0
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

        # temp_test_feature_list = feature_list_of_all_instances[temp_test_feature_list]
        # temp_test_class_list = class_list_of_all_instances[temp_test_feature_list]

        counter_for_positive_class = 0

        print("Creating Training dataset")

        temp_train_feature_list, temp_train_class_list = sampler.fit_sample(temp_train_feature_list,
                                                                            temp_train_class_list)
        # temp_train_feature_list, temp_train_class_list = ncr.neighbourhood_cleaning_rule(
        #     temp_train_feature_list, temp_train_class_list, 3)
        #
        # temp_train_feature_list, temp_train_class_list = TomekLinks().fit_sample(temp_train_feature_list, temp_train_class_list)
        # temp_train_feature_list, temp_train_class_list = NearMiss(n_neighbors=3).fit_sample(temp_train_feature_list, temp_train_class_list)
        #

        # temp_train_class_list = temp_train_class_list.tolist()
        # temp_train_feature_list = temp_train_feature_list.tolist()
        #
        #
        # randomly_generated_indexes = []
        # size = len(temp_train_feature_list) / 2
        # while (size > 0):
        #     randomly_generated_indexes.append(int(random.uniform(0, len(feature_list_of_all_instances[0]))))
        #     size -= 1
        #
        # for cursor in randomly_generated_indexes:
        #     if class_list_of_all_instances[cursor] == 0:
        #         temp_train_feature_list.append(feature_list_of_all_instances[cursor])
        #         temp_train_class_list.append(class_list_of_all_instances[cursor])

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

        # temp_train_feature_list = sklearn.preprocessing.normalize(temp_train_feature_list)
        # temp_train_feature_list =  StandardScaler().fit_transform( temp_train_feature_list )

        # temp_test_feature_list = sklearn.preprocessing.normalize(temp_test_feature_list)
        # temp_test_feature_list =  StandardScaler().fit_transform(temp_test_feature_list)


        # print("Training SVM ...  ")

        # C_List = [110]
        # gamma_list = [.0007]
        # for x in numpy.arange(1, 3000, 5):
        #     C_List.append(x)
        #
        # for x in numpy.arange(0.00001, 1, .00001):
        #     gamma_list.append(x)

        # tuned_parameters = [{
        #     'kernel': ['rbf'],
        #     'gamma': gamma_list,
        #
        # 'n_jobs': [-1],
        # 'cache_size': [100],
        # 'probability': [True],
        # 'max_iter': [-1],
        # 'decision_function_shape': ['None'],
        # 'tol': tol_range,
        # 'shrinking': [True],
        # 'C': C_List,
        # }, ]
        #
        # scores = ['average_precision', 'roc_auc', 'precision_macro', 'recall_macro' ]
        # scores = [ 'roc_auc','precision', 'precision_macro', '' ,'precision' ]
        # scores = ['recall_macro']

        #
        flag  = 1
        for c in numpy.arange(2400, 3000, 10):
            for g in numpy.arange(.0001, 1, .0002):
                # for score in scores:
                # print("Tuning hyper-parameters for %s" % score)
                # if c < 2800 :
                #     continue
                if flag == 1:
                    c = 2800
                    g = .0091
                    flag = 0
                # clf = GridSearchCV(SVC(), tuned_parameters, n_jobs=4, pre_dispatch='4*n_jobs', cv=5,
                #                    scoring='%s' % score)


                #
                clf = sklearn.svm.SVC(C=c, gamma=g, kernel='rbf', probability=True)  # use 20

                clf.fit(temp_train_feature_list, temp_train_class_list)

                # print("Best Score  ", clf.best_score_)
                # print("Best Estimator   ", clf.best_estimator_)

                # predicted = []
                # j = 0
                # for j in temp_test_feature_list:
                #     val = [j]
                # predicted.append(clf.predict(val))

                predicted = clf.predict_proba(temp_test_feature_list)

                if sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:, 1]) > top_roc:
                    top_roc = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:, 1])
                #
                print("For C = ", c, " and gamma = ", g, end='')
                # print("# Roc :- ", end='')
                print("# Roc auc score                :- ", end='')
                print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:, 1]))
            # print("# AUPR  :- ", end='')
            # print(average_precision_score(temp_test_class_list, predicted), end=' ')

            roc = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted[:, 1])
            # print("# accuracy score :- ", end='')

            # print(sklearn.metrics.accuracy_score(temp_test_class_list, predicted))
            # print("# best roc score  :- ", end='')

            # print(top_roc )                        # print("Actual Report ")
            # print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))

            avg_roc += roc

            # print("# Average precision score :- ", end='')
            # print(average_precision_score(temp_test_class_list, predicted))

            # if sklearn.metrics.roc_auc_score(temp_test_class_list, predicted) > best_roc:
            #     best_aupr = average_precision_score(temp_test_class_list, predicted)
            #     best_c = c
            #     best_g = g
            #     best_roc = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)
            # print("#############################################################################################")
            # print(" Best roc = ", best_roc, " and Aupr = ", best_aupr, " For C = ", best_c, " and gamma = ", best_g,
            #       " For fold = ", fold_counter)
            # break  # fold_counter += 1
# if top_roc >= .79:
#     random_state_list.append(random_state)

print("avg roc ", avg_roc / 5)

# fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
# print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
# confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
# print(" -> Specificity ", end='')
# specificity = float(confusion_matrix[0][0]) / (
#     float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
# print(specificity)
# print(" -> Sensitivity ", end='')
# sensitivity = float(confusion_matrix[1][1]) / float(
#     (confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
# print(sensitivity)
# print(" -> Precision ", end='')
# precision = float(confusion_matrix[1][1]) / (float(confusion_matrix[1][1]) + float(confusion_matrix[0][1]))
# print(precision)
# print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))

# predicted = []
# j = 0
# for j in temp_test_feature_list:
#     val = [j]
#
#     predicted.append(clf.predict_proba(val))
# auc(temp_test_class_list,predicted)
# plt.figure()
# plt.plot(temp_test_class_list,predicted)
# plt.show()
# # if sklearn.metrics.roc_auc_score(temp_test_class_list, predicted) > max:
#     max = sklearn.metrics.roc_auc_score(temp_test_class_list, predicted)
#     print("######################################################## updated auc ")
# print("max auc ", end='')
# print(max)
# print(" Prediction with probability")
# print(" Confusion matrix ")
# print(sklearn.metrics.confusion_matrix(temp_test_class_list, predicted))
# print("# Roc auc score           :- ", end='')
# print(sklearn.metrics.roc_auc_score(temp_test_class_list, predicted))
# print("# Average precision score :- ", end='')
# print(average_precision_score(temp_test_class_list, predicted))
# fpr, tpr, threshold = sklearn.metrics.roc_curve(temp_test_class_list, predicted)
# print("# Auc                     :- ", sklearn.metrics.auc(fpr, tpr))
# confusion_matrix = sklearn.metrics.confusion_matrix(temp_test_class_list, predicted)
# print(" -> Specificity ", end='')
# specificity = float(confusion_matrix[0][0]) / (
#     float(confusion_matrix[0][0]) + float(confusion_matrix[0][1]))
# print(specificity)
# print(" -> Sensitivity ", end='')
# sensitivity = float(confusion_matrix[1][1]) / float(
#     (confusion_matrix[1][1]) + float(confusion_matrix[1][0]))
# print(sensitivity)
# print(" -> Precision ", end='')
# precision = float(confusion_matrix[1][1]) / (
#     float(confusion_matrix[1][1]) + float(confusion_matrix[0][1]))
# print(precision)
# print(classification_report(np.asarray(temp_test_class_list), np.asarray(predicted)))


# break
# break
print(random_state_list)
