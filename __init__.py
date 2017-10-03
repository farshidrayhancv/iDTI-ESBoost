import re
import random
import standardalize
from numpy import transpose
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2,mutual_info_classif
from sklearn.feature_selection import f_classif, SelectPercentile


def range_custom(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1


total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []
Total_data_number = int(291920)
# Total_data_number = 10000
# clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
data = []  # this list is to generate index value for k fold validation

print("Opening  Text ...  ")

count_for_number_of_instances = 0
i = 0

# for i in range(0, 1287):
#     file_read.readline()
print("Starting to read from txt")

with open('enzyme_dataset_1477.txt', 'r') as file_read:
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
        count_for_number_of_instances += 1
c = 0

print("Starting To Standardize Total Matrix ...  ")
#
total_matrix = standardalize.std(total_matrix, 882, 522+73)
print("Total instances ", len(total_matrix))
print("Total Features ", len(total_matrix[0])-1)

for l in total_matrix:
    index = len(l) -1
    # print(index)
    feature_list_of_all_instances.append(l[0:index])
    class_list_of_all_instances.append(l[index])
# for t in range_custom(.001,1,.005):

# print("for threshold T ",t )
variance = VarianceThreshold()
print(variance.get_params())
feature_list_of_all_instances = variance.fit_transform(feature_list_of_all_instances)

print("reduced feature number ", len(feature_list_of_all_instances[1]))
feature_list_of_all_instances = SelectKBest(mutual_info_classif,k=600).fit_transform(feature_list_of_all_instances,class_list_of_all_instances)
# feature_list_of_all_instances = SelectPercentile(chi2,percentile=70).fit_transform(feature_list_of_all_instances,class_list_of_all_instances)


print("reduced feature number ", len(feature_list_of_all_instances[1]))  # count = 0
# count = 0
# redunrent_list = []
# for i in range(0,519):
#     # print("checking ", i,"th column" )
#     if ( all(value[i] ==0 for value in feature_list_of_all_instances) ):
#         # print(i,"th feature redundant")
#         count += 1
#         redunrent_list.append(i)
#         continue
#     if (all(value[i] == 1 for value in feature_list_of_all_instances)):
#         # print(i, "th feature redundant")
#         count += 1
#         redunrent_list.append(i)
#         continue
#
# count2 = 0


# print("count for redundant features in 881 matrix  " , count)


file = open('enzyme_dataset_feature_reduced.txt', 'w')

for i in range(0,count_for_number_of_instances):
    print(int(i)/count_for_number_of_instances  , "% written ")
    str1 = ','.join(str(j) for j in feature_list_of_all_instances[i])
    str2 = class_list_of_all_instances[i]
    # print(str2)

    write = str1 + ',' + str(str2) + '\n'

    file.write(write)
file.close()