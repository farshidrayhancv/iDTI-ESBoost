import re

total_matrix = [[]]
feature_list_of_all_instances = []
class_list_of_all_instances = []
total_matrix = []

# clf = MLPClassifier(hidden_layer_sizes=5000,activation='logistic',learning_rate='adaptive')
data = []  # this list is to generate index value for k fold validation

# file_read = open('/home/farshid/Desktop/dataset.txt', 'r')
count_for_number_of_instances = 0


def std(total_matrix ,index , gap ):

    max = [0] * 300000
    min = [999] * 300000

    print("Standardizing ...")

    for col in range(index, index+gap):

        for j in range(0, len(total_matrix)):

            if total_matrix[j][col] > max[col]:
                max[col] = total_matrix[j][col]

            if total_matrix[j][col] < min[col]:
                min[col] = total_matrix[j][col]
        # print("col = ", col, " max = ", max[col], " min = ", min[col])

        for j in range(0, len(total_matrix)):
            # print(" previous value " , total_matrix[j][col] , end=' ')
            total_matrix[j][col] = (total_matrix[j][col] - min[col]) / (max[col] - min[col])
            # print("new value " , total_matrix[j][col])
    print( "Sample data after Standardizing  " , total_matrix[1][880:885] )
    return total_matrix
# i = 0
# while i < 300000:
#
#     x = file_read.readline()
#     if len(x) <= 10:
#         break
#     l = re.findall("-?\d+", x)
#     l = list(map(int, l))
#     total_matrix.append(l)
#     # feature_list_of_all_instances.append(l[0:1282])
#     # class_list_of_all_instances.append(l[1282])
#     i += 1


# std(total_matrix,882,400)

