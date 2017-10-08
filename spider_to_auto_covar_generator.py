import re

import numpy
import numpy as np
import mysql.connector


hsa_list = []
hsa_list_temp = []
cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database='ir_dataset')
cursor = cnx.cursor()

query = ("SELECT hsa FROM hsaSeq")
x = cursor.execute(query)

for i in cursor:
    # print(i)

    hsa_list.append(i)

cnx.close()

count = 0

map = {}

for i in hsa_list:

    i = ''.join(i)
    x = i[0:3]
    y = i[4:]
    i = x + y + '.txt.spd3'
    j = x + y

    # hsa_list.append(i)
    hsa_list_temp.append(i)


hsa_count = 0

for hsa in hsa_list_temp:

    print(hsa)
    file_read = open('/home/farshid/Desktop/IR_seqs/' + hsa, 'r')
    file = file_read.readline()

    final_list = []

    num_lines = sum(1 for line in open('/home/farshid/Desktop/IR_seqs/' + hsa, 'r'))

    matrix = [[0 for x in range(8)] for y in range(10)]
    probability_matrix = [[0 for x in range(3)] for y in range(10)]

    i = 0

    a = []

    while i < num_lines - 1:
        i += 1

        str1 = file_read.readline()


        l = str1.rstrip('\n').split('\t')

        a.append(l[3:])

    array = a
    # print(a)
    # break

    degree_matrix = [[0 for x in range(8)] for y in range(num_lines-1)]

    degree_index = 0

    for x in range(0,num_lines-1):
        degree_index = 0
        for y in range(1,5):
            # print("x = ", x , " y = ", y , " deg_index = ",degree_index)
            degree_matrix[x][degree_index] = numpy.math.sin(float(array[x][y]) * numpy.pi / 180 )
            degree_matrix[x][degree_index+1] = numpy.math.cos(float(array[x][y]) * numpy.pi / 180  )
            degree_index += 2

        # print()


    print(len(array))
    print(len(array[0]))
    print(len(degree_matrix))
    print(len(degree_matrix[0]))

    temp_array = array

    array = degree_matrix

    for k in range(0, 10):
        for j in range(0, 8):
            for i in range(0, num_lines - 1 - k):
                # print("k = ", k, " j = ", j, " i = ", i)
                matrix[k][j] += float(array[i][j]) * float(array[i + k][j])
            matrix[k][j] = matrix[k][j] / (num_lines - 1)


    array = temp_array

    for k in range(0, 10):
        for j in range(5, 8):
            for i in range(0, num_lines - 1 - k):
                # print("k = ", k, " j = ", j-5, " i = ", i)
                probability_matrix[k][j - 5] += numpy.float(array[i][j]) * numpy.float(array[i + k][j])
                # print(probability_matrix[k][j-5])
            probability_matrix[k][j - 5] = probability_matrix[k][j - 5] / (num_lines - 1)
            # print(k,i,j)






    final_matrix = [[0 for x in range(11)] for y in range(10)]
    for m in range(0, len(matrix)):
        for n in range(0, len(matrix[m]) + len(probability_matrix[m])):
            if n < 8:
                final_matrix[m][n] = matrix[m][n]
            else:
                final_matrix[m][n] = probability_matrix[m][n - 8]





    matrix = final_matrix
    # print(len(matrix[0]))
    # print(len(matrix))
    print(matrix)
