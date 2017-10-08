import re
import numpy as np
import mysql.connector


database = 'ir_dataset'


# class do_stuff:
hsa_list = []
cnx = mysql.connector.connect(user='root', password='123',
                              host='127.0.0.1',
                              database=database)
cursor = cnx.cursor()

query = ("SELECT hsa FROM hsaSeq")
x = cursor.execute(query)

for i in cursor:
    # print(i)
    hsa_list.append(i)

cnx.close()

# print(hsa_list[2][0])
count = 0
for l in range(0,len(hsa_list)):

    hsa_list[l] = hsa_list[l][0]
    x = hsa_list[l][0:3]
    y = hsa_list[l][4:]

    hsa_list[l] = x  + y

map = {}



for hsa in hsa_list:

    # hsa = 'hsa:10269.seq'
    # print(hsa)
    path = '/home/farshid/Desktop/IR_seqs/' + hsa + '.txt.pssm'
    file = open(path, 'r')
    file.readline()
    file.readline()
    file.readline()
    final_list = []
    num_lines = sum(1 for line in open(path, 'r'))
    num_lines = num_lines - 6 - 3  # 6 from the bottom 3 from the top
    array = np.zeros(shape=(num_lines, 20))

    matrix = [[0 for x in range(20)] for y in range(20)]





    i = 0

    while i < num_lines:
        str1 = file.read(89)
        str1 = str1[11:]
        a = []

        str1 = re.findall("-?\d+", str1)

        file.readline()
        final_list.append(str1)
        # print(do_stuff.do_shit())
        i = i + 1

    array = final_list
    # print(do_stuff.array)

    # def get_matrix(self):
    for m in range(0, 20):
        for n in range(0, 20):
            for i in range(0, num_lines-1):
                matrix[m][n] += int(array[i][m]) * int(array[i + 1][n])
            matrix[m][n] = matrix[m][n] / num_lines
                # def write(self,matrix):



    map[hsa] = matrix




    file = open('/home/farshid/Desktop/IR_seqs/'+hsa + '.txt', 'w')
    # # file.write("hsa")
    for m in range(0, len(matrix)):
        for n in range(0, len(matrix[m])):
            file.write(str(matrix[m][n]) + ",")
    #         # file.write(str(m) +str(n) + ",")



    file.close()  # print( int ( do_stuff.array[0][2] ) * 2 )
    count+=1


    val = float(count/len(hsa_list))*100
    val = int(val)
    val = str(val)
    print(val+"% completed")


