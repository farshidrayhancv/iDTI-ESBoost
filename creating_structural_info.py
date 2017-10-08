import re
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
# cnx.close

for i in cursor:

    i = ''.join(i)
    x = i[0:3]
    y = i[4:]
    i = x  + y + '.txt.spd3'
    j = x + y

    hsa_list.append(i)
    hsa_list_temp.append(j)

for i in hsa_list:
    z = hsa_list.index(i)
    hsa  = hsa_list_temp[z]
    file_read = open('/home/farshid/Desktop/IR_seqs/'+i, 'r')
    file = file_read.readline()
    file = file_read.readline()

    x = len(file)
    C_count = 0
    E_count = 0
    H_count = 0
    ASA_sum = 0
    psi_sin_sum = 0
    psi_cos_sum = 0
    phi_sin_sum = 0
    phi_cos_sum = 0
    theta_sin_sum = 0
    theta_cos_sum = 0
    tau_sin_sum = 0
    tau_cos_sum = 0

    length_counter = 0

    while len(file) > 1:

        length_counter += 1
        file = file.split()
        # print(file)
        if file[2] == 'C':
            C_count += 1
        if file[2] == 'H':
            H_count += 1
        if file[2] == 'E':
            E_count += 1
        ASA_sum += float(file[3])

        phi_sin_sum += np.sin(float(file[4]))
        phi_cos_sum += np.cos(float(file[4]))

        psi_sin_sum += np.sin(float(file[5]))
        psi_cos_sum += np.sin(float(file[5]))

        theta_sin_sum += np.sin(float(file[6]))
        theta_cos_sum += np.sin(float(file[6]))

        tau_sin_sum += np.sin(float(file[7]))
        tau_cos_sum += np.cos(float(file[7]))

        file = file_read.readline()

    C_count = str(C_count/length_counter)
    E_count = str(E_count/length_counter)
    H_count = str(H_count/length_counter)
    ASA_sum = str(ASA_sum/length_counter)
    phi_sin_sum = str(phi_sin_sum/length_counter)
    phi_cos_sum = str(phi_cos_sum / length_counter)
    psi_sin_sum = str(psi_sin_sum / length_counter)
    psi_cos_sum = str(psi_cos_sum / length_counter)
    theta_sin_sum = str(theta_sin_sum/ length_counter)
    theta_cos_sum = str(theta_cos_sum / length_counter)
    tau_sin_sum = str(tau_sin_sum/length_counter)
    tau_cos_sum = str(tau_cos_sum / length_counter)



    datas  = C_count + ','+ E_count + ','+ H_count + ','+ ASA_sum + ','+ phi_sin_sum + ','+ phi_cos_sum + ','+ psi_sin_sum + ','+ psi_sin_sum + ','+theta_sin_sum + ','+ theta_cos_sum+ ','+tau_sin_sum+ ','+tau_cos_sum

    print(datas)
