from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import keras.backend as K
import pandas as reader
import numpy as np

from network import train,create_network
from Statistics import normalize,unnormalize,MakeLag

#pega o path
dataset_path='Data/Dados UBE.csv'
answers_path='Data/UBE_Results.csv'

#determina o numero de defasagens para o problema em questão
number_of_lag=8

#Determina um limiar de defasagem para os valores minimos para casos onde os valores sejam diferentes do que o esperado
min_lag=.8
max_lag=1.2

#definie os limiares para a padrinização
std_min = 0.3
std_max = 0.7

#armazena os Valores de min e max
min_values = []
max_values = []
#armazena os Valores de min e max para as respostas
Answers_min_values = []
Answers_max_values = []
#csv de entrada
input_csv = reader.read_csv(dataset_path)
#labels
labels = ["ATP","RBG","BLS","SFB","FZB","UBE"]
new_sheet = None
new_result_sheet = None


for name in labels:
    new_elem_sheet = MakeLag(input_csv,name,number_of_lag)
    new_elem_sheet,new_min,new_max = normalize(new_elem_sheet,(min_lag,max_lag),(std_min,std_max))
    min_values.append(new_min)
    max_values.append(new_max)
    if(new_sheet is None):
        new_sheet = new_elem_sheet
    else:
        new_sheet = reader.concat([new_elem_sheet, new_sheet], axis=1,sort=False)

result_csv = reader.read_csv(answers_path) 
for name in range(0,12):
    new_elem_sheet,new_min,new_max = normalize(result_csv[str(name+1)],(min_lag,max_lag),(std_min,std_max))
    Answers_min_values.append(new_min)
    Answers_max_values.append(new_max)
    if(new_result_sheet is None):
        new_result_sheet = new_elem_sheet
    else:
        new_result_sheet = reader.concat([new_elem_sheet, new_result_sheet], axis=1,sort=False)

new_result_sheet=new_result_sheet.drop(range(0,number_of_lag-1))
 
X_train, X_temp, y_train, y_temp = train_test_split(new_sheet.values, new_result_sheet.values,test_size=0.25)
'''
temp_value = None
temp_labels = None
for i in range(0,len(min_values)):
    unormalized_vector = X_temp[:,i*number_of_lag:(i+1)*number_of_lag]
    unormalized_vector = unnormalize(unormalized_vector,(min_values[i],max_values[i]),(std_min,std_max))
    unormalized_label_vector = y_temp[:,i*number_of_lag:(i+1)*number_of_lag]
    unormalized_label_vector = unnormalize(unormalized_label_vector,(min_values[i],max_values[i]),(std_min,std_max))

    if(temp_value is None):
        temp_value = unormalized_vector
        temp_labels = unormalized_label_vector
    else:
        temp_value = np.concatenate((temp_value,unormalized_vector),axis=1)    
        temp_labels = np.concatenate((temp_labels,unormalized_label_vector),axis=1)    
X_temp = temp_value
y_temp = temp_labels   
'''
X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp,test_size=0.4)

    
network = create_network(6*number_of_lag,64,12,Adam(0.001))
train(network,(X_train,y_train),(X_test,y_test),(X_validation,y_validation),num_epochs_without_change=100)

#print(X_train.shape)
#print(X_train)
#print(y_train.shape)
