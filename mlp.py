from sklearn.model_selection import train_test_split
#from keras.optimizers import Adam

#import keras.backend as K
import pandas as reader
import numpy as np

from constants import dataset_path
from constants import answers_path
from constants import number_of_lag
from constants import min_values
from constants import max_values
from constants import Answers_min_values
from constants import Answers_max_values
from constants import labels

#from network import train
#from network import create_network
from Statistics import normalize
from Statistics import unnormalize
from Statistics import MakeLag

#csv de entrada
input_csv = reader.read_csv(dataset_path)
result_csv = reader.read_csv(answers_path) 

new_sheet = None
new_result_sheet = None


for name in labels:
    new_elem_sheet = MakeLag(input_csv,name,number_of_lag)
    new_elem_sheet,new_min,new_max = normalize(new_elem_sheet)
    min_values.append(new_min)
    max_values.append(new_max)
    if(new_sheet is None):
        new_sheet = new_elem_sheet
    else:
        new_sheet = reader.concat([new_elem_sheet, new_sheet], axis=1,sort=False)

for name in range(0,12):
    new_elem_sheet,new_min,new_max = normalize(result_csv[str(name+1)])
    Answers_min_values.append(new_min)
    Answers_max_values.append(new_max)
    if(new_result_sheet is None):
        new_result_sheet = new_elem_sheet
    else:
        new_result_sheet = reader.concat([new_elem_sheet, new_result_sheet], axis=1,sort=False)

new_result_sheet=new_result_sheet.drop(range(0,number_of_lag-1))
 
X_train, X_temp, y_train, y_temp = train_test_split(new_sheet.values, new_result_sheet.values,test_size=0.25)

temp_value = None
temp_labels = None
for i in range(0,len(min_values)):
    unormalized_vector = X_temp[:,i*number_of_lag:(i+1)*number_of_lag]
    unormalized_vector = unnormalize((min_values[i],max_values[i]),unormalized_vector)
    unormalized_label_vector = y_temp[:,i*number_of_lag:(i+1)*number_of_lag]
    unormalized_label_vector = unnormalize((Answers_min_values[i],Answers_max_values[i]),unormalized_label_vector)

    if(temp_value is None):
        temp_value = unormalized_vector
        temp_labels = unormalized_label_vector
    else:
        temp_value = np.concatenate((temp_value,unormalized_vector),axis=1)    
        temp_labels = np.concatenate((temp_labels,unormalized_label_vector),axis=1)    
X_temp = temp_value
y_temp = temp_labels

X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp,test_size=0.4)

    
network = create_network(6*number_of_lag,64,12,Adam(0.001))
train(network,(X_train,y_train),(X_test,y_test),(X_validation,y_validation),num_epochs_without_change=100)

#print(X_train.shape)
#print(X_train)
#print(y_train.shape)
