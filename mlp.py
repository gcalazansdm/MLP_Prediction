from sklearn.model_selection import train_test_split
from keras.regularizers import l2
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

from network import force_brute_tunnig

from utils import unnormalize_6_rows

from Statistics import normalize
from Statistics import unnormalize,unnormalize_unitary
from Statistics import MakeLag

#def test():
#    i = 6
#    test = np.array([np.arange(i),np.arange(i,i*2),np.arange(i*2,i*3),np.arange(i*3,i*4),np.arange(i*4,i*5),np.arange(i*5,i*6)])
#    print(test)
#    ex,min_,max_ = normalize(test,is_numpy=True)
#    print(ex, test,min_,max_)
#    ex = unnormalize_unitary((min_,max_),ex)
#    print(ex)
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

X_temp = unnormalize_6_rows((min_values,max_values),X_temp,True)
y_temp = unnormalize_6_rows((Answers_min_values,Answers_max_values),y_temp)

X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp,test_size=0.4)

#test()
hidden_layer_neuron = [1,(6*number_of_lag+12)/4,(6*number_of_lag+12)/2,(6*number_of_lag+12)*3/4,(6*number_of_lag+12)*2,(6*number_of_lag+12)*4,(6*number_of_lag+12)*8]
alpha = [0.001,0.003,0.005,0.008,0.01]
activation_funcions=["relu","linear","sigmoid","hard_sigmoid","tanh","elu"]
regulizers=[l2(0.01),l2(0.),l2(0.1)]
print(hidden_layer_neuron)
force_brute_tunnig((X_train,y_train),(X_test,y_test),(X_validation,y_validation),10,(6*number_of_lag,12),hidden_layer_neuron,alpha,activation_funcions,regulizers)

