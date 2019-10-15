import pandas as reader
import numpy as np
import functools

from constants import min_lag
from constants import max_lag
from constants import std_min
from constants import std_max
from constants import number_of_lag

	
def MakeLag(array,ArrayName,alpha):
    original = reader.DataFrame(data={ArrayName:array[ArrayName]})
    rVal = original
    for i in range(1, alpha):
        rVal = reader.concat([reader.DataFrame(data={ArrayName+"_"+str(i-1):original.shift(i)[ArrayName]}), rVal], axis=1).dropna()
   
    return rVal
    
def normalize_equation(outliers,elem):
    return np.multiply(np.divide(np.subtract(elem,outliers[0]),np.subtract(outliers[1],outliers[0])),(std_max-std_min))+std_min

def normalize_6_rows(outliers,elem):
    temp_value = None
    min_values = outliers[0]
    max_values = outliers[1]
    for i in range(0,len(min_values)):
        unormalized_vector = elem[:,i*number_of_lag:(i+1)*number_of_lag]
        normalized_vector = normalize_equation((min_values[i],max_values[i]),unormalized_vector)
        if(temp_value is None):
            temp_value = normalized_vector
        else:
            temp_value = np.concatenate((temp_value,normalized_vector),axis=1)   
    return temp_value
    
def normalize_(outliers, array,is_numpy=False):
    if(not is_numpy):
        array.rename_axis(None, inplace=True)
    min = outliers[0]
    max = outliers[1]
    if(is_numpy):
        rValue = np.apply_along_axis(functools.partial(normalize_equation,(min,max)),0,array)
    else:    
        rValue = array.apply(functools.partial(normalize_equation,(min,max)))
    return rValue

def normalize(array,is_numpy_=False):
    min = array.min().min()*min_lag
    max = array.max().max()*max_lag
    return normalize_((min,max),array,is_numpy_),min,max

def unnormalize_6_rows(outliers,normalized_vector,lag=False):
    temp_value = None
    min_values = outliers[0]
    max_values = outliers[1]
    for i in range(0,len(min_values)):
        if(lag):
            unormalized = unnormalize( (min_values[i],max_values[i]) ,normalized_vector[:,i*number_of_lag:(i+1)*number_of_lag])
        else:
            unormalized = unnormalize( (min_values[i],max_values[i]) ,normalized_vector[:,i:(i+1)])                
        if(temp_value is None):
            temp_value = unormalized
        else:        
            temp_value = np.concatenate((temp_value,unormalized),axis=1)
    return temp_value
    
def unnormalize_unitary(outliers,elem):
    return (elem-std_min)/(std_max-std_min)*(outliers[1]-outliers[0]) + outliers[0]
    
def unnormalize(outliers,elem):
    return np.multiply(np.divide(np.subtract(elem,std_min),std_max-std_min),np.subtract(outliers[1],outliers[0])) + outliers[0]
