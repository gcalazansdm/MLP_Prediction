import pandas as reader
import numpy as np
import functools



def MakeLag(array,ArrayName,alpha):
    original = reader.DataFrame(data={ArrayName:array[ArrayName]})
    rVal = original
    for i in range(1, alpha):
        rVal = reader.concat([reader.DataFrame(data={ArrayName+"_"+str(i-1):original.shift(i)[ArrayName]}), rVal], axis=1).dropna()
   
    return rVal

    
def normalize_equation(outliers,alpha,elem):
    return np.multiply(np.divide(np.subtract(elem,outliers[0]),np.subtract(outliers[1],outliers[0])),alpha[0])+(alpha[1]-alpha[0])

def normalize(array,lag,alpha):
    min_lag,max_lag = lag
    array.rename_axis(None, inplace=True)
    min = array.min().min()*min_lag
    max = array.max().max()*max_lag
    rValue = array.apply(functools.partial(normalize_equation,(min,max),alpha))
    return rValue,min,max

def unnormalize(elem,outliers,alpha):
    return np.multiply(np.divide(np.subtract(elem,(alpha[1]-alpha[0])),alpha[0]),np.subtract(outliers[1],outliers[0])) + outliers[0]
