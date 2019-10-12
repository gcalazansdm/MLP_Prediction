from constants import h5_name
from constants import path_graphs
from constants import min_values
from constants import max_values
from constants import Answers_min_values
from constants import Answers_max_values
from Statistics import normalize_6_rows
from Statistics import unnormalize_6_rows

from time import clock

import numpy as np

import functools
import sys
import os

def calculateErrorPerNeuron(values,indexes):
    acummulated_error = np.mean(np.power(values[:,3:10] - indexes[:,3:10],2))
    mse_error = np.mean(np.power(values - indexes,2))
    mape_error = np.mean(np.divide(np.absolute(values - indexes),np.absolute(indexes))*100)
    return [acummulated_error,mse_error,mape_error]

def test_network(base,network,normalize = True):
    normalized_values = None
    if(normalize):
        normalized_values = normalize_6_rows((min_values,max_values),base[0])
    else:
        normalized_values = base[0]
    labels  = base[1]
    results = network.predict(normalized_values)
    test_loss = None
    if(normalize):  
        unnormalized_results = unnormalize_6_rows((Answers_min_values,Answers_max_values),results)
        test_loss = calculateErrorPerNeuron(unnormalized_results,labels)
    else:
        test_loss = calculateErrorPerNeuron(results,labels)
    return test_loss
    
def randomize(dataset, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(dataset.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        functools.reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])
        
def updateBar(progress,toolbar_width,loss,start=0):
    slices = int(float(progress)/toolbar_width * 50)
    write = "[%s%s]" % ("#"*slices," " * (50-slices))
    if(loss is not None):
        if(float(progress) > 0):
            write += " -> ("
            for elem in loss:
                write += "%f," % (float(elem)/progress)
            write += "\b)"
    else:
        write += " -> %d/%d" % (progress,toolbar_width)
    write += " %s" % secondsToStr(clock() - start)
    sys.stdout.write(write)
    sys.stdout.flush()
    sys.stdout.write("\b" * (len(write)+1)) # return to start of line, after '['

def save_weights(model,name):
    file_ = str(name)+".h5"
    directory = os.path.split(file_)[0]
    if not os.path.exists(directory):
       os.makedirs(directory)
    model.save_weights(str(name)+".h5") 

def trysave(test_loss,network,epoch,best_loss,printing=False):        
    targetloss = test_loss[2]
    rLoss = best_loss
    if(targetloss > 0. and best_loss > targetloss):
        save_weights(network,h5_name)
        if(printing):
            print("Saving, Old Value: " + str(best_loss) +" & New value: " +str(targetloss) + " gain : "+str(abs(best_loss - targetloss) ) )
        rLoss = targetloss
    elif(printing):
        print("Not saving, Old Value: " + str(best_loss) +" & New value: " +str(targetloss) + " lost : "+str(abs(targetloss-best_loss)) )
    return rLoss
    
