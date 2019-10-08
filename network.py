from keras.models import Sequential
from keras.layers import Dense
from time import clock

import matplotlib.pyplot as plt

import math
import numpy as np
import datetime
import functools
import sys
import os

#nome do arquivo hf5 com os pesos da rede
h5_name = os.path.join(os.getcwd(),"SavedModels/Rede")

#path para os graficos
path_graphs = os.path.join(os.getcwd(),"Graphs")

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

def calculateErrorPerNeuron(values,indexes):
    acummulated_error = np.mean(np.power(values[:,3:10] - indexes[:,3:10],2))
    mse_error = np.mean(np.power(values - indexes,2))
    mape_error = np.mean(np.divide(np.absolute(values - indexes),np.absolute(indexes))*100)
    return [acummulated_error,mse_error,mape_error]
        
def train(network, train_base,test_base,val_base,batch_size=20,num_epochs_without_change=500):
    start_time = datetime.datetime.now()
    best_loss = np.inf
    Stop = False
    h=0
    num_error_starvation=0
    lenBase = len(train_base[0])
    maxLen = math.ceil(lenBase/float(batch_size))
    labels = ["Por neuronio","MSE","MAPE"]
    oldLabels = [[],[],[]]
    oldValLabels = [[],[],[]]
    oldTrainLabels = [[],[],[]]
    while(not Stop):
        
        epoch_start =  datetime.datetime.now()
        
        train_elems = train_base[0]
        train_labels = train_base[1]
        
        randomize(train_elems,train_labels)
        
        loss = [0,0]
        
        for x in range(0,maxLen):
            updateBar(x,maxLen,loss)
            x_train = train_elems[batch_size*x:batch_size*(x+1)]
            x_test = train_labels[batch_size*x:batch_size*(x+1)]

            train_loss = network.train_on_batch(x_train, x_test)
            for index in range(0, len(train_loss)-1):
                loss[index] += train_loss[index+1]
    
        results_train = network.predict(train_elems)
        tLoss = calculateErrorPerNeuron(results_train,train_labels)
        for index in range(0, len(tLoss)):
            oldTrainLabels[index].append(tLoss[index])  
        print()
               
        test_values  = test_base[0]
        test_labels  = test_base[1]
        results_test = network.predict(test_values)
        test_loss = calculateErrorPerNeuron(results_test,test_labels)
            
        for i in range(0,len(test_loss)):
            oldLabels[i].append(test_loss[i])
            
        val_values  = val_base[0]
        val_labels  = val_base[1]
        results_val = network.predict(val_values)
        val_loss = calculateErrorPerNeuron(results_val,val_labels)
        for i in range(0,len(val_loss)):
            oldValLabels[i].append(val_loss[i])
        
        newError = trysave(test_loss,network,h,best_loss)
        if(newError == best_loss):
            num_error_starvation += 1
        else:
            best_loss = newError
            num_error_starvation = 0
            
        elapsed_time = datetime.datetime.now() - start_time
        epoch_time = datetime.datetime.now() - epoch_start
        print ("Treino -> %d  Epoch | num no changes %d | total time: %s %s" % (h+1,num_error_starvation,epoch_time, elapsed_time))
    
        for i in range(0,len(labels)):
            plt.plot(oldTrainLabels[i],'r',label='Treino')
            plt.plot(oldLabels[i],'b',label='Teste')
            plt.ylabel(labels[i])
            plt.savefig(os.path.join(path_graphs,labels[i] +'.png'))
            plt.clf()
        h+=1
        Stop = num_error_starvation >= num_epochs_without_change
        
def save_weights(model,name):
    model.save_weights(str(name)+".h5") 

def trysave(test_loss,network,epoch,best_loss):        
    targetloss = test_loss[2]
    rLoss = best_loss
    if(targetloss > 0. and best_loss > targetloss):
        save_weights(network,h5_name)
        print("Saving, Old Value: " + str(best_loss) +" & New value: " +str(targetloss) + " gain : "+str(abs(best_loss - targetloss) ) )
        rLoss = targetloss
    else:
        print("Not saving, Old Value: " + str(best_loss) +" & New value: " +str(targetloss) + " lost : "+str(abs(targetloss-best_loss)) )
    return rLoss
    
def create_network(input_size,hidden_layer_neurons,output_size,optimizer):
    classifier = Sequential()

    #camada escondida
    classifier.add(Dense(units = hidden_layer_neurons, activation='linear', input_dim= input_size))
    
    #camada de saida
    classifier.add(Dense(units = output_size, activation='sigmoid'))
    
    classifier.compile( optimizer = optimizer, loss = 'MAE',metrics = ['mean_absolute_percentage_error','MSE']  )
  
    return classifier