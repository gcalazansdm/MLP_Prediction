from keras.models import Sequential
from keras.layers import Dense

from constants import path_graphs

from utils import randomize
from utils import updateBar
from utils import calculateErrorPerNeuron
from utils import test_network
from utils import trysave

import matplotlib.pyplot as plt
import numpy as np

import math
import datetime
import os

        
def train(network, train_base,test_base,val_base,batch_size=20,num_epochs_without_change=100):
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
               
        test_loss = test_network(test_base,network)
        for i in range(0,len(test_loss)):
            oldLabels[i].append(test_loss[i])
            
        val_loss = test_network(val_base,network)
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
        

def create_network(input_size,hidden_layer_neurons,output_size,optimizer):
    classifier = Sequential()

    #camada escondida
    classifier.add(Dense(units = hidden_layer_neurons, activation='linear', input_dim= input_size))
    
    #camada de saida
    classifier.add(Dense(units = output_size, activation='sigmoid'))
    
    classifier.compile( optimizer = optimizer, loss = 'MAE',metrics = ['mean_absolute_percentage_error','MSE']  )
  
    return classifier