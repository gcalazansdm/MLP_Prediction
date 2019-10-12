from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.regularizers import l2
from keras.regularizers import l1

from constants import path_graphs
from constants import h5_name

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

def force_brute_tunnig(train_base,test_base,validation_base,num_epochs_without_change,boders_neurons,hidden_layer_neuron,alpha,activation_funcions,regulizers):
    num_test = 0
    print("iniciando")
    for hl in hidden_layer_neuron:
       network = create_network(boders_neurons[0], round(hl),boders_neurons[1],Adam(alpha[0]),activation_funcions[0],activation_funcions[1],regulizers[0],regulizers[0])
       error = train(network,train_base,test_base,validation_base,num_epochs_without_change=num_epochs_without_change,printing=True)
       print("teste",num_test,"Numero de neuronios ",hl,"Error",error)
       num_test += 1
    h5_name = tempName

def train(network, train_base,test_base,val_base,batch_size=20,num_epochs_without_change=100,printing=True):
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

    directory = path_graphs
    if not os.path.exists(directory):
       os.makedirs(directory)

    while(not Stop):
        
        epoch_start =  datetime.datetime.now()
        
        train_elems = train_base[0]
        train_labels = train_base[1]
        
        randomize(train_elems,train_labels)
        
        loss = [0,0]
        
        for x in range(0,maxLen):
            if(printing):  
                updateBar(x,maxLen,loss)
            x_train = train_elems[batch_size*x:batch_size*(x+1)]
            x_test = train_labels[batch_size*x:batch_size*(x+1)]

            train_loss = network.train_on_batch(x_train, x_test)
            for index in range(0, len(train_loss)-1):
                loss[index] += train_loss[index+1]
        if(printing):  
            print()
        results_train = network.predict(train_elems)
        tLoss = calculateErrorPerNeuron(results_train,train_labels)
        for index in range(0, len(tLoss)):
            oldTrainLabels[index].append(tLoss[index])  
               
        test_loss = test_network(test_base,network)
        for i in range(0,len(test_loss)):
            oldLabels[i].append(test_loss[i])

        val_loss = test_network(val_base,network)
        for i in range(0,len(val_loss)):
            oldValLabels[i].append(val_loss[i])
        
        if(printing):  
            print ("Treino -> %d  Epoch" % (h+1))
        newError = trysave(val_loss,network,h,best_loss,printing)
        if(newError == best_loss):
            num_error_starvation += 1
        else:
            best_loss = newError
            num_error_starvation = 0
            
        elapsed_time = datetime.datetime.now() - start_time
        epoch_time = datetime.datetime.now() - epoch_start
        if(printing):  
           print("\tnum no changes %d\n\ttotal time: %s %s" %(num_error_starvation,epoch_time, elapsed_time))

        for i in range(0,len(labels)):
            plt.plot(oldTrainLabels[i],'r',label='Treino')
            plt.plot(oldLabels[i],'b',label='Teste')
            plt.plot(oldValLabels[i],'g',label='Teste')
            plt.ylabel(labels[i])
            plt.savefig(os.path.join(path_graphs,labels[i] +'.png'))
            plt.clf()
        h+=1
        Stop = num_error_starvation >= num_epochs_without_change
    return best_loss

def create_network(input_size,hidden_layer_neurons,output_size,optimizer,activation_1_layer,activation_2_layer,regulazier_1_layer,regulazier_2_layer):
    classifier = Sequential()

    #camada escondida
    classifier.add(Dense(units = hidden_layer_neurons, activation=activation_1_layer, input_dim= input_size,kernel_regularizer=regulazier_1_layer,
                activity_regularizer=l1(0.01)))
    
    #camada de saida
    classifier.add(Dense(units = output_size, activation=activation_2_layer,kernel_regularizer=regulazier_2_layer,
                activity_regularizer=l1(0.01)))
    
    classifier.compile( optimizer = optimizer, loss = 'MAE',metrics = ['mean_absolute_percentage_error','MSE']  )
  
    return classifier
