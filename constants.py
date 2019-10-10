import os

#nome do arquivo hf5 com os pesos da rede
h5_name = os.path.join(os.getcwd(),"SavedModels/Rede")

#path para os graficos
path_graphs = os.path.join(os.getcwd(),"Graphs")

#pega o path
dataset_path='Data/Dados UBE.csv'
answers_path='Data/UBE_Results.csv'

#determina o numero de defasagens para o problema em questão
number_of_lag=8

#Determina um limiar de defasagem para os valores minimos para casos onde os valores sejam diferentes do que o esperado
min_lag=.8
max_lag=1

#definie os limiares para a padrinização
std_min = 0.3
std_max = 0.7

#armazena os Valores de min e max
min_values = []
max_values = []
#armazena os Valores de min e max para as respostas
Answers_min_values = []
Answers_max_values = []

#labels
labels = ["ATP","RBG","BLS","SFB","FZB","UBE"]
