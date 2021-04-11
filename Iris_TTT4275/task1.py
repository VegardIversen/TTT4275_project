import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#maybe just import the dataset from sklearn, makes it much easier to work with. 
#------------Variables--------------#
C = 3
D = 4
iris_names = ['Setosa', 'Versicolour', 'Virginica']
path = './Iris_TTT4275/iris.csv'
#-----------^Variables^-------------#

#---------Importing data------------#
def load_data(path, d=','):
    data = pd.read_csv(path, sep=',') #reading csv file, and splitting with ","
    #data.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']#making columnnames, for easier understanding
    #data.describe()#this gives all the information you need: count, mean, std, min, 25%, 50%,75%,max
    return data
#--------^Importing data^-----------#

#---------------Math----------------#
def sigmoid(x):
    return 1/(1+np.exp(-x))
#--------------^Math^---------------#

#----------Preprocessing------------#
data = load_data(path)

#---------^Preprocessing^-----------#

#------------Processing-------------#
def train(data, iterations, alpha, classes, features):

    MSE_list = []

    gk = np.zeros(classes)
    gk[0] = 1                                   #?
    tk = np.zeros((classes, 1))
    W  = np.zeros((classes, (features+1)))

    for i in range(iterations):

        grad_W_MSE = 0
        MSE = 0

        for xk in data:

            #Eq. 20
            zk = np.matmul(W, xk)
            gk = sigmoid()                          #Nani

            #Eq. 22
            grad_gk_MSE = gk - tk
            grad_zk_g   = np.matmul(gk, (1-gk))
            grad_W_zk   = xk.reshape(features, 1)   #WHAT IS THE SHAPE OF XK?

            grad_W_MSE += 
        
            #Eq. 19
            MSE += 0.5*np.multiply((gk-tk).reshape(), (gk-tk))

    MSE_list.append(MSE)



    #g = Wx
    #From eq. 20
    # MSE_list = []

    # gk = np.zeros(C)       
    # gk[0] = 1              #
    # tk = np.zeros((C, 1))  #The target variable of a dataset is the feature of a dataset about which you want to gain a deeper understanding
    # W = np.zeros((C, 1))  #

    # for i in range(iterations):

    #     grad_W_MSE = 0
    #     MSE = 0

    #     for xk in data:

    #         #Eq. 20: zk = W*xk


    #         gk = sigmoid(np.matmul(W,xk))

    #         tk = 

    #         # Eq. 22
    #         grad_gk_MSE = np.multiply((gk-tk), gk) #(gk_t_k)*g_k
    #         grad_W_zk = xk.reshape() #x_k is an array, we want it to be reshaped into 

    #         #np.multiply(x1,x2,..,xN) element-wise multiplication
    #         #np.matmul(x1,x2,...,xN) 
    #         grad_W_MSE += np.matmul(np.multiply(grad_gk_MSE, 1-gk), grad_W_zk) #Eq. 22

    #         MSE += 0.5*np.matmul(gk-tk).T, (gk-tk))
    #     MSE_list.append(MSE[0])


    # # Eq. 23
    # W = W - alpha*grad_W_MSE 
    # return W

#-----------^Processing^------------#
print(len(train()[0]))