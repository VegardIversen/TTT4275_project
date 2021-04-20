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

#----------Preprocessing------------#
data = load_data(path)

#---------^Preprocessing^-----------#

#------------Processing-------------#
def sigmoid(x):
    return 1/(1+np.exp(-x))

def accumulateMSE(gk, tk):
    return 0.5*np.matmul((gk-tk).T, (gk-tk))

def get_gk(W, xk):
### Equation 20
    zk = np.matmul(W, xk)
    return sigmoid(zk)

def get_grad_gk_MSE(gk, tk):
    return gk - tk

def get_grad_zk_g(gk):
    return np.matmul(gk, (1-gk))

def get_grad_W_zk(xk, features):
    return xk.reshape(1, features + 1)

def accumulate_grad_W_MSE(x, y, z):
    return np.matmul(np.multiply(x, y), z)

def train(data, iterations, alpha, classes, features):

    MSE_list = []

    gk = np.zeros(classes)
    gk[0] = 1                                 #?
    tk = np.zeros((classes, 1))
    W  = np.zeros((classes, (features+1)))

    for i in range(iterations):

        grad_W_MSE = 0
        MSE = 0

        for xk in data:
            #matmul = normal multiplication, #multiply = element-wise multiplication
            xk.reshape(features + 1, 1)
            gk = get_gk(W, xk)

            #Eq. 22
            grad_gk_MSE = get_grad_gk_MSE(gk, tk)
            grad_zk_g   = get_grad_zk_g(gk)
            grad_W_zk   = get_grad_W_zk(xk, features)   #xk is the shape of 5x1 |x1k| which transposes to |x1k x2k x3k x4k 1|
                                                        #                       |x2k|
                                                        #                       |x3k|
                                                        #                       |x4k|
                                                        #                       |1  |

            grad_W_MSE += accumulate_grad_W_MSE(grad_gk_MSE, grad_zk_g, grad_W_zk)
        
            #Eq. 19
            MSE += accumulateMSE(gk, tk)

    MSE_list.append(MSE)

#-----------^Processing^------------#