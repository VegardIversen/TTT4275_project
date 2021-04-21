import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#to make a LCD, we take in a training, test an r_k list, and number of iteraterions, alpha and list of features
#make the class, d = LCD(train,test,t_l,iterations, alpha, list_of_features)
#then use w = d.train()
#could use @dataclass to not use self
class LCD:
    def __init__(self, train, test, t_k, iterations, alpha, list_of_features):
        #attributes under
        self.train = train
        self.test = test
        self.t_k = t_k
        self.iterations = iterations
        self.alpha = alpha
        self.list_of_features = list_of_features
        self.features = len(self.list_of_features)
        self.classes = 3 #could have this as an input but didnt bother, shouldnt change
        self.weigths = np.zeros((self.classes,self.features))
        self.g_k = np.zeros(self.classes)
        self.mses = np.zeros(self.iterations)
    
        
    def set_iterations(self, iterations):
        self.iterations = iterations
    
    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test
    
    def set_train_test(self,train,test):
        self.train = train
        self.test = test
    
    def set_tk(self, tk):
        self.t_k = tk
    
    def set_list_of_features(self, list_of_features):
        self.list_of_features = list_of_features
    
    def set_num_of_classes(self,classes):
        self.classes = classes

    def get_iterations(self):
        return self.iterations
    
    def get_alpha(self):
        return self.alpha

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def get_train_test(self):
        return self.train, self.test


    def get_tk(self):
        return self.t_k

    def get_list_of_features(self):
        return self.list_of_features

    def get_num_of_classes(self):
        return self.classes

    
    def sigmoid(self, x):

        return np.array(1/(1+ np.exp(-x)))

    def sigmoid2(self, x, w):
        return 1/(1+np.exp(-np.matmul(w,x)))

    def grad_gk_mse(self, g_k, t_k):
        grad = np.multiply((g_k-t_k),g_k)
        return grad

    def grad_W_zk(self, x):
        grad = x.reshape(1,self.features)
        return grad
    

    def train_model(self): 
        self.g_k[0] = 1

        for i in range(self.iterations):
            grad_W_MSE = 0
            MSE = 0
            k = 0
            for j, x in enumerate(self.train): #isnt really necessary to use enumerate, see if i should change
                if j%30==0 and j!=0:
                    k += 1
                self.g_k = self.sigmoid(np.matmul(self.weigths,x.reshape(self.features,1)))

                MSE += 0.5*np.matmul()
                grad_gk_mse = self.grad_gk_mse(self.g_k,self.t_k[k])
                

    def fit(self,):
        pass
    
    def error(self,):
        pass
    def confusion_matrix(self,):
        pass


# t = [1,1,1,1,1]
# t1 = [1,1,1,1,1]
# x = LCD(1,t1,2,1,[])
# print(x.sigmoid())