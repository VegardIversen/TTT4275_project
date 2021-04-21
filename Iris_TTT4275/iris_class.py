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
        self.features = len(self.list_of_features) +1
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

    def grad_gk_mse_f(self, g_k, t_k):
        grad = np.multiply((g_k-t_k),g_k)
        return grad

    def grad_W_zk_f(self, x):
        grad = x.reshape(1,self.features)
        return grad
    
    def grad_W_MSE_f(self, g_k, grad_gk_mse, grad_W_zk):
        return np.matmul(np.multiply(grad_gk_mse,(1-g_k)),grad_W_zk)

    def MSE_f(self, g_k,t_k):
        return 0.5*np.matmul((g_k-t_k).T,(g_k-t_k))



    def train_model(self):
        print(f'Number of iterations {self.iterations}') 
        self.g_k[0] = 1
        
        for i in range(self.iterations):
            grad_W_MSE = 0
            MSE = 0
            k = 0
            
            for j, x in enumerate(self.train): #isnt really necessary to use enumerate, see if i should change
                if j%30==0 and j!=0:
                    k += 1
                self.g_k = self.sigmoid(np.matmul(self.weigths,x.reshape(self.features,1)))

                MSE += self.MSE_f(self.g_k,self.t_k[k])

                grad_gk_mse = self.grad_gk_mse_f(self.g_k,self.t_k[k])
                grad_W_zk = self.grad_W_zk_f(x)
                
                grad_W_MSE += self.grad_W_MSE_f(self.g_k, grad_gk_mse, grad_W_zk)
            
            self.mses[i] = MSE[0]
            self.weigths = self.weigths-self.alpha*grad_W_MSE

            if(100*i /self.iterations) % 10 == 0: #printer bare til 90 bare 90%
                
                print(f"\rProgress passed {100 * i / self.iterations}%", end='\n')
                
        
        print(f"\rProgress passed {(i+1)/self.iterations *100}%", end='\n')
        print('Done')
        return self.weigths

                

    def fit(self,):
        pass
    
    def error(self,):
        pass
    def confusion_matrix(self,):
        pass


classes = 3
iris_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features = ['sepal_length','sepal_width','petal_length','petal_width']
path = 'iris.csv'
path_setosa = 'class_1.csv'
path_versicolour = 'class_2.csv'
path_virginica = 'class_3.csv'

def load_data(path, one=True, maxVal=None, normalize=False, d=','): #change normalize to true to normalize the feature data
    data = pd.read_csv(path, sep=d) #reading csv file, and splitting with ","
    #data.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']#making columnnames, for easier understanding
    #data.describe()#this gives all the information you need: count, mean, std, min, 25%, 50%,75%,max
    if one: #dont wont a column of ones when plotting
        lenght = len(data)
        #adding ones
        if lenght>60:

            data.insert(4,'Ones',np.ones(lenght),True)
        
        else:
            data['Ones'] = np.ones(lenght)
    #normalize
    if normalize:
        data = data.divide(maxVal)

    return data


if __name__ == '__main__':

    #----------------get data--------------------#
    tot_data = load_data(path, normalize=False)

    max_val = tot_data.max(numeric_only=True).max() #first max, gets max of every feature, second max gets max of the features
    setosa = load_data(path_setosa,max_val) 
    
    versicolor = load_data(path_versicolour, max_val)
    virginica = load_data(path_virginica, max_val)
    split_data_array = [setosa,versicolor,virginica]

    train = pd.concat([setosa[0:30],versicolor[0:30],virginica[0:30]])
    test = pd.concat([setosa[30:],versicolor[30:],virginica[30:]])
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]])
    t_k_test = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]])

    #just making dataframe to numpy array
    train = train.to_numpy()
    test = test.to_numpy()

    w1 = LCD(train,test,t_k,2000,0.01, features)
    weigths1 = w1.train_model()
    print('\n')
    print(weigths1)