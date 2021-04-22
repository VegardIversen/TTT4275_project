import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

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
        self.class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'] #could have this as an input to generlize class
        self.features = len(self.list_of_features) +1
        self.classes = 3 #could have this as an input but didnt bother, shouldnt change
        self.weigths = np.zeros((self.classes,self.features))
        self.g_k = np.zeros(self.classes)
        self.mses = np.zeros(self.iterations)
        self.confusion_matrix = np.zeros((self.classes,self.classes))
    
        
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

    def test_model(self, test=None): #or call this def fit(), to be simular as other lib.
        
        if(np.all((self.weigths==0 ))):
            print('You need to train the model first')
            return False
        if test is None:
            test = self.test
        else:
            print(test)
            print('Testing model with training set')
            print('Resetting confusion matrix')

            self.confusion_matrix = np.zeros((self.classes,self.classes))

        for clas, test_set in enumerate(test):
            for row in test_set:
                prediction = np.argmax(np.matmul(self.weigths,row))
                self.confusion_matrix[clas,prediction] += 1

        return self.confusion_matrix

    def print_confusion_matrix(self):
        print(self.confusion_matrix)
        dia_sum = 0
        for i in range(len(self.confusion_matrix)):
            dia_sum += self.confusion_matrix[i, i]
        error = 1 - dia_sum / np.sum(self.confusion_matrix)
        print(f'error rate = {100 * error:.1f}%')

    def plot_confusion_matrix(self, name='ok', save=False):
        dia_sum = 0
        for i in range(len(self.confusion_matrix)):
            dia_sum += self.confusion_matrix[i, i]
        error = 1 - dia_sum / np.sum(self.confusion_matrix)

        df_cm = pd.DataFrame(self.confusion_matrix, index = [i for i in self.class_names],
                  columns = [i for i in self.class_names])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
        plt.title(f'Confusion matrix for Iris task\n iteration: {self.iterations}, alpha: {self.alpha}.\n error rate = {100 * error:.1f}%')
        if save:
            plt.savefig(f'./figurer/confusionmatrixIris20_{name}_it{self.iterations}_alpha{self.alpha}.png',dpi=200)
        else:
            plt.show()
    
    def plot_MSE(self, save=False, log=False):
        plt.plot(self.mses)
        plt.title(f'MSE for Iris task\n iteration: {self.iterations}, alpha: {self.alpha}.')
        plt.xlabel('Iteration')
        plt.ylabel('Mean square error')
        plt.grid('on')
        if log:
            plt.xscale('log')
        if save:
            plt.savefig(f'mse_it{self.iterations}_alpha{self.alpha}.png',dpi=200)
        else:
            plt.show()

def plot_mses_array(arr, alphas, name='ok', save=False):
    a = 0
    alpha = r'$ \alpha $'
    for i in arr:
        plt.plot(i,label=f'{alpha}={alphas[a]}')
        a += 1

    plt.title('Mean square error for all test')
    plt.grid('on')
    plt.xlabel('Iteration')
    plt.ylabel('Mean square error')
    plt.legend(loc=1)
    if save:
        plt.savefig(f'./figurer/MSE_all_last30_{name}.png', dpi=200)
    else:
        plt.show()
    
    




    



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

classes = 3
iris_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features = ['sepal_length','sepal_width','petal_length','petal_width']
path = 'iris.csv'
path_setosa = 'class_1.csv'
path_versicolour = 'class_2.csv'
path_virginica = 'class_3.csv'

#----------------get data--------------------#
tot_data = load_data(path, normalize=False)
max_val = tot_data.max(numeric_only=True).max() #first max, gets max of every feature, second max gets max of the features
setosa = load_data(path_setosa,max_val) 
versicolor = load_data(path_versicolour, max_val)
virginica = load_data(path_virginica, max_val)
#---------------^get data^-------------------#
alphas = [1,0.1,0.01,0.001,0.0001,0.00001]
#alphas = [0.01]
def task1a():
    train_size = 30
    arr= []
    
    #----------------prepros data--------------------#
    #split_data_array = [setosa,versicolor,virginica] #not necessary

    #splitting up in test and train sets
    train = pd.concat([setosa[0:train_size],versicolor[0:train_size],virginica[0:train_size]])
    train_for_test = np.array([setosa[0:train_size],versicolor[0:train_size],virginica[0:train_size]])
    test = np.array([setosa[train_size:],versicolor[train_size:],virginica[train_size:]]) #could mb have done this for train to, 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) #making array to check whats the true class is
    #just making dataframe to numpy array
    train = train.to_numpy()
    #---------------^prepros data^-------------------#

    for i in range(len(alphas)):
        model = f'w{i}'
        model = LCD(train,test,t_k,2000,alphas[i], features)
        
        model.train_model()
        arr.append(model.mses)
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test', save=False)
        model.test_model(train_for_test)
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train', save=False)
        

    plot_mses_array(arr, alphas, name='test', save=False)
    # print('Starting with train set')
    # for i in range(len(alphas)):
    #     model = f'wt{i}'
    #     model = LCD(train,test,t_k,2000,alphas[i], features)
        
    #     model.train_model()
    #     model.test_model(test=train_for_test)
    #     model.print_confusion_matrix()
    #     model.plot_confusion_matrix(name='train', save=False)
        
    
    #plot_mses_array(arr_test, alphas, name='test', save=True)
    #plot_mses_array(arr_train, alphas, name='train', save=True)

def task1d():
    train_size = 30
    arr = []
   
    #----------------prepros data--------------------#
    #split_data_array = [setosa,versicolor,virginica] #not necessary

    #splitting up in test and train sets
    train = pd.concat([setosa[train_size:],versicolor[train_size:],virginica[train_size:]])
    train_for_test = np.array([setosa[train_size:],versicolor[train_size:],virginica[train_size:]])
    test = np.array([setosa[0:train_size],versicolor[0:train_size],virginica[0:train_size]]) #could mb have done this for train to, 
    t_k = np.array([[[1],[0],[0]],[[0],[1],[0]],[[0],[0],[1]]]) #making array to check whats the true class is
    #just making dataframe to numpy array
    train = train.to_numpy()
    #---------------^prepros data^-------------------#

    for i in range(len(alphas)):
        model = f'w{i}'
        model = LCD(train,test,t_k,2000,alphas[i], features)
        
        model.train_model()
        model.test_model()
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='test', save=True)
        
        arr.append(model.mses)
    
    plot_mses_array(arr, alphas, name='ok', save=False)

    for i in range(len(alphas)):
        model = f'wt{i}'
        model = LCD(train,test,t_k,2000,alphas[i], features)
        
        model.train_model()
        model.test_model(test=train_for_test)
        model.print_confusion_matrix()
        model.plot_confusion_matrix(name='train', save=True)

    



if __name__ == '__main__':
    task1a()
    #task1d()