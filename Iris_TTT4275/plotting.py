import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.mlab as mlab
#from scipy.stats import norm
import scipy.stats
from scipy.stats import norm
#maybe just import the dataset from sklearn, makes it much easier to work with. 
#could have used pandas.plot function but didnt do it now. 
#todo, optimize code, and make it easier to run. now i gets the data many times.it is slow.
#------------Variables--------------#
classes = 3
iris_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features = ['sepal_length','sepal_width','petal_length','petal_width']
path = 'iris.csv'
path_setosa = 'class_1.csv'
path_versicolour = 'class_2.csv'
path_virginica = 'class_3.csv'
#-----------^Variables^-------------#

#---------Importing data------------#
def load_data(path, one=False, maxVal=None, normalize=False, d=','): #change normalize to true to normalize the feature data
    data = pd.read_csv(path, sep=d) #reading csv file, and splitting with ","
    #data.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']#making columnnames, for easier understanding
    #data.describe()#this gives all the information you need: count, mean, std, min, 25%, 50%,75%,max
    # if one: #dont wont a column of ones when plotting
    #     lenght = len(data)
    #     #adding ones
    #     if lenght>60:

    #         data.insert(4,'Ones',np.ones(lenght),True)
        
    #     else:
    #         data['Ones'] = np.ones(lenght)
    #normalize
    t = one
    if normalize:
        data = data.divide(maxVal)

    return data



#--------^Importing data^-----------#

#-----------plotting----------------#
def plot_petal(data):
    color = ['red', 'blue', 'green'] #get different colors to the plot
    #petal_length = np.array(data['petal_length'])
    #petal_width = np.array(data['petal_width'])
    for i in range(len(data)):# iterate through the three datasets
       name = iris_names[i] #get the name for the classes from the global array, iris_names
       plt.scatter(np.array(data[i]['petal_width']),np.array(data[i]['petal_length']), label=name, color=color[i]) #plot a scatter plot,with length as x-axis and width as y-axis
    #add som useful information to plot under.
    plt.legend()
    plt.xlabel('Petal width in cm') 
    plt.ylabel('Petal length in cm')
    plt.title('Petal data')
    plt.grid('On')
    plt.savefig('petal_scatterplot_gridon_width-length.png')
    plt.show()

def plot_sepal(data):
    color = ['red', 'blue', 'green'] #get different colors to the plot
    #petal_length = np.array(data['petal_length'])
    #petal_width = np.array(data['petal_width'])
    for i in range(len(data)):# iterate through the three datasets
       name = iris_names[i] #get the name for the classes from the global array, iris_names
       plt.scatter(np.array(data[i]['sepal_width']),np.array(data[i]['sepal_length']), label=name, color=color[i]) #plot a scatter plot,with length as x-axis and width as y-axis
    #add som useful information to plot under.
    plt.legend()
    plt.xlabel('Sepal width in cm')
    plt.ylabel('Sepal length in cm')
    plt.title('Sepal data')
    #plt.grid('On')
    plt.show()

def plot_histogram(data): #change step size, to change the dimension on the histogram bars, org:0.03, used 0.003 when normalized
    sns.set()
    sns.set_style("white")
    
    # make the 'species' column categorical to fix the order
    data['species'] = pd.Categorical(data['species'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for col, ax in zip(data.columns[:4], axs.flat):
        sns.histplot(data=data, x=col, kde=True, hue='species', common_norm=False, legend=ax==axs[0,0], ax=ax)
    plt.tight_layout()
    plt.savefig('newhist_withbestfit.png',dpi=200)
    plt.show()
   
def oldhist(data,step=0.03):
    #--------making histogram basis-------#
    fig, axes = plt.subplots(nrows= 2, ncols=2, sharex='col', sharey='row')#basis for subplots
    colors= ['blue', 'red', 'green', 'black'] #colors for histogram
    max_val = np.amax(data)# Finds maxvalue in samples



    for i, ax in enumerate(axes.flat):#loop through every feature
        for label, color in zip(range(len(iris_names)), colors): #loop through every class
            #plot histogram from class[feature]
            ax.hist(data[label][features[i]], label=iris_names[label], color=color, stacked=True,alpha=0.5)
            ax.set_xlabel(features[i]+'( cm)') #add axis name
            ax.legend(loc='upper right')
        
        ax.set(xlabel='Measured [cm]', ylabel='Number of samples') #sets label name
        ax.label_outer() #makes the label only be on the outer part of the plots
        ax.legend(prop={'size': 7}) #change size of legend
        ax.set_title(f'Feature {i+1}: {features[i]}') #set title for each plot
        #ax.grid('on') #grid on or off
        
        #plt.savefig('histogram_rap.png',dpi=200)

        plt.show()






#----------^plotting^---------------#
if __name__ == '__main__':

    #----------------get data--------------------#
    tot_data = load_data(path, normalize=False)

    max_val = tot_data.max(numeric_only=True).max() #first max, gets max of every feature, second max gets max of the features
    setosa = load_data(path_setosa,max_val) 
    print(setosa.head())
    
    versicolor = load_data(path_versicolour, max_val)
    virginica = load_data(path_virginica, max_val)
    split_data_array = [setosa,versicolor,virginica]
    
    #-----------------plot------------------------#
    #plot_histogram(split_data_array)
    plot_histogram(tot_data)
    #plot_petal(split_data_array)


