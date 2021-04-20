import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
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
def load_data(path, maxVal=None, normalize=False, d=','): #change normalize to true to normalize the feature data
    data = pd.read_csv(path, sep=d) #reading csv file, and splitting with ","
    #data.columns = ['sepal_length','sepal_width','petal_length','petal_width','species']#making columnnames, for easier understanding
    #data.describe()#this gives all the information you need: count, mean, std, min, 25%, 50%,75%,max
    
    #normalize
    
    if normalize:
        data = data.divide(maxVal)

    return data



#--------^Importing data^-----------#

#---------preprocessing-------------#


#---------splitting of data-------------# ugly code, should be rewritten
#gets the individual classes
#check if this can work for splitting. 
# for vowel in vowels:
    #     train_df = train_df.append(df.loc[vowel][:train_samples])
    #     test_df = test_df.append(df.loc[vowel][train_samples:])
    # return train_df, test_df



#print(train.shape[1])
#---------splitting of data-------------#



#--------^preprocessing^------------#

#------------functions--------------#
def sigmoid(x):
    return np.array(1/(1+ np.exp(-x)))

def train():
    features = train.shape[1]
    


#-----------^functions^-------------#

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

def plot_histogram(data, step=0.03): #change step size, to change the dimension on the histogram bars, org:0.03, used 0.003 when normalized
    
    #--------making histogram basis-------#
    fig, axes = plt.subplots(nrows= 2, ncols=2, sharex='col', sharey='row')#basis for subplots
    colors= ['blue', 'red', 'green', 'black'] #colors for histogram
    max_val = np.amax(data)# Finds maxvalue in samples
    bins = np.linspace(0.0 ,int(max_val+step), num=int((max_val/step)+1), endpoint=False) #making x axis
    #-------^making histogram basis^------#
    
    
    for i, ax in enumerate(axes.flat):#loop through every feature
        for label, color in zip(range(len(iris_names)), colors): #loop through every class
            ax.hist(data[label][features[i]], bins, label=iris_names[label], color=color, alpha=0.5, stacked=True) #plot histogram from class[feature]
            
            #ax.set_xlabel(features[i]+'( cm)') #add axis name
            #ax.legend(loc='upper right')
        ax.set(xlabel='Measured [cm]', ylabel='Number of samples') #sets label name
        ax.label_outer() #makes the label only be on the outer part of the plots
        ax.legend(prop={'size': 7}) #change size of legend
        ax.set_title(f'Feature {i+1}: {features[i]}') #set title for each plot
        #ax.grid('on') #grid on or off
        

    plt.show()






#----------^plotting^---------------#
if __name__ == '__main__':

    #----------------get data--------------------#
    tot_data = load_data(path, normalize=False)
    max_val = tot_data.max(numeric_only=True).max() #first max, gets max of every feature, second max gets max of the features
    setosa = load_data(path_setosa,max_val) 
    versicolor = load_data(path_versicolour, max_val)
    virginica = load_data(path_virginica, max_val)
    split_data_array = [setosa,versicolor,virginica]
    #--------------preprocessing------------------#
    #splits it up to train and test. watch out for the index, they take their indexes with them
    train = setosa[0:30].append((versicolor[0:30], virginica[0:30]))
    test = setosa[30:].append((versicolor[30:], virginica[30:]))



    #-----------------plot------------------------#
    #plot_histogram(split_data_array)
    #plot_petal(split_data_array)

