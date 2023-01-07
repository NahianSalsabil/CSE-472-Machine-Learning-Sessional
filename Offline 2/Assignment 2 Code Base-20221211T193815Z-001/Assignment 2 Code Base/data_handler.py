import pandas as pd
import random
import numpy as np

def load_dataset(filename):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    df = pd.read_csv(filename)
    # print(df)
    features = df.iloc[:,0:len(df.columns)-1].to_numpy()
    
    class_vector = df.iloc[:,-1].to_numpy().reshape(len(df),1)
    
    return features, class_vector


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    if shuffle == True:
        indeces = list(range(0,len(X)))
        random.shuffle(indeces)
        X = X[indeces]
        y = y[indeces]
        
    indeces = list(range(0,len(X)))
    test_index = random.sample(indeces, int(test_size*len(X)))
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i in range(len(X)):
        for j in range(len(test_index)):
            flag = 1
            if i == test_index[j]:
                X_test.append(X[i])
                y_test.append(y[i])
                flag = 0
                break
        if flag == 1:
            X_train.append(X[i])
            y_train.append(y[i])    
                
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    print("length of train and test data: ", len(X_train), len(y_train), len(X_test), len(y_test))
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    indeces = list(range(0,len(X)))
    sample_index = random.choices(indeces, k=len(X))
    X_sample = []
    y_sample = []
    
    for j in range(len(sample_index)):
        X_sample.append(X[j])
        y_sample.append(y[j])
        
    X_sample, y_sample = np.array(X_sample), np.array(y_sample)
    
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
