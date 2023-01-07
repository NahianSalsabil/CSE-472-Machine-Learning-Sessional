import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.weights = np.random.rand(5,1)
        # print("initial weights: ", self.weights)

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        # Add a column of ones to X
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        
        learning_rate = 0.01
        
        for i in range(1500):
            # calculate Theta transpose * X
            theta_transpose_X = np.dot(X, self.weights)  
            
            # calculate sigmoid of theta transpose * X
            h_theta = 1/(1+np.exp(-theta_transpose_X))
            
            diff = h_theta - y
            for j in range(len(X[0])):
                # calculate weight update
                weights_update = learning_rate * np.dot(X[:,j].T, diff)
                self.weights[j] = self.weights[j] - weights_update
            
            # calculate cost function
            # try:
            #     J_theta = (-1/len(X)) * np.sum(y*np.log(h_theta) + (1-y)*np.log(1-h_theta))
            # except ZeroDivisionError:
            #     J_theta = 9999999999
            #     print("dhuksi")
            # except ValueError:
            #     J_theta = 9999999999
            # print("cost: ", J_theta)
            

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # predict label for each data point
        # calculate Theta transpose * X
        theta_transpose_X = np.dot(X, self.weights) 
        
        # calculate sigmoid of theta transpose * X
        h_theta = 1/(1+np.exp(-theta_transpose_X))
        
        predictions = []
        for i in range(len(h_theta)):
            if h_theta[i] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return predictions
            
            
            
