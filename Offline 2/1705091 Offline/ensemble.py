from data_handler import bagging_sampler
import copy


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = []
        for i in range(n_estimator):
            self.base_estimator.append(copy.deepcopy(base_estimator))
        self.n_estimator = n_estimator
        

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        
        # for each base_estimator in self.base_estimator fit it on a bagged sample
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.base_estimator[i].fit(X_sample, y_sample)
        return self

    def majority_vote(self, y):
        """
        :param y:
        :return:
        """
        # todo: implement
        # return the majority vote of y
        count = 0
        for i in range(len(y)):
            if y[i] == 1:
                count += 1
        if count > len(y)/2:
            return 1
        else:
            return 0
            
        
    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        # predict label for each data point using each base_estimator in self.base_estimator
        predictions = []
        for i in range(self.n_estimator):
            predictions.append(self.base_estimator[i].predict(X))
        
        # apply majority voting
        final_predictions = []
        for i in range(len(predictions[0])):
            final_predictions.append(self.majority_vote([predictions[j][i] for j in range(self.n_estimator)]))
        
        return final_predictions
            
