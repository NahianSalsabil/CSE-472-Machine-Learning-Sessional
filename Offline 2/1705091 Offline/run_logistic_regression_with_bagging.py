"""
main code that you will run
"""

from linear_model import LogisticRegression
from ensemble import BaggingClassifier
from data_handler import load_dataset, split_dataset, load_test_dataset
from metrics import precision_score, recall_score, f1_score, accuracy
import sys

if __name__ == '__main__':
    # data load
    X_train, y_train = load_dataset(sys.argv[1])
    X_test, y_test = load_test_dataset(sys.argv[2])
    X_test2, y_test2 = load_test_dataset(sys.argv[3])
    # split train and test
    # X_train, y_train, X_test, y_test = split_dataset(X, y, test_size=0.2, shuffle=True)

    # training
    params = dict()
    base_estimator = LogisticRegression(params)
    classifier = BaggingClassifier(base_estimator=base_estimator, n_estimator=9)
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))

    y_pred = classifier.predict(X_test2)
    print("Test Dataset 2:")
    print('Accuracy ', accuracy(y_true=y_test2, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test2, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test2, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test2, y_pred=y_pred))