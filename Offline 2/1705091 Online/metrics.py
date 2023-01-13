"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""
def calculate_data(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    #calclate true positive, false positive, true negative, false negative
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    positive = 0
    negative = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] == 1:
            true_positive += 1
        if y_true[i] == y_pred[i] == 0:
            true_negative += 1 
        if y_true[i] == 0 and y_pred[i] == 1:
            false_positive += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            false_negative += 1
        if y_true[i] == 1:
            positive += 1
        if y_true[i] == 0:
            negative += 1
    
    return true_positive, true_negative, false_positive, false_negative
    

def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positive, true_negative, false_positive, false_negative = calculate_data(y_true, y_pred)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return accuracy
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positive, true_negative, false_positive, false_negative = calculate_data(y_true, y_pred)
    precision = true_positive / (true_positive + false_positive)
    return precision


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    true_positive, true_negative, false_positive, false_negative = calculate_data(y_true, y_pred)
    recall = true_positive / (true_positive + false_negative)
    return recall


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
