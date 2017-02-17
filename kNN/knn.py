import csv
import re
import time
import math
import heapq
import string
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import Counter, defaultdict
from sklearn.cross_validation import train_test_split

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../naive_bayes'))

from data_cleaner import clean_data_pipeline

TRAIN_FILE_INPUT = '../data/train_input.csv'
TEST_INPUT_FILEPATH = '../data/test_input.csv'
TRAIN_LABELS_FILEPATH = '../data/train_output.csv'

class KNearestNeighbours():
    def _calculate_wordbags_distance(self,word_bag1, word_bag2):
        return 1.0/(len(word_bag2.intersection(word_bag1)) + 0.0001)

    def _get_majority(self,elems):
        ignore_distances = [elem[0] for elem in elems]
        return Counter(ignore_distances).most_common()[0][0]

    def _get_weighted_majority(self,elems):
        dct = defaultdict(float)
        for category, distance in elems:
            dct[category] = dct[category] + (1.0/distance)
        return max(dct, key=dct.get)

    def _calculate_dicts_distance(self, dict_tuple1, dict_tuple2):
        distance = math.sqrt(sum((dict_tuple1[0].get(k, 0) - dict_tuple2[0].get(k, 0))**2 for k in dict_tuple2[1].union(dict_tuple1[1])))
        return 1.0/(0.0001 + distance)

    def __init__(self, binarize = True, uniform_weight = True, k = 3):
        self.binarize = binarize
        self.k = k
        self.decide_majority_func = self._get_majority if uniform_weight else self._get_weighted_majority
        self.vectorizer_func = set if binarize else Counter
        self.distance_func = self._calculate_wordbags_distance if binarize else self._calculate_dicts_distance

    def fit(self, X ,Y):
        self._X_fit = zip(range(len(X)), map(self.vectorizer_func, X))
        self._Y_fit = Y


    def _knn_predict_category(self, x, k = 5):
        x_vect = self.vectorizer_func(x)
        distances = [self.distance_func(x_vect, other_x[1]) for other_x in self._X_fit]
        k_highest_indices = np.argpartition(distances, k)[:k]
        k_highest_categs = [(self._Y_fit[self._X_fit[i][0]], distances[i]) for i in k_highest_indices]
        prediction = self.decide_majority_func(k_highest_categs)
        return prediction

    def predict(self,to_predict, k = None):
        k_param = k
        if not k:
            k_param = self.k
        return [self._knn_predict_category(x, k = k_param) for x in to_predict]

def do_cross_val_for_k(min_k = 3, max_k = 11, count = True, verbose = True):
    data = clean_data_pipeline(pd.read_csv(TRAIN_FILE_INPUT))
    category_mapping = list(pd.read_csv(TRAIN_LABELS_FILEPATH)['category'])

    train_data, test_data, train_labels, test_labels = train_test_split(data, category_mapping,test_size=0.05, random_state=105, stratify = category_mapping)

    clf = KNearestNeighbours(k = min_k, uniform_weight = False)
    clf.fit(train_data, train_labels)

    best_k = 0
    best_accuracy = 0
    accuracies = []
    for k_param in range(min_k, max_k + 1):
        predicted_labels = clf.predict(test_data)
        accuracy = np.mean(np.array(predicted_labels) == test_labels)
        accuracies.append(accuracy)

        if verbose:
            print '\nresults for k:', (k_param)
            print 'accuracy:', accuracy
            print metrics.classification_report(test_labels, predicted_labels)
            print metrics.confusion_matrix(test_labels, predicted_labels)

        if accuracy > best_accuracy:
            best_k = k_param
            best_accuracy = accuracy

        clf.k = k_param

    return best_k, best_accuracy, accuracies

def plot_accuracies(min_k = 3, max_k = 20, action = 'display', filename = None):
    best_k, best_acc, accuracies = do_cross_val_for_k(3,20)
    print best_k, best_acc, accuracies
    plt.plot(range(3,20+1), accuracies)
    plt.xlabel("k value")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs K")
    if action == 'save':
        plt.savefig("accuracy growth.png")
    elif action == 'display':
        plt.show()
    else:
        raise ValueError("action not recognized")

def predict_and_print(k = 19, uniform_weight = False):
    clf = KNearestNeighbours(k = 19, uniform_weight = False)
    to_predict = clean_data_pipeline(pd.read_csv(TEST_INPUT_FILEPATH))
    full_train_data = clean_data_pipeline(pd.read_csv(TRAIN_FILE_INPUT))
    full_train_labels = list(pd.read_csv(TRAIN_LABELS_FILEPATH)['category'])

    clf.fit(full_train_data, full_train_labels)

    predicted = clf.predict(to_predict)

    print "id,category"
    for i, line in enumerate(predicted):
        print "{},{}".format(i, line)

def save_confusion_matrix(matrix):
    categs = ['hockey','movies','nba','news','nfl','politics','soccer','worldnews']
    df_cm = pd.DataFrame(matrix, index = categs,columns = categs)
    plt.figure(figsize = (10,7))
    plt.title("Confusion Matrix: KNN classifier")
    sn.heatmap(df_cm, annot=True, fmt = 'g')
    plt.savefig("Confusion Matrix")

def read_train_data_for_classifier():
    return clean_data_pipeline(pd.read_csv(TRAIN_FILE_INPUT))

def read_labels_for_classifier():
    return list(pd.read_csv(TRAIN_LABELS_FILEPATH)['category'])

def read_test_data_for_classifier():
    return clean_data_pipeline(pd.read_csv(TEST_INPUT_FILEPATH))

predict_and_print()

    

