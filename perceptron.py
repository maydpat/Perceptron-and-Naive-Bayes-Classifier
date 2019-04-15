#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019
    Implementation by: Mayur Patil
    School: Purdue University

"""

from classifier import BinaryClassifier
from utils import get_feature_vectors
import numpy as np
import random as random

class Perceptron(BinaryClassifier):
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args = args

    def findSign(self, calculation):
        if calculation <= 0:
            return -1
        else:
            return 1

    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        self.weights = []
        for i in range(self.args.f_dim):
            self.weights.append(0)
        self.bias = 0
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        random.seed(5) #this line is to ensure that you get the same shuffled order everytime
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        predictionTestList = train_data[1];
        feature_vectors = get_feature_vectors(train_data[0])

        for epoch in range(self.args.num_iter):
            for row in range(len(train_data[0])):
                tempArray = self.weights
                tempArray = np.transpose(tempArray)
                calculation = np.matmul(tempArray, feature_vectors[row]) + self.bias
                prediction = self.findSign(calculation)
                if prediction != predictionTestList[row]:
                    self.weights = self.weights + (self.args.lr * predictionTestList[row] * np.array(feature_vectors[row]))
                    self.bias = self.bias + (self.args.lr * predictionTestList[row])

    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        feature_vectors = get_feature_vectors(test_x)
        testPredictions = []
        for epoch in range(self.args.num_iter):
            for row in range(len(test_x)):
                tempArray = self.weights
                tempArray = np.transpose(tempArray)
                calculation = np.dot(tempArray, feature_vectors[row]) + self.bias # with bias
                calculationWithoutBias = np.dot(tempArray, feature_vectors[row]) # without bias
                prediction = self.findSign(calculation)
                testPredictions.append(prediction)
        return testPredictions

class AveragedPerceptron(BinaryClassifier):

    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args = args

    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        self.weights = []
        for i in range(self.args.f_dim):
            self.weights.append(0)
        self.bias = 0
        self.survival = 0
        self.weightsDerivative = []
        tr_size = len(train_data[0])
        indices = list(range(tr_size))
        random.seed(5) #this line is to ensure that you get the same shuffled order everytime
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        predictionTestList = train_data[1];
        feature_vectors = get_feature_vectors(train_data[0])

        for epoch in range(self.args.num_iter):
            for row in range(len(train_data[0])):
                tempArray = self.weights
                tempArray = np.transpose(tempArray)
                calculation = np.matmul(tempArray, feature_vectors[row]) + self.bias
                prediction = self.findSign(calculation)
                if prediction != predictionTestList[row]:
                    self.weightsDerivative = self.weights + (self.args.lr * predictionTestList[row] * np.array(feature_vectors[row]))
                    self.bias = self.bias + (self.args.lr * predictionTestList[row]) / (self.survival + 1)
                    self.weights = ((self.survival * np.array(self.weights)) + self.weightsDerivative) / (self.survival + 1)
                    self.survival = 1
                else:
                    self.survival = self.survival + 1

    def findSign(self, calculation):
        if calculation <= 0:
            return -1
        else:
            return 1

    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        feature_vectors = get_feature_vectors(test_x)
        testPredictions = []
        for epoch in range(self.args.num_iter):
            for row in range(len(test_x)):
                tempArray = self.weights
                tempArray = np.transpose(tempArray)
                calculation = np.dot(tempArray, feature_vectors[row]) + self.bias #with Bias
                calculationWithoutBias = np.dot(tempArray, feature_vectors[row])  #without Bias
                prediction = self.findSign(calculation)
                testPredictions.append(prediction)
        return testPredictions
