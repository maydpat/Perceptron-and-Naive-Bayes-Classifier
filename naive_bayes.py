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
import math

class NaiveBayes(BinaryClassifier):

    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.args = args

    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        self.positiveProbVector = [0] * self.args.f_dim
        self.negativeProbVector = [0] * self.args.f_dim
        totalPositiveReviews = 0
        totalNegativeReviews = 0
        totalReviews = 0
        for i in range(len(train_data[1])):
            if train_data[1][i] == 1:
                totalPositiveReviews = totalPositiveReviews + 1
            else:
                totalNegativeReviews = totalNegativeReviews + 1
            totalReviews = totalReviews + 1
        feature_vectors = get_feature_vectors(train_data[0])
        for word in range(self.args.f_dim):
            positiveDocCount = 0
            negativeDocCount = 0
            for doc in range(len(feature_vectors)):
                if feature_vectors[doc][word] > 0 and train_data[1][doc] == 1:
                    positiveDocCount = positiveDocCount + 1
                elif feature_vectors[doc][word] > 0 and train_data[1][doc] == -1:
                    negativeDocCount = negativeDocCount + 1
            # Now we have the positive and negative doc counts. Compute conditional probs below here.
            positiveDocProbability = float(positiveDocCount + 1) / (totalPositiveReviews + self.args.f_dim)
            negativeDocProbability = float(negativeDocCount + 1) / (totalNegativeReviews + self.args.f_dim)
            self.positiveProbVector[word] = positiveDocProbability
            self.negativeProbVector[word] = negativeDocProbability

    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        feature_vectors = get_feature_vectors(test_x)
        testPredictions = []
        posProbLogSum = 0
        for row in range(len(feature_vectors)):
            posLogSum = 0
            negLogSum = 0
            for i in range(len(feature_vectors[row])):
                if feature_vectors[row][i] > 0:
                    posLogSum = posLogSum + math.log(self.positiveProbVector[i])
                    negLogSum = negLogSum + math.log(self.negativeProbVector[i])
            if posLogSum > negLogSum:
                testPredictions.append(1)
            else:
                testPredictions.append(-1)
        return testPredictions
