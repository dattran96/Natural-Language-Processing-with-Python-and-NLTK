# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 12:43:44 2019

@author: Tran Nguyen Tien Dat, TU Darmstadt
"""

import nltk
from nltk.corpus import movie_reviews
import numpy as np

def data_training_generation(Labeled_Data):
    featureSet= [ (feature_extractor_movieReview(sent), tag) for (sent,tag) in Labeled_Data ]
    return featureSet

def feature_extractor_movieReview(moviReview_sent):
     sent_set = set(moviReview_sent)
     features = {}  #dictionary for training
     for word in words_mostApprear:
         features['contain(%s)'%word]= word in sent_set
     return features
 
# prepare review data as a list of tuples:
# (list of tokens, category)
# category is positive / negative
#get Raw Data 
Labeled_Review_Data_sents = [(movie_reviews.words(fileid), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]
threshhold = 2000 #threshhold for experience
train_data= Labeled_Review_Data_sents[:int(0.8*len(Labeled_Review_Data_sents))]
devTest_data=  Labeled_Review_Data_sents[int(0.8*len(Labeled_Review_Data_sents)):int(0.9*len(Labeled_Review_Data_sents))]
test_data= Labeled_Review_Data_sents[int(0.9*len(Labeled_Review_Data_sents)):]

#Criteria for chosing features
words_FreDis = nltk.FreqDist(word.lower() for sent,category in train_data for word in sent )
words_mostApprear = words_FreDis.most_common(threshhold)
words_mostApprear= [word for word,_ in words_mostApprear]

# get Data for training and testing, by pushing raw data through feature_extractor function     
trainSet = data_training_generation(train_data)
devSet=data_training_generation(devTest_data)
testSet=data_training_generation(test_data)

#training
classifier_NaivBayes = nltk.NaiveBayesClassifier.train(trainSet)
print("Accuracy:")
print(nltk.classify.accuracy(classifier_NaivBayes, testSet))
#classifier_NaivBayes.classify(feature_extractor_movieReview(test2))
#np.random.seed(1)
#np.random.random_sample()
