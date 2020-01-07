# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:13:42 2019

@author: Tran Nguyen Tien Dat
"""

def feature_extractor_movieReview(moviReview_sent):
     sent_set = set(moviReview_sent)
     features = {}  #dictionary for training
     for word in words_mostApprear:
         features['contain(%s)'%word]= word in sent_set
     return features