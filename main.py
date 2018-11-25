#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:51:29 2018

@author: hiteshsapkota
"""
import json
from utils import load_attributes
from svc import predict
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
def load_json(file):
    with open(file, 'r') as outfile:
        json_file = json.load(outfile)
    return json_file
"""Takes features, trains the SVC by calling predict function of svc and makes prediction for test
    images based on test features"""
def main():
    """Loads training, development, and test features"""
    train_feat = np.load("Features/train2017.npy")
    val_feat = np.load("Features/val2017.npy")
    test_feat = np.load("Features/test2017.npy")
    train_X = np.concatenate((train_feat, val_feat), axis=0)
    test_X = test_feat
    """Loads attributes"""
    [attr_id_to_name, _]= load_attributes()
    attributes=[k for k,v in attr_id_to_name.items()]
    accuracy={}
    for attribute in attributes:
        """Loads binary output for each imges. 1 if attribute is present in image
            and 0 otherwise. So dimension is number of images*1(presence or absense
            of attribute in that image"""
        train_file_path = 'binary_output/train2017/'+attribute+'.json'
        val_file_path = 'binary_output/val2017/'+attribute+'.json'
        test_file_path = 'binary_output/test2017/'+attribute+'.json'
        train_actual_output =  load_json(train_file_path)
        val_actual_output = load_json(val_file_path)
        test_actual_output = load_json(test_file_path)
        train_y = train_actual_output + val_actual_output
        test_pred_output = predict(train_X, train_y, test_X).tolist()
        
        with open ("prediction/"+attribute+".json", "w") as infile:
            json.dump(test_pred_output, infile)
        precision = precision_score(test_actual_output, test_pred_output)
        recall = recall_score(test_actual_output, test_pred_output)
        fscore = f1_score(test_actual_output, test_pred_output)
        accuracy[attribute]={}
        accuracy[attribute]['precision']= precision
        accuracy[attribute]['recall']= recall
        accuracy[attribute]['fscore']= fscore
        
        print("Precision for attribute", attribute, "is:", precision)
        print("Recall for attribute", attribute, "is:", recall)
        print("F score for attribute", attribute, "is:", fscore)
    with open("per_attr_accuracy.json", "w") as outfile:
        json.dump(accuracy, outfile)
        
if __name__=='__main__':
    main() 
        
        
        
        
        
            
    
        
        
  
