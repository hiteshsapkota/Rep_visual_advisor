#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:59:57 2018

@author: hiteshsapkota
"""

import json
import numpy as np
import matplotlib.pyplot as plt
with open("per_attr_accuracy.json") as outfile:
    per_attr_accr=json.load(outfile)
    
attributes =[k for k, v in per_attr_accr.items()]
precision=[]
recall=[]
fscore=[]
for attribute in attributes:
    precision.append(per_attr_accr[attribute]['precision'])
    recall.append(per_attr_accr[attribute]['recall'])
    fscore.append(per_attr_accr[attribute]['fscore'])
    

print("Average Precision:", np.mean(precision))
print("Average Recall:", np.mean(recall))
print("Average F score:", np.mean(fscore))
    
y_pos = np.arange(len(attributes))    
plt.figure(figsize=(16,6))
plt.bar(y_pos, fscore, align='center', alpha=0.5)
plt.xticks(y_pos, attributes, rotation=90)
plt.ylabel('Fscore')
plt.title('Fscore corresponding to each attribute type')