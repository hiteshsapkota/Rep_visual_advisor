#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:32:46 2018

@author: hiteshsapkota
"""
from imgfeat import *
from PIL import Image
import numpy as np
import json
import argparse
import os.path as osp
from utils import load_tsv
import numpy as np
img2vec=Img2Vec()
"""Feature extraction using imagenet. The features for each image is extracted using imgfeat program"""
def main():
    
    parser= argparse.ArgumentParser()
    parser.add_argument("file_type", type=str, help="Type of file: train2017, dev2017, test2017")
    parser.add_argument("anno_list", type=str, help="path to file containing annotation filepaths")
    args = parser.parse_args()
    params=vars(args)
    user_profiles = load_tsv('user_profiles.tsv')
    features=[]
    i=1
    with open(params['anno_list']) as anno_file:
         for line in anno_file:
             print("I am working on image", i)
             i+=1
             anno_path = osp.join ('', line.strip())
             with open(anno_path) as jf:
                 anno=json.load(jf)
                 image_path=anno['image_path']
                 img = Image.open(image_path).convert("RGB")
                 img_vec = img2vec.get_vec(img)
                 features.append(img_vec)
                 
    features=np.asarray(features)
    file_path='Features/'+line.split('/')[1]
    np.save(file_path+'.npy', features)

if __name__=='__main__':
    main()
                 
                 
                 
