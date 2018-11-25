#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:03:34 2018

@author: hiteshsapkota
"""

import os.path as osp
import argparse
import json
from utils import load_attributes
"""
    This produces the imagewise attribute presence/absense status. For example if there are
    four images with attributes (att1, att2), (att2, att3), (att1, att4), (att2, att4)
    then binary output output for first attribute att1 is = [1, 0, 1, 0]. Similarly for other attributes. Here binary output is stored to the directory binary_output
    """

def main():
    """Use command python "train2017" "train2017.txt" to store the binary output for training images"""
    parser= argparse.ArgumentParser()
    parser.add_argument("file_type", type=str, help="Type of file: train2017, dev2017, test2017")
    parser.add_argument("anno_list", type=str, help="path to file containing annotation filepaths")
    args = parser.parse_args()
    params=vars(args)
    [attr_id_to_name, _]= load_attributes()
    attributes=[k for k,v in attr_id_to_name.items()]
    for attribute in attributes:
        bin_actual_output=[]
        with open(params['anno_list']) as anno_file:
            for line in anno_file:
                anno_path = osp.join ('', line.strip())
                with open(anno_path) as jf:
                    anno=json.load(jf)
                    image_labels=anno['labels']
                    if attribute in image_labels:
                        bin_actual_output.append(1)
                    else:
                        bin_actual_output.append(0)
        file_name=attribute
        file_path='binary_output/'+line.split('/')[1]
        with open(file_path+'/'+file_name+".json", 'w') as inf:
            json.dump(bin_actual_output, inf)
                        
   
if __name__ == "__main__":
    main()    
