#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:27:48 2018

@author: jsaavedr

@description: Formating the input file to have integer labels
"""
import argparse
import os

def is_integer(string):
    """Check if a string represents a integer"""    
    try:
        int(string)
        return True
    except ValueError:
        return False
#%%        return False
def validate_labels(labels):
    """Process labels checking if these are in the correct format [int]
       labels are not integer this will be converted to int and the mapping is saved
    """    
    new_labels=[]
    label_map = {}
    if not is_integer(labels[0]):                
        new_code = 0
        for label in labels:
            if not label in label_map:
                label_map[label] = new_code
                new_code = new_code + 1 
            new_labels.append(label_map[label])    
    else :
        new_labels = [int(label) for label in labels]
    label_set = set(new_labels)
    #checking the completness of the label set
    if (len(label_set) == max(label_set) + 1) and (min(label_set) == 0):
        return new_labels, label_map
    else:            
        raise ValueError("Some codes are missed in label set! {}".format(label_set))
#%%  
def load_mapping(mapping_file):
    with open(mapping_file) as file:
        lines = [line.rstrip() for line in file]     
        lines = [tuple(line.rstrip().split('\t'))  for line in lines] 
        str_labels, int_label = zip(*lines)
        label_map = {}
        for i, label in enumerate(str_labels) :
            label_map[label] = int_label[i]
    return label_map
        
def save_mapping(mapping, mapping_file):
    """ save mapping for labels """
    print("saving mapping to {}".format(mapping_file))            
    mapping_file = open(mapping_file, "w+")
    for key, val in mapping.items() :
        mapping_file.write("{}\t{}\n".format(key,val))
    mapping_file.close()
        
#%%
def read_data_from_file(filename, mapping_file = None):    
    """read data from text files and convert
    string labels to integer labels
    """     
    print(filename)
    # reading data from files, line by line
    with open(filename) as file :        
        lines = [line.rstrip() for line in file]             
        lines_ = [tuple(line.rstrip().split('\t'))  for line in lines ] 
        filenames, labels = zip(*lines_)
        if mapping_file is None :
            labels, mapping = validate_labels(labels)
            print(mapping)
            if bool(mapping) :            
                mapping_file = os.path.join(os.path.dirname(filename), "mapping.txt")
                print('save mapping at {}'.format(mapping_file))            
                save_mapping(mapping, mapping_file)
                
        else :
            label_map = load_mapping(mapping_file)        
            labels = [label_map[l] for l in labels]
    return filenames, labels
#%%
if __name__ == '__main__':
    """
    the output is a jfile
    """
    parser = argparse.ArgumentParser(description = "Convert string labels to integer labels")
    parser.add_argument("-path", type = str, help = "<string> input file" , required = True)
    parser.add_argument("-mapping", type = str, help = "<string> mapping file" , required = False)
    pargs = parser.parse_args()    
    #file list should be in -path-
    filename = os.path.join(pargs.path, "list.txt")
    assert os.path.exists(filename), "file {} doesn't exist".format(filename)
    print("Starting")
    imagenames, labels = read_data_from_file(filename, pargs.mapping)
    out_file = os.path.join(os.path.dirname(filename), "list.jfile")
    ff = open(out_file, "w+")
    for image, label in zip(imagenames, labels) :
        #print("{} {}".format(image, label))
        ff.write("{}\t{}\n".format(image, label))
    ff.close()