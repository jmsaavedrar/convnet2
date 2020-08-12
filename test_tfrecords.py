import sys
sys.path.append("/media/hdvision/jsaavedr/Research/git/tensorflow-2/")
import tensorflow as tf
import datasets.data as data
import utils.configuration as conf
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

"""
This script test the data stored in a tfrecord file. It needs a config file to read the data_dir in order to locate the 
mean and shape files. In addition, the file to be tested is required. 
"""
    
if __name__ == '__main__' :        
    parser = argparse.ArgumentParser(description = "Train a simple mnist model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)    
    parser.add_argument("-file", type=str, help=" name of section in the configuration file", required = True)
    pargs = parser.parse_args() 
    configuration_file = pargs.config
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)               
    #parser_tf_record
    #/home/vision/smb-datasets/MNIST-5000/ConvNet2.0/
    tfr_train_file = os.path.join(configuration.get_data_dir(), pargs.file)   
    
    mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
    shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
    #
    input_shape =  np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)
    
    number_of_classes = configuration.get_number_of_classes()
     
    tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
    tr_dataset = tr_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, 'test'));    
    tr_dataset = tr_dataset.shuffle(configuration.get_shuffle_size())        
    tr_dataset = tr_dataset.batch(batch_size = configuration.get_batch_size())    
    
    fig, xs = plt.subplots(2,5)
    for image,label in tr_dataset:            
        for i in range(10) :
            row = int (i / 5)
            col = i % 5
            im = 255 * (image[i] - np.min(image[i]))/ (np.max(image[i])-np.min(image[i]))
            xs[row,col].imshow(np.uint8(im), cmap = 'gray')        
        plt.pause(1)
            
        