"""
@author: jsaavedr
Description: Create tfrecords
This code allows  you to generate tfrecords files for test.txt and train.txt
Saving data as tfrecords is an efficient manner to store the data, which allows the model to read the data quickly, especially during training.
"""
import pathlib
import sys
sys.path.append(str(pathlib.Path().absolute()))
import argparse
import utils.configuration as conf
import utils.imgproc as imgproc
import datasets.data as data

if __name__ == '__main__':                    
    parser = argparse.ArgumentParser(description = "Create a dataset for training an testing")
    """ pathname must include train.txt and test.txt files """  
    parser.add_argument("-type", type = str, help = "train| test | all",  choices = ['train', 'test', 'all'], required = True )
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)                
    pargs = parser.parse_args() 
    configuration_file = pargs.config
    #assert os.path.exists(configuration_file), "configuration file does not exist {}".format(configuration_file)   
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)
                   
    process_fun = imgproc.process_image
    if configuration.get_image_type() == 'SKETCH'  : 
        process_fun = imgproc.process_sketch        
    elif configuration.get_image_type() == 'MNIST'  : 
        process_fun = imgproc.process_mnist
    elif configuration.get_image_type() == 'CUSTOM'  : 
        process_fun = imgproc.create_processing_function(configuration.get_image_processing_params())    
    data.create_tfrecords(configuration, 
                          pargs.type, 
                          processFun = process_fun)
    print("tfrecords created for " + configuration.get_data_dir())