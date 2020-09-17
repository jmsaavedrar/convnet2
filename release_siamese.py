"""
release_siamese.py is used for splitting a siamese network into its two backbones, sk_backbone and ph_backbone
to be used independently for feature extraction
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys
sys.path.append("/home/jsaavedr/Research/git/tensorflow-2/convnet2")
import tensorflow as tf
from models import resnet
import utils.configuration as conf
import numpy as np
import argparse

if __name__ == '__main__' :        
    parser = argparse.ArgumentParser(description = "Train a simple mnist model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)    
    parser.add_argument("-image_type", type=str, help=" sketch or photo", choices =['sketch', 'image'], required = True)
    pargs = parser.parse_args()  
    configuration_file = pargs.config
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)
    number_of_classes = configuration.get_number_of_classes()               
    
    shape_file = os.path.join(configuration.get_data_dir(),'sketches', 'shape.dat')    
    input_shape =  np.fromfile(shape_file, dtype=np.int32)
                
                   
    if pargs.image_type == 'sketch' : 
        model = resnet.SiameseNetSketch([3,4,6,3],[64,128,256,512], False, se_factor = 0)
    else:        
        model = resnet.SiameseNetImage([3,4,6,3],[64,128,256,512], False, se_factor = 0)        
    input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_' + pargs.image_type)     
    model(input_image)    
    model.summary()
    assert configuration.use_checkpoint() 
    model.load_weights(configuration.get_checkpoint_file(), by_name = True, skip_mismatch = True)                        
    #save the model        
    saved_to = os.path.join(configuration.get_data_dir(), "saved-model-"+pargs.image_type)
    tf.saved_model.save(model, saved_to)
    print("model saved to {}".format(saved_to))
    
