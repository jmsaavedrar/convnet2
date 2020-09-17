"""
jsaavedr

train_siamese allows you to train a siamese network for sbir, using triplet loss
It uses the same parameters as those in train.py. Furthermore, instead of using a dataset object
it uses a generator for training and testing data
 
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys
sys.path.append("/home/jsaavedr/Research/git/tensorflow-2/convnet2")
import tensorflow as tf
from models import resnet
import datasets.datagenerator as datagenerator
import utils.configuration as conf
import utils.losses as losses
import utils.metrics as metrics
import numpy as np
import argparse

if __name__ == '__main__' :        
    parser = argparse.ArgumentParser(description = "Train a simple mnist model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)    
    parser.add_argument("-mode", type=str, choices=['train', 'test', 'variables'],  help=" train or test", required = False, default = 'train')
    parser.add_argument("-save", type= bool,  help=" True to save the model", required = False, default = False)
    pargs = parser.parse_args()  
    configuration_file = pargs.config
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)
    number_of_classes = configuration.get_number_of_classes()               
    #create datagenerator for training
    #(self, data_path, batch_size,  num_classes, shuffle = True ):
    shape_file = os.path.join(configuration.get_data_dir(),'sketches', 'shape.dat')    
    input_shape =  np.fromfile(shape_file, dtype=np.int32)
    if pargs.mode == 'train' :
        tra_dataset = datagenerator.SiameseDataGenerator(configuration.get_data_dir(), configuration.get_batch_size(),  number_of_classes, datasettype = 'train')
    if pargs.mode == 'test' or pargs.mode == 'train' : 
        val_dataset = datagenerator.SiameseDataGenerator(configuration.get_data_dir(), configuration.get_batch_size(),  number_of_classes, datasettype = 'test')
    #The following code is used for running in a multiple gpus, we tested with 2 gpus.     
    tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #Defining callback for saving checkpoints
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=configuration.get_snapshot_dir() + '{epoch:03d}.h5',
            save_weights_only=True,
            mode='max',
            save_best_only=False,
            save_freq = 'epoch',
            )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=configuration.get_snapshot_dir(), histogram_freq=1)
            
        #SiameseNet using a ResNet 34 as backbone
        model = resnet.SiameseNet([3,4,6,3],[64,128,256,512], configuration.get_number_of_classes(), se_factor = 0)
        #the model is trained in a triplet mode    
        input_sketch = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_sketch')
        input_positive = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image') 
        input_negative = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image')    
        model([input_sketch, input_positive, input_negative])    
        model.summary()        
        
        if configuration.use_checkpoint() :
            model.load_weights(configuration.get_checkpoint_file(), by_name = True, skip_mismatch = True)
            print('weights loaded  from {}'.format(configuration.get_checkpoint_file()))
            sys.stdout.flush()
            
        if configuration.use_checkpoint_for_photo() :
            model.load_weights(configuration.get_checkpoint_file_photo(), by_name = True, skip_mismatch = True)
            print('ph_backbone initialized from {}'.format(configuration.get_checkpoint_file_photo()))
            sys.stdout.flush()
            
        if configuration.use_checkpoint_for_sketch() :
            model.load_weights(configuration.get_checkpoint_file_sketch(), by_name = True, skip_mismatch = True)
            print('sk_backbone initialized from {}'.format(configuration.get_checkpoint_file_sketch()))
            sys.stdout.flush()    
                    
        #define the training parameters
        #Here, you can test SGD vs Adam
        initial_learning_rate= configuration.get_learning_rate()
        lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate = initial_learning_rate,
                                                            decay_steps = configuration.get_decay_steps(),
                                                            alpha = 0.0001)
            
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = lr_schedule), #(learning_rate = configuration.get_learning_rate()), # 'adam'                           
                      loss =  [losses.triplet_loss(0.5), losses.crossentropy_triplet_loss],                 
                      metrics = [[metrics.d_positive, metrics.d_negative],[metrics.metric_accuracy_siamese]],
                      loss_weights = [0.4, 0.6])
                    #metrics=['accuracy'])
        if pargs.mode == 'train' :                             
            history = model.fit(tra_dataset, 
                                epochs = configuration.get_number_of_epochs(),                        
                                validation_data=val_dataset,
                                validation_steps = configuration.get_validation_steps(),
                                callbacks=[model_checkpoint_callback])
        elif pargs.mode == 'test' :
            model.evaluate(val_dataset,
                           steps = configuration.get_validation_steps(),
                           callbacks=[tensorboard_callback])
        elif pargs.mode == 'variables' :        
            for variable in model.layers :
                print(variable.name)
                
        if pargs.save :
            saved_to = os.path.join(configuration.get_data_dir(),"saved-model")
            tf.saved_model.save(model, os.path.join(configuration.get_data_dir(),"saved-model"))
            print("model saved to {}".format(saved_to))            
        
