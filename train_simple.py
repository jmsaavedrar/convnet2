"""
jsaavedr, 2020

This is a simple version of train.py. 

To use train.py, you will require to set the following parameters :
 * -config : A configuration file where a set of parameters for data construction and training is defined.
 * -name: The section name in the configuration file.
 * -mode: [train, test] for training, testing, or showing  variables of the current model. By default this is set to 'train'
 * -save: Set true for saving the model
"""
import pathlib
import sys
sys.path.append(str(pathlib.Path().absolute()))
import tensorflow as tf
from models import simple
from models import alexnet
import datasets.data as data
import utils.configuration as conf
import utils.losses as losses
import utils.imgproc as imgproc
import utils.metrics as metrics
import numpy as np
import argparse
import os

if __name__ == '__main__' :        
    parser = argparse.ArgumentParser(description = "Train a simple mnist model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)
    parser.add_argument("-mode", type=str, choices=['train', 'test', 'predict'],  help=" train or test", required = False, default = 'train')
    parser.add_argument("-save", type=lambda x: (str(x).lower() == 'true'),  help=" True to save the model", required = False, default = False)    
    pargs = parser.parse_args()     
    configuration_file = pargs.config
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)                   
    if pargs.mode == 'train' :
        tfr_train_file = os.path.join(configuration.get_data_dir(), "train.tfrecords")
    if pargs.mode == 'train' or  pargs.mode == 'test':    
        tfr_test_file = os.path.join(configuration.get_data_dir(), "test.tfrecords")
    if configuration.use_multithreads() :
        if pargs.mode == 'train' :
            tfr_train_file=[os.path.join(configuration.get_data_dir(), "train_{}.tfrecords".format(idx)) for idx in range(configuration.get_num_threads())]
        if pargs.mode == 'train' or  pargs.mode == 'test':    
            tfr_test_file=[os.path.join(configuration.get_data_dir(), "test_{}.tfrecords".format(idx)) for idx in range(configuration.get_num_threads())]        
    sys.stdout.flush()
        
    mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
    shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
    #
    input_shape =  np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)        
    number_of_classes = configuration.get_number_of_classes()
    #loading tfrecords into a dataset object
    if pargs.mode == 'train' : 
        tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
        tr_dataset = tr_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation = False));    
        tr_dataset = tr_dataset.shuffle(configuration.get_shuffle_size())        
        tr_dataset = tr_dataset.batch(batch_size = configuration.get_batch_size())            

    if pargs.mode == 'train' or  pargs.mode == 'test':
        val_dataset = tf.data.TFRecordDataset(tfr_test_file)
        val_dataset = val_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation = False));    
        val_dataset = val_dataset.batch(batch_size = configuration.get_batch_size())
                        
       
    #Defining a callback for saving checkpoints
    #save_freq: frequency in terms of number steps each time checkpoint is saved
    #here, we define save_freq equal to an epoch 
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(configuration.get_snapshot_dir(), '{epoch:03d}.h5'),
        save_weights_only=True,
        mode = 'max',
        monitor='val_acc',
        save_freq = 'epoch',            
        )
    #save_freq = configuration.get_snapshot_steps())                
    if configuration.get_model_name() == 'SKETCH' :
        model = alexnet.AlexNetModel(configuration.get_number_of_classes())            
        process_fun = imgproc.process_sketch
    else:
        model = simple.SimpleModel(configuration.get_number_of_classes())
        process_fun = imgproc.process_mnist    
            
    #resnet_50
    #model = resnet.ResNet([3,4,6,3],[64,128,256,512], configuration.get_number_of_classes(), use_bottleneck = True)
    #build the model indicating the input shape    
    input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image')     
    model(input_image)    
    model.summary()
    #use_checkpoints to load weights
    if configuration.use_checkpoint() :                
        model.load_weights(configuration.get_checkpoint_file(), by_name = True, skip_mismatch = True)        
    #opt = tf.keras.optimizers.Adam() #learning_rate = configuration.get_learning_rate())
    initial_learning_rate = configuration.get_learning_rate()    
    opt = tf.keras.optimizers.SGD(learning_rate = initial_learning_rate, momentum = 0.9, nesterov = True)
    model.compile(optimizer=opt,
                  loss= losses.crossentropy_loss,
                  metrics=[metrics.simple_accuracy])
 
    if pargs.mode == 'train' :                             
        history = model.fit(tr_dataset, 
                        epochs = configuration.get_number_of_epochs(),                        
                        validation_data=val_dataset,
                        validation_steps = configuration.get_validation_steps(),
                        callbacks=[model_checkpoint_callback])
                
    elif pargs.mode == 'test' :
        model.evaluate(val_dataset,
                       steps = configuration.get_validation_steps())
        
    elif pargs.mode == 'predict':
        filename = input('file :')
        while(filename != 'end') :
            target_size = (configuration.get_image_height(), configuration.get_image_width())
            image = process_fun(data.read_image(filename, configuration.get_number_of_channels()), target_size )
            image = image - mean_image
            image = tf.expand_dims(image, 0)        
            pred = model.predict(image)
            pred = pred[0]
            #softmax to estimate probs
            pred = np.exp(pred - max(pred))
            pred = pred / np.sum(pred)            
            cla = np.argmax(pred)
            print('{} [{}]'.format(cla, pred[cla]))
            filename = input('file :')
    #save the model   
    if pargs.save :
        saved_to = os.path.join(configuration.get_data_dir(),"cnn-model")
        model.save(saved_to)
        print("model saved to {}".format(saved_to))