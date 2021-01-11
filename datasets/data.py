"""
@author: jsaavedra

Module for data processing, this includes tfrecords generation

"""
import os
import sys
import numpy as np
import random
import utils.imgproc as imgproc
import tensorflow as tf
import skimage.io as io
import skimage.color as color
import threading
from datetime import datetime



#%% int64 should be used for integer numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#%% byte should be used for string  | char data
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#%% float should be used for floating point data
def _float_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
#%% float list
def _float_list_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
#%% int64 list
def _int64_list_feature(value):    
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def read_image(filename, number_of_channels):
    """ read_image using skimage
        The output is a 3-dim image [H, W, C]
    """    
    if number_of_channels  == 1 :            
        image = io.imread(filename, as_gray = True)
        image = imgproc.toUINT8(image)
        assert(len(image.shape) == 2)
        image = np.expand_dims(image, axis = 2) #H,W,C                    
        assert(len(image.shape) == 3 and image.shape[2] == 1)
    elif number_of_channels == 3 :
        image = io.imread(filename)
        if(len(image.shape) == 2) :
            image = color.gray2rgb(image)
        elif image.shape[2] == 4 :
            image = color.rgba2rgb(image) 
        image = imgproc.toUINT8(image)        
        assert(len(image.shape) == 3 and image.shape[2]==3)
    else:
        raise ValueError("number_of_channels must be 1 or 3")
    if not os.path.exists(filename):
        raise ValueError(filename + " does not exist!")
    return image

def validate_labels(labels) :
    """It checks if labels are in the correct format [int]
       labels need to be integers, from 0 to NCLASSES -1 
    """  
    new_labels = [int(label) for label in labels]
    label_set = set(new_labels)
    #checking the completness of the label set
    if (len(label_set) == max(label_set) + 1) and (min(label_set) == 0):
        return new_labels
    else:            
        raise ValueError("Some codes are missed in label set! {}".format(label_set))
    
def read_data_from_file(str_path, dataset = "train" , shuf = True):    
    """read data from text files
    and apply shuffle by default 
    """            
    datafile = os.path.join(str_path, dataset + ".txt")    
    assert os.path.exists(datafile)        
    # reading data from files, line by line
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
        if shuf:
            random.shuffle(lines)
        _lines = [tuple(line.rstrip().split('\t'))  for line in lines ] 
        filenames, labels = zip(*_lines)
        labels = validate_labels(labels)    
    return filenames, labels

def create_tfrecords_from_file(filenames, labels, image_shape, tfr_filename, process_function = imgproc.resize_image):    
    #create tf-records
    writer = tf.io.TFRecordWriter(tfr_filename)
    #filenames and lables should  have the same size    
    assert len(filenames) == len(labels)
    mean_image = np.zeros(image_shape, dtype=np.float32)
    n_reading_error = 0    
    for i in range(len(filenames)):   
        try :      
            if i % 500 == 0 or (i + 1) == len(filenames):
                print("---{}".format(i))           
            image = read_image(filenames[i], image_shape[2]) #scikit-image
            image = process_function(image, (image_shape[0], image_shape[1]))
            #print(image)
            #cv2.imshow("image", image)
            #print(" {} {} ".format(image.shape, labels[i]))        
            #cv2.waitKey()        
            #create a feature                
            feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                       'label': _int64_feature(labels[i])}
            
            #create an example protocol buffer
            example = tf.train.Example(features = tf.train.Features(feature=feature))        
            #serialize to string and write on the file
            writer.write(example.SerializeToString())
            mean_image = mean_image + image / len(filenames)
        except ValueError :
            n_reading_error = n_reading_error + 1 
            print("Error reading {}:{}".format(n_reading_error, filenames[i]))
                            
    writer.close()
    sys.stdout.flush()
    return mean_image

def process_batch_threads(thr_index, ranges, filenames, labels, image_shape, tfr_filename, process_function = imgproc.resize_image):    
    #create tf-records    
    tfr_filename_batch = '{}_{}.tfrecords'.format(tfr_filename, thr_index)
    mean_filename_batch = '{}_{}_mean.npy'.format(tfr_filename, thr_index)    
    writer = tf.io.TFRecordWriter(tfr_filename_batch)
    #filenames and lables should  have the same size    
    assert len(filenames) == len(labels)
    mean_batch = np.zeros(image_shape, dtype=np.float32)
    n_reading_error = 0    
    batch_size = ranges[thr_index][1] - ranges[thr_index][0]
    count = 0
    for idx in np.arange(ranges[thr_index][0], ranges[thr_index][1]) :   
        try :                          
            image = read_image(filenames[idx], image_shape[2]) #scikit-image
            image = process_function(image, (image_shape[0], image_shape[1]))                            
            feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                       'label': _int64_feature(labels[idx])}            
            #create an example protocol buffer
            example = tf.train.Example(features = tf.train.Features(feature=feature))        
            #serialize to string and write on the file
            writer.write(example.SerializeToString())
            mean_batch = mean_batch + image / batch_size;
            count = count + 1
            if count % 100 == 0:
                print('{} Thread {} --> processing {} of {} [{}, {}]'.format(datetime.now(), thr_index, count, batch_size, ranges[thr_index][0], ranges[thr_index][1]))
        except ValueError :
            n_reading_error = n_reading_error + 1 
            print('Error reading {}:{}'.format(n_reading_error, filenames[idx]))
    
    writer.close()
    mean_batch.astype(np.float32).tofile(mean_filename_batch)
    print('Thread {} --> saving mean at {}'.format(mean_filename_batch, thr_index))
    sys.stdout.flush()
    
def create_tfrecords_threads(filenames, labels, image_shape, tfr_filename, process_function, n_threads):
    assert len(filenames) == len(labels) 
    #break whole dataset int batches according to the number of threads
    spacing = np.linspace(0, len(filenames), n_threads + 1).astype(np.int)
    ranges = []    
    for i in np.arange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    
    #launch a thread for each batch
    print('Launching {} threads for spacings: {}'.format(n_threads, ranges))
    sys.stdout.flush()
    threads = []
    for thr_index in np.arange(len(ranges)):
        args = (thr_index, ranges, filenames, labels, image_shape, tfr_filename, process_function)
        t = threading.Thread(target = process_batch_threads, args=args)
        t.start()
        threads.append(t)         
    #wait until all threads end        
    for idx, thread in enumerate(threads):        
        thread.join()
    print('*******All threads have finished!!*******')    
    #compute the mean image from those computed by each thread    
    for idx in range(n_threads) :
        mean_filename = '{}_{}_mean.npy'.format(tfr_filename, idx)
        if idx == 0 :        
            mean_image = np.reshape(np.fromfile(mean_filename,  dtype=np.float32), image_shape)
        else :
            mean_image = mean_image + np.reshape(np.fromfile(mean_filename,  dtype=np.float32), image_shape)
    mean_image = mean_image / n_threads
    return mean_image           
        

def create_tfrecords(config, _type, processFun = imgproc.resize_image) :
    """ 
    data_dir: Folder where data is located (train.txt and test.txt should be found)
    type: 'train' | 'test' | 'all' (default='all')             
    im_shape: [H,W,C] of the input          
    processFun: processing function which depends on the problem we are dealing with
    """
    data_dir = config.get_data_dir()    
    image_shape = np.asarray(config.get_image_shape())
    n_threads = config.get_num_threads()    
    #------------- creating train data
    if (_type == 'train') or (_type == 'all') : 
        filenames, labels = read_data_from_file(data_dir, dataset = 'train', shuf = True)
        if config.use_multithreads() :
            tfr_filename = os.path.join(data_dir, 'train')
            training_mean = create_tfrecords_threads(filenames, labels, image_shape, tfr_filename, processFun, n_threads)
        else :        
            tfr_filename = os.path.join(data_dir, 'train.tfrecords')            
            training_mean = create_tfrecords_from_file(filenames, labels, image_shape, tfr_filename, processFun)
            
        print('train_record saved at {}.'.format(tfr_filename))
        #saving training mean
        mean_file = os.path.join(data_dir, "mean.dat")
        print("mean_file {}".format(training_mean.shape))
        training_mean.astype(np.float32).tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))
        #saving shape file    
        shape_file = os.path.join(data_dir, "shape.dat")
        image_shape.astype(np.int32).tofile(shape_file)
        print("shape_file saved at {}.".format(shape_file))  
    #-------------- creating test data    
    if (_type == 'test') or (_type == 'all') :
        filenames, labels = read_data_from_file(data_dir, dataset="test", shuf = True)
        if config.use_multithreads() :
            tfr_filename = os.path.join(data_dir, 'test')
            create_tfrecords_threads(filenames, labels, image_shape, tfr_filename, processFun, n_threads)
        else :    
            tfr_filename = os.path.join(data_dir, "test.tfrecords")
            create_tfrecords_from_file(filenames, labels, image_shape, tfr_filename, processFun)
        print("test_record saved at {}.".format(tfr_filename))    
                
    
#parser tf_record to be used for dataset mapping
def parser_tfrecord(serialized_input, input_shape, mean_image, number_of_classes, with_augmentation = False):
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'image': tf.io.FixedLenFeature([], tf.string),
                                        'label': tf.io.FixedLenFeature([], tf.int64)
                                        })
        #image
        #rgb_mean = [123.68, 116.779, 103.939]
        #rgb_std = [58.393, 57.12, 57.375]         
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, input_shape)
        #data augmentation
        #central crop
        #dataagumentation prob = 0.4
        if with_augmentation:
            data_augmentation_prob = 0.5
            prob = tf.random.uniform((), 0 ,1)
            if prob < data_augmentation_prob :
                #image = tf.image.flip_left_right(image)                
                #fraction = tf.random.uniform((), 0.5, 0.9, dtype = tf.float32)
                if prob < data_augmentation_prob * 0.5 :
                    image = tf.image.central_crop(image, central_fraction = 0.7)
                    image = tf.cast(tf.image.resize(image, (input_shape[0], input_shape[1])), tf.uint8)
                else :
                    image = tf.image.flip_left_right(image)                
                
            #TODO
        
        image = tf.cast(image, tf.float32)
        #image = (image - rgb_mean) / rgb_std
        image = image - mean_image
        
        #label
        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, depth = number_of_classes)
        label = tf.reshape(label, [number_of_classes])
        
        return image, label          
    
    
    #parser tf_record to be used for dataset mapping
def parser_tfrecord_siamese(serialized_input, input_shape, mean_image,  with_augmentation = False):
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'image': tf.io.FixedLenFeature([], tf.string),
                                        'label': tf.io.FixedLenFeature([], tf.int64)
                                        })
        #image
        #rgb_mean = [123.68, 116.779, 103.939]
        #rgb_std = [58.393, 57.12, 57.375]         
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, input_shape)
        #data augmentation
        #central crop
        #dataagumentation prob = 0.4
        if with_augmentation:
            data_augmentation_prob = 0.5
            prob = tf.random.uniform((), 0 ,1)
            if prob < data_augmentation_prob :
                #image = tf.image.flip_left_right(image)                
    #             #fraction = tf.random.uniform((), 0.5, 0.9, dtype = tf.float32)
                if prob < data_augmentation_prob * 0.5 :
                    image = tf.image.central_crop(image, central_fraction = 0.7)
                    image = tf.cast(tf.image.resize(image, (input_shape[0], input_shape[1])), tf.uint8)
                else :
                    image = tf.image.flip_left_right(image)                
                
            #TODO
        
        image = tf.cast(image, tf.float32)
        #image = (image - rgb_mean) / rgb_std
        image = image - mean_image
        
        #label
        label = tf.cast(features['label'], tf.int32)                
        return image, label
    