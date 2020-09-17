import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("/home/jsaavedr/Research/git/tensorflow-2/convnet2")
import matplotlib.pyplot as plt
import datasets.data as data

def parser_tfrecord(serialized_input, input_shape):
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'image': tf.io.FixedLenFeature([], tf.string),
                                        'label': tf.io.FixedLenFeature([], tf.int64)
                                        })
                 
        image = tf.io.decode_raw(features['image'], tf.uint8)        
        image = tf.reshape(image, input_shape)                                
        label = tf.cast(features['label'], tf.int32)        
        return image, label       
        
class SiameseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, batch_size, num_classes, datasettype, shuffle = True ):
        """
        @data_path: path where the data is stored sk_train, im_train
        @batch_size: size for each bach
        @num_classes: number of the classes to be trained
        @shuffle: True if shuffle is applied at starting each epoch
        """
        self.batch_size = batch_size        
        self.num_classes = num_classes 
        self.shuffle = shuffle
        self.group_idx = np.arange(num_classes)        
        #sk_images and ph_images keeps fix during all the process
        #loading data
        #Let's go with sketch data        
        #mean_file = os.path.join(os.path.join(data_path, 'sketches', 'mean.dat'))
        shape_file = os.path.join(os.path.join(data_path, 'sketches', 'shape.dat'))        
        self.sk_shape =  np.fromfile(shape_file, dtype=np.int32)        
        #self.sk_mean_image = np.fromfile(mean_file, dtype=np.float32)
        #self.sk_mean_image = np.reshape(self.sk_mean_image, self.sk_shape)
        rgb_mean = np.array([123.68 / 58.393, 116.779 / 57.12, 103.939 / 57.375], dtype = np.float32)
        self.sk_mean_image = rgb_mean;
        #rgb_std = [58.393, 57.12, 57.375]
        self.sk_images, self.sk_labels = self.__load_data(os.path.join(data_path, 'sketches', datasettype + '.tfrecords'), self.sk_shape, self.sk_mean_image)        
        self.data_size = len(self.sk_images)
        #Let's go with image data        
        #shape example (224,224,3)
        #mean_file = os.path.join(os.path.join(data_path, 'photos', 'mean.dat'))
        shape_file = os.path.join(os.path.join(data_path, 'photos', 'shape.dat'))
        self.ph_shape =  np.fromfile(shape_file, dtype=np.int32)                
        #self.ph_mean_image = np.fromfile(mean_file, dtype=np.float32)
        #self.ph_mean_image = np.reshape(self.ph_mean_image, self.ph_shape)
        self.ph_mean_image = rgb_mean
        self.ph_images, self.ph_labels = self.__load_data(os.path.join(data_path, 'photos', datasettype + '.tfrecords'), self.ph_shape, self.ph_mean_image)
        
        self.shape = self.sk_shape  
        #idx_sk : order of the sk_images
        #idx_pos: order of the positive pairs from ph_images
        #idx_neg: order of the negative pairs from ph_images
        #idx_sk, idx_pos and idx_neg should work as parallel arrays 
        self.sk_idx  = []
        self.idx_pos = []
        self.idx_neg = []
        self.sk_groups = {}
        #grouping sketches ids by the group label
        for idx in self.group_idx :
            self.sk_groups[idx] = np.where(self.sk_labels == idx)[0]
        #grouping images ids by the group label    
        self.ph_groups = {}
        for idx in self.group_idx :
            self.ph_groups[idx] = np.where(self.ph_labels == idx)[0]            
        self.on_epoch_end()
    
    def __load_data(self, filename, shape, mean_file):
        """
        We'll assume that the datafile comes in a tfrecord_file
        """
        raw_dataset = tf.data.TFRecordDataset(filename)
        dataset_size = sum(1 for _ in raw_dataset)
        images = np.empty((dataset_size, shape[0], shape[1], shape[2]), dtype = np.float32)
        labels = np.empty(dataset_size, dtype = np.int32) 
        print('Loading {} images'.format(dataset_size))
        sys.stdout.flush()
        #todo parallel
        for i, record in enumerate(raw_dataset):
            im, lbl = data.parser_tfrecord_siamese(record, shape, mean_file)            
            images[i, ] = im
            labels[i] = lbl
        return images, labels    
                                           
    def __len__(self):
        return int(self.data_size / self.batch_size) 
    
    def __getitem__(self, index):
        idx = np.arange(index * self.batch_size , min((index + 1) * self.batch_size, self.data_size))                
        X, y = self.__get_batch(idx)
        return X, y
    
    def on_epoch_end(self):
        self.__make_pairs()                 
           
    def __get_batch(self, idxs):
        X_a = np.empty((len(idxs), self.shape[0], self.shape[1], self.shape[2]), dtype = np.float32)                
        X_p = np.empty((len(idxs), self.shape[0], self.shape[1], self.shape[2]), dtype = np.float32)
        X_n = np.empty((len(idxs), self.shape[0], self.shape[1], self.shape[2]), dtype = np.float32)        
        y = np.zeros((len(idxs), 3, self.num_classes), dtype = np.float32)
        y_emb = np.zeros(len(idxs), dtype = np.float32)
        for i, idx in enumerate(idxs) :            
            X_a[i,] = self.sk_images[self.sk_idx[idx]]            
            X_p[i,] = self.ph_images[self.idx_positives[idx]]
            X_n[i,] = self.ph_images[self.idx_negatives[idx]]
            y[i,0,self.sk_labels[self.sk_idx[idx]]] = 1.0
            y[i,1,self.ph_labels[self.idx_positives[idx]]] = 1.0
            y[i,2,self.ph_labels[self.idx_negatives[idx]]] = 1.0
            
        return [X_a, X_p, X_n], [y_emb, y]
        
    def __make_pairs(self):
        """        
        It makes up pairs for every image in the sketch collection
        """
        self.sk_idx = []
        self.idx_negatives = []
        self.idx_positives = []
        for idx in  self.group_idx :            
            idx_sk = self.sk_groups[idx]
            np.random.shuffle(idx_sk)
            idx_ph = np.random.choice(self.ph_groups[idx], len(idx_sk))                                                
            np.random.shuffle(idx_ph)
            idx_negatives = np.random.choice(np.where(self.ph_labels != idx)[0], len(idx_sk))
            np.random.shuffle(idx_negatives)            
            self.sk_idx.extend(idx_sk)
            self.idx_positives.extend(idx_ph)
            self.idx_negatives.extend(idx_negatives)                    
        self.sk_idx = np.array(self.sk_idx)
        self.idx_positives = np.array(self.idx_positives)
        self.idx_negatives = np.array(self.idx_negatives)    
        
        if self.shuffle :
            _idx = np.arange(len(self.sk_idx))
            np.random.shuffle(_idx)            
            self.sk_idx = self.sk_idx[_idx]
            self.idx_positives = self.idx_positives[_idx]
            self.idx_negatives = self.idx_negatives[_idx]
        
#unit test          
if __name__ == '__main__'   :
    data = SiameseDataGenerator('/home/vision/smb-datasets/SBIR/SiameseNet', 10, 250, 'test', True)
    fig, xs = plt.subplots(1,3)
    ii = 0;
    epoch = 0;
    while True :
        if ii % data.__len__() == 0 :
            ii = 0;
            data.on_epoch_end()       
            epoch = epoch + 1
            print(epoch)     
        x, y = data.__getitem__(ii)
        #x_a, x_p, x_n = np.split(x, 3, axis = 3)
        x_a = x[0]
        x_p = x[1]
        x_n = x[2]
        for i, aa in enumerate(x_a) :            
            xs[0].imshow(x_a[i])
            xs[0].set_axis_off()
            xs[0].set_title('Anchor')
            xs[1].imshow(x_p[i])
            xs[1].set_axis_off()
            xs[1].set_title('Positive')
            xs[2].imshow(x_n[i])
            xs[2].set_axis_off()
            xs[2].set_title('Negative')
            plt.pause(0.5)
              