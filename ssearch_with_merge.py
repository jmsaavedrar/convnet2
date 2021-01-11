import sys
sys.path.append("/home/jsaavedr/Research/git/tensorflow-2/convnet2")
import tensorflow as tf
import models.resnet as resnet
import datasets.data as data
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.io as io
import skimage.transform as trans
import os
import argparse
import numpy as np

class SSearchMerge :
    def __init__(self, config_file, model_name):
        
        self.configuration = conf.ConfigurationFile(config_file, model_name)
        #loading data
        mean_file = os.path.join(self.configuration.get_data_dir(), "mean.dat")
        shape_file = os.path.join(self.configuration.get_data_dir(),"shape.dat")
        #
        self.input_shape =  np.fromfile(shape_file, dtype=np.int32)
        self.mean_image = np.fromfile(mean_file, dtype=np.float32)
        self.mean_image = np.reshape(self.mean_image, self.input_shape)
       
        #loading classifier model
#         self.model = resnet.ResNetFeatureVector([3,4,6,3],[64,128,256,512], se_factor = 0)
        input_image = tf.keras.Input((self.input_shape[0], self.input_shape[1], self.input_shape[2]), name = 'input_image')     
#         self.model(input_image)    
#         self.model.summary()
#         self.model.load_weights(self.configuration.get_checkpoint_file(), by_name = True, skip_mismatch = True)
        
        #color model
        model = resnet.ResNetFeatureVector([3,4,6,3],[64,128,256,512], se_factor = 0)        
        model(input_image)    
        model.summary() 
        model.load_weights(self.configuration.get_checkpoint_file(), by_name = True, skip_mismatch = True)                    
        #create the sim-model with a customized layer         
        output1 = model.output
        output2 = model.get_layer('backbone').get_layer('block_1').output
        output2 = tf.keras.layers.GlobalAveragePooling2D()(output2)
        output = tf.concat([output1, output2], axis = 1)
        self.model_color = tf.keras.Model(model.input, output)                
        self.model_color.summary()                                 
        self.model_color.save('color_model')   
        print('model was loaded OK')
        #defining process arch
        self.process_fun =  imgproc.process_image
        #loading catalog
        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), 'ssearch')
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt')        
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file)
        print('loading catalog ...')
        self.load_catalog(catalog_file)
        print('loading catalog ok ...')
        sys.stdout.flush()
        self.enable_search = False        
        
    #load model
    def read_image(self, filename):
        target_size = (self.configuration.get_image_height(), self.configuration.get_image_width())
        image = self.process_fun(data.read_image(filename, self.configuration.get_number_of_channels()), target_size )
        return image
    
    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")                        
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype = np.int32)
        self.features = np.fromfile(fvs_file, dtype = np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')
        
    def load_catalog(self, catalog):
        with open(catalog) as f_in :
            self.filenames = [filename.strip() for filename in f_in ]
        self.data_size = len(self.filenames)    
            
    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]
        
    def compute_features(self, image, expand_dims = False):
        image = image - self.mean_image
        if expand_dims :
            image = tf.expand_dims(image, 0)                
        fv= self.model_color.predict(image)
        return fv        
    
    def normalize(self, data) :
        norm = np.sum(np.square(data), axis = 1)
        norm = np.expand_dims(norm, 0)        
        data = data / np.transpose(norm)
        return data
 
    def search(self, im_query):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims = True)
        #sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
        #sim = np.reshape(sim, (-1))
        #it seems that Euclidean performs better than cosine
        d = np.sqrt(np.sum(np.square(self.features - q_fv[0]), axis = 1))
        #idx_sorted = np.argsort(-sim)        
        idx_sorted = np.argsort(d)
        print(d[idx_sorted[:10]])
        return idx_sorted[:90]
        
                                
    def compute_features_from_catalog(self):
        n_batch = self.configuration.get_batch_size()        
        images = np.empty((self.data_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = np.uint8)
        print('reading images')
        for i, filename in enumerate(self.filenames) :
            if i % 1000 == 0:
                print('reading {}'.format(i))
                sys.stdout.flush()            
            images[i, ] = self.read_image(filename)        
        n_iter = np.int(np.ceil(self.data_size / n_batch))
        result = []
        for i in range(n_iter) :
            print('iter {} / {}'.format(i, n_iter))  
            sys.stdout.flush()             
            batch = images[i*n_batch : min((i + 1) * n_batch, self.data_size), ]
            result.append(self.compute_features(batch))
        fvs = np.concatenate(result)    
        print('fvs {}'.format(fvs.shape))    
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)       
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def draw_result(self, filenames):
        w = 1000
        h = 1000
        w_i = np.int(w / 10)
        h_i = np.int(h / 10)
        image_r = np.zeros((w,h,3), dtype = np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames) :
            pos = (i * w_i)
            x = pos % w
            y = np.int(np.floor(pos / w)) * h_i
            image = self.read_image(filename)            
            image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
            image_r[y:y+h_i, x : x +  w_i, :] = image              
        return image_r    
                    
#unit test        
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description = "Similarity Search")        
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)                
    parser.add_argument("-mode", type=str, choices = ['search', 'compute','extract'], help=" mode of operation", required = True)
    parser.add_argument("-image", type=str,  help=" image for extracting test", required = False)
    parser.add_argument("-list", type=str,  help=" list of image to process", required = False)
    parser.add_argument("-odir", type=str,  help=" output dir", required = False, default = '.')
    pargs = parser.parse_args()     
    configuration_file = pargs.config        
    ssearch = SSearchMerge(pargs.config, pargs.name)    
    #filename = '/home/vision/smb-datasets/clothing-dataset/classifier_data/dress/91f98166-e897-47d8-b.png'    
    #fv = ssearch.compute_features(filename)
    if pargs.mode == 'extract' :
        fquery = pargs.image
        im_query = ssearch.read_image(fquery)
        fv = ssearch.compute_features(im_query, expand_dims = True)
        print('{} {}'.format(fv.shape, fv[0, :10]))
            
    if pargs.mode == 'compute' :
        ssearch.compute_features_from_catalog()
        
    if pargs.mode == 'search' :
        ssearch.load_features()
        #fquery =  '/home/vision/smb-datasets/clothing-dataset/classifier_data/dress/1b550b68-e499-49dd-9.png'
        #fquery = '/home/vision/smb-datasets/missodd/queries/missodd-query-2.png'
        if pargs.list is not None :
            with open(pargs.list) as f_list :
                filenames  = [ item.strip() for item in f_list]
            for fquery in filenames :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query)                
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)#           
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_result.png'
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))                
        else :
            fquery = input('Query:')
            while fquery != 'quit' :
                im_query = ssearch.read_image(fquery)
                idx = ssearch.search(im_query)
                #print(idx)
                r_filenames = ssearch.get_filenames(idx)
                r_filenames.insert(0, fquery)
    #             for f in r_filenames :
    #                 print(f)
                image_r= ssearch.draw_result(r_filenames)
                output_name = os.path.basename(fquery) + '_result.png'
                output_name = os.path.join(pargs.odir, output_name)
                io.imsave(output_name, image_r)
                print('result saved at {}'.format(output_name))
                fquery = input('Query:')
        
        