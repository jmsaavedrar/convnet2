from configparser import SafeConfigParser

class ConfigurationFile:
    """
     An instance of ConfigurationFile contains required parameters to train a 
     convolutional neural network
    """    
    def __init__(self, str_config, modelname):
        config = SafeConfigParser({'USE_MULTITHREADS' : 'False',                                    
                                   'IMAGE_SIZE': '1',
                                   'CKPFILE': 'NONE',
                                   'KEEP_ASPECT_RATIO' : 'True',
                                   'USE_L2' : 'False',
                                   'WEIGHT_DECAY' : '0.0',
                                   'SHUFFLE_SIZE' : '1000'
                                   })
        config.read(str_config)
        self.sections = config.sections()                
        if modelname in self.sections:
            try :
                self.modelname = modelname                
                #self.arch = config.get(modelname, "ARCH")
                self.process_fun = 'default'
                if 'PROCESS_FUN' in config[modelname] is not None :
                    self.process_fun = config[modelname]['PROCESS_FUN']                                                    
                self.number_of_classes = config.getint(modelname,"NUM_CLASSES")
                self.number_of_epochs= config.getint(modelname,"NUM_EPOCHS")                                
                self.batch_size = config.getint(modelname, "BATCH_SIZE")
                self.snapshot_steps = config.getint(modelname, "SNAPSHOT_STEPS")
                #test time sets when test is run (in seconds)
                self.validation_steps = config.getint(modelname, "VALIDATION_STEPS")
                self.lr = config.getfloat(modelname, "LEARNING_RATE")
                #snapshot folder, where training data will be saved
                self.snapshot_prefix = config.get(modelname, "SNAPSHOT_DIR")
                self.data_dir = config.get(modelname,"DATA_DIR")
                self.channels = config.getint(modelname,"CHANNELS")
                self.keep_aspect_ratio = config.getboolean(modelname, "KEEP_ASPECT_RATIO")                            
                self.image_size = config.getint(modelname, "IMAGE_SIZE")
                self.image_width = config.getint(modelname, "IMAGE_WIDTH")
                self.image_height = config.getint(modelname, "IMAGE_HEIGHT")                
                self.image_type = config.get(modelname, "IMAGE_TYPE").upper()
                self.checkpoint_file = config.get(modelname, "CKPFILE")
                self.use_l2 = config.getboolean(modelname, "USE_L2")
                self.shuffle_size = config.getint(modelname, "SHUFFLE_SIZE")
                self.weight_decay = 0.0
                if self.use_l2 :
                    self.weight_decay = config.getfloat(modelname, 'WEIGHT_DECAY')
                self.is_multithreads = config.getboolean(modelname, "USE_MULTITHREADS")
                self.num_threads = 1
                if self.is_multithreads :
                    self.num_threads = config.getint(modelname, "NUM_THREADS")
                assert(self.channels == 1 or self.channels == 3)                
                assert(self.num_threads > 0)
            except Exception:
                raise ValueError("something wrong with configuration file " + str_config)
        else:
            raise ValueError(" {} is not a valid section".format(modelname))
        
    def get_model_name(self):
        return self.modelname
    
    def get_process_fun(self):
        return self.process_fun
       
    def get_number_of_classes(self) :
        return self.number_of_classes
    
    def get_number_of_epochs(self):
        return self.number_of_epochs
    
    def get_batch_size(self):
        return self.batch_size
   
    def get_snapshot_steps(self):
        return self.snapshot_steps
    
    def get_snapshot_dir(self):
        return self.snapshot_prefix
    
    def get_number_of_channels(self):
        return self.channels
    
    def get_data_dir(self):
        return self.data_dir
    
    def get_learning_rate(self):
        return self.lr
    
    def get_validation_steps(self):
        return self.validation_steps      
    
    def use_keep_aspect_ratio(self):
        return self.keep_aspect_ratio
    
    def get_image_size(self):
        return self.image_size
    
    def get_image_width(self):
        return self.image_width
    
    def get_image_height(self):
        return self.image_height
    
    def use_multithreads(self):
        return self.is_multithreads
    
    def get_num_threads(self):
        return self.num_threads
    
    def is_a_valid_section(self, str_section):
        return str_section in self.sections
    
    def get_image_shape(self):
        return (self.image_height, self.image_width, self.channels)
    
    def get_image_type(self):
        return self.image_type
    
    def use_checkpoint(self):
        use_ckp = False
        if self.checkpoint_file != "NONE" :
            use_ckp = True
        return use_ckp
    
    def get_checkpoint_file(self):
        return self.checkpoint_file
    
    def use_l2_regularization(self):
        return self.use_l2
    
    def get_weight_decay(self):
        return self.weight_decay
    
    def get_shuffle_size(self):
        return self.shuffle_size
    
    def show(self):
        print("NUM_EPOCHS: {}".format(self.get_number_of_epochs()))        
        print("LEARNING_RATE: {}".format(self.get_learning_rate()))                
        print("SNAPSHOT_DIR: {}".format(self.get_snapshot_dir()))
        print("DATA_DIR: {}".format(self.get_data_dir()))
        print("USE_CHECKP: {}".format(self.use_checkpoint()))
        print("IMAGE_TYPE: {}".format(self.get_image_type()))