"""
 author: jsaavedr
 April, 2020 
 This is a general implementation of ResNet, and it optionally includes SE blocks  
 all layers are initialized as "he_normal"
"""
import tensorflow as tf
import sys
sys.path.append("/home/jsaavedr/Research/git/tensorflow-2/convnet2")




# a conv 3x3

def conv3x3(channels, stride = 1, **kwargs):
    return tf.keras.layers.Conv2D(channels, (3,3), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal', 
                                  **kwargs)

def conv1x1(channels, stride = 1, **kwargs):
    return tf.keras.layers.Conv2D(channels, 
                                  (1,1), 
                                  strides = stride, 
                                  padding = 'same', 
                                  kernel_initializer = 'he_normal',
                                  **kwargs)


class SEBlock(tf.keras.layers.Layer):
    """
    Squeeze and Excitation Block
    r_channels is the number of reduced channels
    """
    
    def __init__(self, channels, r_channels, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.gap  = tf.keras.layers.GlobalAveragePooling2D(name = 'se_gap')
        self.fc_1 = tf.keras.layers.Dense(r_channels, name = 'se_fc1' )
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'se_bn1')        
        self.fc_2 = tf.keras.layers.Dense(channels, name = 'se_fc2')    
            
    def call(self, inputs, training = True):       
        y = self.gap(inputs)
        y = tf.keras.activations.relu(self.bn_1(self.fc_1(y), training))
        scale = tf.keras.activations.sigmoid(self.fc_2(y))
        scale = tf.reshape(scale, (-1,1,1,self.channels))
        y = tf.math.multiply(inputs, scale)
        return y        
        

class ResidualBlock(tf.keras.layers.Layer):
    """
    residual block implementated in a full preactivation mode
    input bn-relu-conv1-bn-relu-conv2->y-------------------
      |                                                    |+
      ------------------(projection if necessary)-->shortcut--> y + shortcut
        
    """    
    def __init__(self, filters, stride, use_projection = False, se_factor = 0,  **kwargs):        
        super(ResidualBlock, self).__init__(**kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(name = 'bn_0')
        self.conv_1 = conv3x3(filters, stride, name = 'conv_1', use_bias = False)
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1', )
        self.conv_2 = conv3x3(filters, 1, name = 'conv_2', use_bias = False)
        self.use_projection = use_projection;
        self.projection = 0
        if self.use_projection :                            
            self.projection = conv1x1(filters, stride, name = 'projection', use_bias = False)
        
        self.se = 0
        self.use_se_block = False
        if se_factor > 0 :
            self.se = SEBlock(filters, filters / se_factor)
            self.use_se_block = True
        
    #using full pre-activation mode
    def call(self, inputs, training = True):
        y = self.bn_0(inputs)
        y = tf.keras.activations.relu(y)
        if self.use_projection :
            shortcut = self.projection(y)
        else :
            shortcut = inputs
        y = self.conv_1(y)
        y = self.bn_1(y, training)
        y = tf.keras.activations.relu(y)
        y = self.conv_2(y)        
        if self.use_se_block :
            y = self.se(y)        
        y = shortcut + y # residual function        
        return y


class BottleneckBlock(tf.keras.layers.Layer):
    """
    BottleneckBlock
    expansion rate = x4
    """
    
    def __init__(self, filters, stride, use_projection = False, se_factor = 0, **kwargs):        
        super(BottleneckBlock, self).__init__(**kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(name = 'bn_0')
        #conv_0 is the compression layer
        self.conv_0 = conv1x1(filters, stride, name = 'conv_0', use_bias = False)
        self.conv_1 = conv3x3(filters, 1, name = 'conv_1')
        self.bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1')
        self.conv_2 = conv1x1(filters * 4, 1, name = 'conv_2', use_bias = False)
        self.bn_2 = tf.keras.layers.BatchNormalization(name = 'bn_2')
        self.use_projection = use_projection
        self.projection = 0
        if self.use_projection :                            
            self.projection = conv1x1(filters * 4, stride, name = 'projection', use_bias = False)
        self.se = 0
        self.use_se_block = False
        if se_factor > 0 :
            self.se = SEBlock(filters, filters / se_factor)
            self.use_se_block = True
        
    #using full pre-activation mode
    def call(self, inputs, training = True):
        #full-preactivation
        y = self.bn_0(inputs, training)
        y = tf.keras.activations.relu(y)
        if self.use_projection :
            shortcut = self.projection(y)
        else :
            shortcut = inputs            
        y = self.conv_0(y)
        y = self.bn_1(y, training)
        y = tf.keras.activations.relu(y)
        y = self.conv_1(y)
        y = self.bn_2(y, training)
        y = tf.keras.activations.relu(y)
        y = self.conv_2(y)        
        if self.use_se_block :
            y = self.se(y)        
        y = shortcut + y # residual function        
        return y

class ResNetBlock(tf.keras.layers.Layer):
    """
    resnet block implementation
    A resnet block contains a set of residual blocks
    Commonly, the residual block of a resnet block starts with a stride = 2, except for the first block
    The number of blocks together with the number of filters used in each block  are defined in __init__    
    """
    
    def __init__(self, filters,  block_size, with_reduction = False, use_bottleneck = False, se_factor = 0, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)        
        self.filters = filters    
        self.block_size = block_size
        if use_bottleneck :
            residual_block = BottleneckBlock
        else:
            residual_block = ResidualBlock            
        #the first block is nos affected by a spatial reduction 
        stride_0 = 1
        use_projection_at_first = False
        if with_reduction or use_bottleneck :
            stride_0 = 2
            use_projection_at_first = True
        self.block_collector = [residual_block(filters = filters, stride = stride_0, use_projection = use_projection_at_first, se_factor = se_factor, name = 'rblock_0')]        
        for idx_block in range(1, block_size) :
            self.block_collector.append(residual_block(filters = filters, stride = 1, se_factor = se_factor, name = 'rblock_{}'.format(idx_block)))
                    
    def call(self, inputs, training):
        x = inputs;
        for block in self.block_collector :
            x = block(x, training)
        return x;


class ResNetBackbone(tf.keras.Model):
    
    def __init__(self, block_sizes, filters, use_bottleneck = False, se_factor = 0, **kwargs) :
        super(ResNetBackbone, self).__init__(**kwargs)
        self.conv_0 = tf.keras.layers.Conv2D(64, (7,7), strides = 2, padding = 'same', 
                                             kernel_initializer = 'he_normal', 
                                             name = 'conv_0', use_bias = False)
        
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')
        self.resnet_blocks = [ResNetBlock(filters = filters[0], 
                                          block_size = block_sizes[0], 
                                          with_reduction = False,  
                                          use_bottleneck = use_bottleneck, 
                                          se_factor = se_factor, 
                                          name = 'block_0')] 
        for idx_block in range(1, len(block_sizes)) :                     
            self.resnet_blocks.append(ResNetBlock(filters = filters[idx_block], 
                                                  block_size = block_sizes[idx_block], 
                                                  with_reduction = True,  
                                                  use_bottleneck = use_bottleneck,
                                                  se_factor = se_factor,
                                                  name = 'block_{}'.format(idx_block)))
            self.bn_last= tf.keras.layers.BatchNormalization(name = 'bn_last')
            
        
    def call(self, inputs, training):
        x = inputs
        x = self.conv_0(x)
        x = self.max_pool(x)                 
        for block in self.resnet_blocks :
            x = block(x, training)      
        x = self.bn_last(x)                
        x = tf.keras.activations.relu(x)  
        return x
    
class ResNet(tf.keras.Model):
    """ 
    ResNet model 
    e.g.    
    block_sizes: it is the number of residual components for each block e.g  [2,2,2] for 3 blocks 
    filters : it is the number of channels within each block [32,64,128]
    number_of_classes: The number of classes of the underlying problem
    use_bottleneck: Is's true when bottlenect blocks are used.
    se_factor : reduction factor in  SE module, 0 if SE is not used
    """        
    
    def __init__(self, block_sizes, filters, number_of_classes, use_bottleneck = False, se_factor = 0, **kwargs) :
        super(ResNet, self).__init__(**kwargs)
        self.backbone = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, name = 'backbone')                            
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()                     
        self.classifier = tf.keras.layers.Dense(number_of_classes, name='classifier')
        
    def call(self, inputs, training):
        x = inputs
        x = self.backbone(x, training)    
        x = self.avg_pool(x)                
        x = tf.keras.layers.Flatten()(x)                        
        x = self.classifier(x)
        return x
    

class SiameseNet(tf.keras.Model):
    
    def __init__(self, block_sizes, filters,  number_of_classes, use_bottleneck = False, se_factor = 0, **kwargs) :
        super(SiameseNet, self).__init__(**kwargs)
        #the following backbones does not share  weights 
        self.sk_backbone = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, name = 'sk_backbone')
        self.ph_backbone = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, name = 'ph_backbone')
        #next are shared block
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()            
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc_1')
        self.bn1 = tf.keras.layers.BatchNormalization(name = 'bn_1') 
        self.fc2 = tf.keras.layers.Dense(512, name = 'fc_2')
        self.classifier = tf.keras.layers.Dense(number_of_classes, name = 'classifier')
     
    def call(self, inputs, training):
        #split net into anchor, positive and negative
        #x_a , x_p, x_n =  tf.split(inputs, axis = 3, num_or_size_splits = 3)
        x_a= inputs[0]
        x_p= inputs[1]
        x_n= inputs[2]
        f_sketch = self.sk_backbone(x_a, training)
        f_sketch = self.avg_pool(f_sketch)
        f_sketch = tf.keras.layers.Flatten()(f_sketch)
        f_sketch = self.fc2(tf.keras.activations.relu(self.bn1(self.fc1(f_sketch), training)))
        #classifier_sketch
        cl_sketch = self.classifier(f_sketch)
        #normalized_feature_sketch
        f_sketch = tf.truediv(f_sketch, (tf.norm(f_sketch) + 1.0e-10))        
                
        f_positive = self.ph_backbone(x_p, training)
        f_positive = self.avg_pool(f_positive)
        f_positive = tf.keras.layers.Flatten()(f_positive)
        f_positive = self.fc2(tf.keras.activations.relu(self.bn1(self.fc1(f_positive), training)))
        #classifier_positive
        cl_positive = self.classifier(f_positive)
        #normalized_feature_positive
        f_positive = tf.truediv(f_positive, (tf.norm(f_positive) + 1.0e-10), name = 'emb_image')
        
        f_negative = self.ph_backbone(x_n, training)        
        f_negative = self.avg_pool(f_negative)
        f_negative = tf.keras.layers.Flatten()(f_negative)
        f_negative = self.fc2(tf.keras.activations.relu(self.bn1(self.fc1(f_negative), training)))
        #classifier_negative
        cl_negative = self.classifier(f_negative)
        #normalized_feature_negative
        f_negative = tf.truediv(f_negative , (tf.norm(f_negative) + 1.0e-10), name = 'emb_image')                        
        #f_sketch, f_negative, f_positive, 
        #axis 1 for keeping all features together
        f_sketch = tf.expand_dims(f_sketch, 1)
        f_positive = tf.expand_dims(f_positive, 1)
        f_negative = tf.expand_dims(f_negative, 1)        
        #concanenate
        embeddings = tf.concat([f_sketch, f_positive, f_negative], axis = 1)
        #for classification, it is possible to return classification as embeddings, classification
        #let's concatenate logits
        cl_sketch = tf.expand_dims(cl_sketch, 1)
        cl_positive = tf.expand_dims(cl_positive, 1)
        cl_negative = tf.expand_dims(cl_negative, 1)
        #finally concatenate the three logits
        logits = tf.concat([cl_sketch, cl_positive, cl_negative], axis = 1)
        return  embeddings, logits


class SiameseNetImage(tf.keras.Model):
    
    def __init__(self, block_sizes, filters,  use_bottleneck = False, se_factor = 0, **kwargs) :
        super(SiameseNetImage, self).__init__(**kwargs)        
        self.ph_backbone = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, name = 'ph_backbone')
        #next are shared block
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()            
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc_1')
        self.bn1 = tf.keras.layers.BatchNormalization(name = 'bn_1') 
        self.fc2 = tf.keras.layers.Dense(512, name = 'fc_2')
     
    def call(self, inputs, training):        
        f_positive = self.ph_backbone(inputs, training)
        f_positive = self.avg_pool(f_positive)
        f_positive = tf.keras.layers.Flatten()(f_positive)
        f_positive = self.fc2(tf.keras.activations.relu(self.bn1(self.fc1(f_positive), training)))
        f_positive = tf.truediv(f_positive, (tf.norm(f_positive) + 1.0e-10), name = 'emb_image')
                                    
        return f_positive    

class SiameseNetSketch(tf.keras.Model):
    
    def __init__(self, block_sizes, filters,  use_bottleneck = False, se_factor = 0, **kwargs) :
        super(SiameseNetSketch, self).__init__(**kwargs)        
        self.sk_backbone = ResNetBackbone(block_sizes, filters, use_bottleneck, se_factor, name = 'sk_backbone')
        #next are shared block
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()            
        self.fc1 = tf.keras.layers.Dense(512, name = 'fc_1')
        self.bn1 = tf.keras.layers.BatchNormalization(name = 'bn_1') 
        self.fc2 = tf.keras.layers.Dense(512, name = 'fc_2')                
     
    def call(self, inputs, training):        
        f_sketch = self.sk_backbone(inputs, training)
        f_sketch = self.avg_pool(f_sketch)
        f_sketch = tf.keras.layers.Flatten()(f_sketch)
        f_sketch = self.fc2(tf.keras.activations.relu(self.bn1(self.fc1(f_sketch), training)))
        f_sketch = tf.truediv(f_sketch, (tf.norm(f_sketch) + 1.0e-10), name = 'emb_sketch')
                                    
        return f_sketch        
    
"""
A unit test
"""    
if __name__ == '__main__' :
    #a = tf.constant([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]], dtype = tf.float32)    
    #b = tf.constant([1,2,3], dtype = tf.float32)
    #b = tf.reshape(b, [1,1,3])
    #print(a)
    #print(b)
    #c = tf.math.multiply(a,b)
    #sess = tf.Session()
    #print(sess.run(c))
    #model = ResNet(block_sizes=[3,4,6,3], filters = [16, 128, 256, 512], number_of_classes = 10)
    input_sketch = tf.keras.Input((224,224,3), name = 'input_sketch')
    input_positive = tf.keras.Input((224,224,3), name = 'input_image') 
    input_negative = tf.keras.Input((224,224,3), name = 'input_image')
    model = SiameseNet([3,4,6,3],[64,128,256,512], 250)    
    model([input_sketch, input_positive, input_negative])
    model.summary()
    
        #print('{} {}'.format(v.name, v.shape))
        
    #model.save('the-model.pb', save_format='tf')
    #model.save("the-model")
#         
#     
    