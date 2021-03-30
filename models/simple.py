"""
A simple cnn used for training mnist.
"""
import tensorflow as tf


class SimpleModel(tf.keras.Model):
    def __init__(self, number_of_classes):
        super(SimpleModel, self).__init__()
        #define layers which require parameters to be learned
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'valid',  kernel_initializer = 'he_normal')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'valid')
        self.relu = tf.keras.layers.ReLU();        
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'valid',  kernel_initializer='he_normal')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding = 'valid', kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()
        
        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal')
        self.bn_fc_1 = tf.keras.layers.BatchNormalization()

        self.fc2 = tf.keras.layers.Dense(number_of_classes)

    # here the architecture is defined
    def call(self, inputs):
        # input  # [B, 31,31, 1]  # [B, H, W, C]
        #first block
        x = self.conv_1(inputs) # output is 29x29   
        x = self.bn_conv_1(x) # 29x29
        x = self.relu(x) 
        x = self.max_pool(x) # 27 x 27 -> ((27+1)/2) = 14 x14  
        #second block
        x = self.conv_2(x)  # 12 x 12
        x = self.bn_conv_2(x) 
        x = self.relu(x) 
        x = self.max_pool(x) # 10 / 2 -> 5x5
        #third block
        x = self.conv_3(x)  # 3x3
        x = self.bn_conv_3(x) # 3x3
        x = self.relu(x)  
        #last block        
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc1(x) 
        x = self.bn_fc_1(x) 
        x = self.relu(x) 

        x = self.fc2(x)

        return x