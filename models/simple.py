"""
A simple cnn used for training mnist.
"""
import tensorflow as tf


class SimpleModel(tf.keras.Model):
    def __init__(self, number_of_classes):
        super(SimpleModel, self).__init__()        
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'same',  kernel_initializer = 'he_normal')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'same')
        self.relu = tf.keras.layers.ReLU();        
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same',  kernel_initializer='he_normal')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()
        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding = 'same', kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()     
        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal')
        self.bn_fc_1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(number_of_classes)

    # here, connecting the modules
    def call(self, inputs):
        # input  # [B, 31,31, 1]  # [B, H, W, C]
        #first block
        x = self.conv_1(inputs) #Bx31x31x32   
        x = self.bn_conv_1(x) #Bx31x31xx32
        x = self.relu(x) 
        x = self.max_pool(x) #31 x 31 -> ((31+1)/2) = Bx16x16x32   
        #second block
        x = self.conv_2(x)  # Bx16x16x64
        x = self.bn_conv_2(x) # Bx16x16x64
        x = self.relu(x) 
        x = self.max_pool(x) #16x16x64->16/ 2 -> Bx8x8x64
        #third block
        x = self.conv_3(x)  # Bx8x8x128
        x = self.bn_conv_3(x) #Bx8x8x128
        x = self.relu(x)  
        x = self.max_pool(x) #8x8x128->8/ 2 -> Bx4x4x128
        #last block        
        x = tf.keras.layers.Flatten()(x) 
        x = self.fc1(x)  # 256
        x = self.bn_fc_1(x) # 256
        x = self.relu(x) 
        x = self.fc2(x) #10
        return x