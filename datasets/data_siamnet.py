import tensorflow as tf
import numpy as np

def _filter(sk, im):
    _, y_1 = sk
    _, y_2 = im
    label_cond = tf.equal(y_1, y_2)

    different_labels = tf.fill(tf.shape(label_cond), 0.5)
    same_labels = tf.fill(tf.shape(label_cond), 0.5)

    weights = tf.where(label_cond, same_labels, different_labels)
    random_tensor = tf.random.uniform(shape=tf.shape(weights))

    return weights > random_tensor

def parser(a,b):
    x_1, y_1 = a
    x_2, y_2 = b
    y = tf.cast(tf.equal(y_1, y_2), tf.int32)
    return x_1, x_2, y
    
     
if __name__ == '__main__' :
    x_1 = np.array([10,100,1000,10000])
    x_2 = np.array([20,200,2000, 20000])
    y_1 = np.array([1,2,3,4])
    y_2 = np.array([1,2,3,4])
    
    
    dataset_1 = tf.data.Dataset.from_tensor_slices((x_1,y_1))
    dataset_1 = dataset_1.repeat(3)
    dataset_1 = dataset_1.shuffle(2)
    dataset_2 = tf.data.Dataset.from_tensor_slices((x_2,y_2))
    dataset_2 = dataset_2.repeat(2)
    dataset_2 = dataset_2.shuffle(10)
    dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
    dataset = dataset.filter(_filter)
    dataset = dataset.repeat(3)
    dataset = dataset.map(lambda r, l : parser(r,l))
    
    
    for x, y, z in dataset :        
        print('{} {} {}'.format(x,y, z))
    
