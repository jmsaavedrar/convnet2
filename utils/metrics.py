import tensorflow as tf

 
def d_positive(y_true, y_pred):
    #y_true will be used for training cross_entropy
    e_a, e_p, _ = tf.split(y_pred, axis = 1, num_or_size_splits = 3)    
    d_p = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_p),2))
    d_p_hard = tf.math.reduce_mean(d_p)    
    #hardest negative and hardest positive
    return d_p_hard
 
def d_negative(y_true, y_pred):
    #y_true will be used for training cross_entropy
    e_a, _, e_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)    
    d_n = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_n),2))
    d_n_hard = tf.math.reduce_mean(d_n)    
    #hardest negative and hardest positive
    return d_n_hard

def simple_accuracy(y_true, y_pred):
    #B x n_clases
    correct_prediction = tf.equal(tf.argmax(y_true,1), tf.argmax(y_pred,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc
    
def metric_accuracy_siamese(y_true, y_pred):
    y_true_a, y_true_p, y_true_n = tf.split(y_true, axis = 1, num_or_size_splits = 3)
    cl_a, cl_p, cl_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)
                
    correct_prediction_a = tf.equal(tf.argmax(tf.squeeze(y_true_a),1), tf.argmax(tf.squeeze(cl_a),1))
    correct_prediction_p = tf.equal(tf.argmax(tf.squeeze(y_true_p),1), tf.argmax(tf.squeeze(cl_p),1))
    correct_prediction_n = tf.equal(tf.argmax(tf.squeeze(y_true_n),1), tf.argmax(tf.squeeze(cl_n),1))
    
    acc_a = tf.reduce_mean(tf.cast(correct_prediction_a, tf.float32))
    acc_p = tf.reduce_mean(tf.cast(correct_prediction_p, tf.float32))
    acc_n = tf.reduce_mean(tf.cast(correct_prediction_n, tf.float32))
    acc = (acc_a + acc_p + acc_n) / 3.0 
    return acc

