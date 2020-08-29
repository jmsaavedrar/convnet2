import tensorflow as tf

 
def d_positive(y_true, y_pred):
    #y_true will be used for training cross_entropy
    e_a, e_p, _ = tf.split(y_pred, axis = 1, num_or_size_splits = 3)    
    d_p = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_p),2))
    d_p_hard = tf.math.reduce_max(d_p)    
    #hardest negative and hardest positive
    return d_p_hard
 
def d_negative(y_true, y_pred):
    #y_true will be used for training cross_entropy
    e_a, _, e_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)    
    d_n = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_n),2))
    d_n_hard = tf.math.reduce_min(d_n)    
    #hardest negative and hardest positive
    return d_n_hard

