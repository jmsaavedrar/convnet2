import tensorflow as tf

def d_positive(y_true, y_pred):
    #y_true will be used for training cross_entropy
    e_a, e_p, _ = tf.split(y_pred, axis = 2, num_or_size_splits = 3)
    d_p = tf.sqrt(tf.square(e_a - e_p))
    d_p_hard = tf.math.reduce_mean(d_p)    
    #hardest negative and hardest positive
    return d_p_hard
    
def d_negative(y_true, y_pred):
    #y_true will be used for training cross_entropy
    e_a, _, e_n = tf.split(y_pred, axis = 2, num_or_size_splits = 3)
    d_n = tf.sqrt(tf.square(e_a - e_n))
    d_n_hard = tf.math.reduce_mean(d_n)    
    #hardest negative and hardest positive
    return d_n_hard