import tensorflow as tf

def l2_regularization_loss(model, weight_decay):
    variable_list = []
    for variable in model.trainable_variables :
        if 'kernel' in variable.name :             
            variable_list.append(tf.nn.l2_loss(variable))        
    val_loss = tf.add_n(variable_list)
    return val_loss*weight_decay; 

def crossentropy_loss(y_true, y_pred):
    """
    This is the classical categorical crossentropy
    """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)    
    return ce

def crossentropy_l2_loss(y_true, y_pred, model, weight_decay = 0):
    """ 
    This uses crossentropy plus l2 regularization
    """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    l2_loss = l2_regularization_loss(model, weight_decay)
    return ce + l2_loss


def constrastive_loss(y_true, y_pred, margin = 20):
    #y_true will be used for training cross_entropy
    e_a, e_p, e_n = tf.split(y_pred, axis = 2, num_or_size_splits = 3)
    d_p = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_p)))
    d_p_hard = tf.math.reduce_max(d_p)
    d_n = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_n)))
    d_n_hard = tf.math.reduce_min(d_n)
    #hardest negative and hardest positive
    return tf.maximum(1e-10, d_p_hard + margin - d_n_hard)
    