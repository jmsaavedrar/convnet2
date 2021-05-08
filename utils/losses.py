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
    shape of y_true = [Bx10]
    shape of y_pred = [Bx10]
    This is the classical categorical crossentropy
    """
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)    
    return ce

def crossentropy_l2_loss(model, weight_decay = 0):
    def loss(y_true, y_pred):
        """ 
        This uses crossentropy plus l2 regularization
        """
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        l2_loss = l2_regularization_loss(model, weight_decay)
        return ce + l2_loss
    return loss

#constrastive loss for triplet larning
def triplet_loss(margin = 20):
    def loss(y_true, y_pred):
        #y_true will be used for training cross_entropy
        e_a, e_p, e_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)    
        d_p = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_p), 2))
        d_p_hard = tf.math.reduce_mean(d_p)
        d_n = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_n), 2))
        d_n_hard = tf.math.reduce_mean(d_n)
        #hardest negative and hardest positive
        return tf.maximum(1e-10, d_p_hard + margin - d_n_hard)
    return loss

#crossentropy loss for triplets    
def crossentropy_triplet_loss(y_true, y_pred):
    y_true_a, y_true_p, y_true_n = tf.split(y_true, axis = 1, num_or_size_splits = 3)
    cl_a, cl_p, cl_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)            
    ce_a = tf.keras.losses.categorical_crossentropy(tf.squeeze(y_true_a), tf.squeeze(cl_a), from_logits=True)
    ce_p = tf.keras.losses.categorical_crossentropy(tf.squeeze(y_true_p), tf.squeeze(cl_p), from_logits=True)
    ce_n = tf.keras.losses.categorical_crossentropy(tf.squeeze(y_true_n), tf.squeeze(cl_n), from_logits=True)    
    ce = (ce_a + ce_p + ce_n) / 3.0 
    return ce