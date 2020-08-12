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