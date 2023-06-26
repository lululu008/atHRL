import tensorflow as tf

@tf.function
def get_intention(intention):
    if tf.less(intention[0], tf.constant(1.00)):
        return tf.constant(0)
    elif tf.less(intention[0], tf.constant(2.00)):
        return tf.constant(1)
    elif tf.less(intention[0], tf.constant(3.00)):
        return tf.constant(2)
    elif tf.less(intention[0], tf.constant(4.00)):
        return tf.constant(3)
    else:
        return tf.constant(4)

@tf.function
def intention_value(intention_action):
    intention = intention_action[0]
    if tf.less(intention[0], tf.constant(1.00)):
        return tf.constant([0])
    elif tf.less(intention[0], tf.constant(2.00)):
        return tf.constant([1])
    elif tf.less(intention[0], tf.constant(3.00)):
        return tf.constant([2])
    elif tf.less(intention[0], tf.constant(4.00)):
        return tf.constant([3])
    else:
        return tf.constant([4])

def duplicate_digits_2(logits):

    indices_to_remove = tf.convert_to_tensor([False, True])

    # 1d indices
    idx_remove = tf.where( indices_to_remove == True )[:,-1]
    idx_keep = tf.where( indices_to_remove == False )[:,-1]
    
    values_remove = tf.gather( logits[0], idx_keep )
    values_keep = tf.gather( logits[0], idx_keep )

    # to create a sparse vector we still need 2d indices like [ [0,1], [0,2], [0,10] ]
    # create vectors of 0's that we'll later stack with the actual indices
    zeros_remove = tf.zeros_like(idx_remove)
    zeros_keep = tf.zeros_like(idx_keep)

    idx_remove = tf.stack( [ zeros_remove, idx_remove], axis=1 )
    idx_keep = tf.stack( [ zeros_keep, idx_keep], axis=1 )

    # now we can create a sparse matrix
    logits_remove = tf.SparseTensor( idx_remove, values_remove, tf.shape(logits, out_type = tf.int64))
    logits_keep = tf.SparseTensor( idx_keep, values_keep, tf.shape(logits, out_type = tf.int64))

    # add together the two matrices (need to convert them to dense first)
    filtered_logits = tf.add(
        tf.sparse.to_dense(logits_remove, default_value = 0. ),
        tf.sparse.to_dense(logits_keep, default_value = 0. )
    )

    return filtered_logits

def duplicate_digits_3(logits):

    indices_to_remove = tf.convert_to_tensor([False, True, True])

    # 1d indices
    idx_remove = tf.where( indices_to_remove == True )[:,-1]
    idx_keep = tf.where( indices_to_remove == False )[:,-1]
    
    values_remove = tf.gather( logits[0], idx_keep )
    values_keep = tf.gather( logits[0], idx_keep )

    # to create a sparse vector we still need 2d indices like [ [0,1], [0,2], [0,10] ]
    # create vectors of 0's that we'll later stack with the actual indices
    zeros_remove = tf.zeros_like(idx_remove)
    zeros_keep = tf.zeros_like(idx_keep)

    idx_remove = tf.stack( [ zeros_remove, idx_remove], axis=1 )
    idx_keep = tf.stack( [ zeros_keep, idx_keep], axis=1 )
    values_remove = tf.repeat(values_remove, repeats=[2], axis=0)

    # now we can create a sparse matrix
    logits_remove = tf.SparseTensor( idx_remove, values_remove, [1, 3])
    logits_keep = tf.SparseTensor( idx_keep, values_keep, [1, 3])

    # add together the two matrices (need to convert them to dense first)
    filtered_logits = tf.add(
        tf.sparse.to_dense(logits_remove, default_value = 0. ),
        tf.sparse.to_dense(logits_keep, default_value = 0. )
    )

    return filtered_logits
