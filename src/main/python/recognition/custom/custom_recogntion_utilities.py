import tensorflow as tf
def create_training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.
    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def create_target_classes(num_classes):
    return tf.placeholder(tf.float32, [None, num_classes])

def create_loss_function(output, target):
    loss = tf.reduce_sum(-tf.clip_by_value(target, 1e-10, 1.0) * tf.log(tf.clip_by_value(output, 1e-10, 1.0)))
    return tf.Print(loss, [loss], "loss amount")
    #return tf.reduce_mean(-tf.reduce_sum(target * tf.log(output), reduction_indices=[1]))