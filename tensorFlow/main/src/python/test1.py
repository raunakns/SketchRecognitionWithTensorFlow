import tensorflow as tf
import numpy as np
import math

def computeCentroid(tf, points):
    """Computes the centroid of the points."""
    return tf.reduce_mean(points, 0)

def move_to_center(tf, points):
    return points - computeCentroid(tf, points, name="compute center")

def rotate(tf, points, theta):
    rotation_matrix = [[tf.cos(theta), -tf.sin(theta)],
                       [tf.sin(theta), tf.cos(theta)]]
    return tf.matmul(points, rotation_matrix, name="rotate matrices")

def rotate_around_center(tf, points, theta):
    centroid = computeCentroid(tf, points)
    return rotate(tf, (points - centroid), theta) + centroid

tf.reset_default_graph()
example_points = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
example_point_list = tf.placeholder(tf.float32)
PI = tf.constant(math.pi)
# centroid = tf.reduce_mean(example_point_list, 0)
result = rotate_around_center(tf, example_point_list, PI)

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[1., 1.],
                       [2., 2.],
                       [3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])
theta = tf.placeholder(tf.float32, shape=(1, 1))

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.

# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    writer = tf.train.SummaryWriter('./log', sess.graph)
    result = sess.run(result, feed_dict={example_point_list: example_points})
    print(result)
# ==> [[ 12.]]
exit(0)
