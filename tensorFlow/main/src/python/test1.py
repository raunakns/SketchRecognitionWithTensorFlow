import tensorflow as tf
import numpy as np
import math

X = tf.constant([1, 0])
Y = tf.constant([0, 1])
BOTH = tf.constant([1, 1])
WORKING = tf.constant([1])

def computeCentroid(tf, points):
    """Computes the centroid of the points."""
    return tf.reduce_mean(points, 0)

def move_to_center(tf, points):
    return points - computeCentroid(tf, points, name="compute center")

def rotate(tf, points, theta):
    top = tf.pack([tf.cos(theta), -tf.sin(theta)])
    bottom = tf.pack([tf.sin(theta), tf.cos(theta)])
    rotation_matrix = tf.pack([top, bottom])
    return tf.matmul(points, rotation_matrix, name="rotate_matrices")

def rotate_around_center(tf, points, theta):
    centroid = computeCentroid(tf, points)
    return rotate(tf, (points - centroid), theta) + centroid

def create_mult_func(tf, amount, list):
    def f1():
        return tf.scalar_mul(amount, list)
    return f1

def create_no_op_func(tensor):
    def f1():
        return tensor
    return f1

def stretch(tf, points, dim, amount):
    """points is a 2 by ??? tensor, dim is a 1 by 2 tensor, amount is tensor scalor"""
    x_list, y_list = tf.split(0, 2, points)
    x_stretch, y_stretch = tf.split(0, 2, dim)
    #x_list_stretched = tf.cond(tf.equal(x_stretch, WORKING, name="is_stretch_x"),
    #                           create_mult_func(tf, amount, x_list), create_no_op_func(x_list))
    #y_list_stretched = tf.cond(tf.equal(y_stretch, WORKING, name="is_stretch_Y"),
    #
    return tf.pack(x_list_stretched, y_list_stretched)


def stretch_around_center(tf, points, dim, amount):
    """Stretches the points around their center"""
    centroid = computeCentroid(tf, points)
    return stretch(tf, points - centroid, dim, amount) + centroid
tf.reset_default_graph()
example_points = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
example_point_list = tf.placeholder(tf.float32)
PI = tf.constant(math.pi)

# centroid = tf.reduce_mean(example_point_list, 0)

#result = rotate_around_center(tf, example_point_list, PI)
result = stretch_around_center(tf, example_point_list, X, 1)

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
summary_op = tf.merge_all_summaries()
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
    writer = tf.train.SummaryWriter('../../../../log', sess.graph)
    result = sess.run(result, feed_dict={example_point_list: example_points})
    print(result)
# ==> [[ 12.]]
exit(0)
