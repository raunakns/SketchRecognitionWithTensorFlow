import tensorflow as tf
import numpy as np
import math

tf.reset_default_graph()

X = tf.constant([1, 0])
Y = tf.constant([0, 1])
BOTH = tf.constant([1, 1])
WORKING = tf.constant(1)
PI = tf.constant(math.pi)

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
    x_list, y_list = tf.split(1, 2, points)
    x_stretch, y_stretch = tf.split(0, 2, dim)
    is_stretch_X = tf.equal(x_stretch, WORKING, name="is_stretch_x")
    is_stretch_Y = tf.equal(y_stretch, WORKING, name="is_stretch_Y")
    x_list_stretched = tf.cond(tf.reshape(is_stretch_X, []),
                               create_mult_func(tf, amount, x_list), create_no_op_func(x_list))
    y_list_stretched = tf.cond(tf.reshape(is_stretch_Y, []),
                               create_mult_func(tf, amount, y_list), create_no_op_func(y_list))
    return tf.concat(1, [x_list_stretched, y_list_stretched])


def stretch_around_center(tf, points, dim, amount):
    """Stretches the points around their center"""
    centroid = computeCentroid(tf, points)
    return stretch(tf, points- centroid, dim, amount) + centroid

example_points = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
example_point_list = tf.placeholder(tf.float32)

#result = move_to_center(tf, example_point_list)
#result = rotate_around_center(tf, example_point_list, PI)
result = stretch_around_center(tf, example_point_list, X, tf.constant(2.))

sess = tf.Session()
with tf.Session() as sess:
    # writer = tf.train.SummaryWriter('../../../../log', sess.graph)
    result = sess.run(result, feed_dict={example_point_list: example_points})
    print(result)
# ==> [[ 12.]]
exit(0)
