import math

import sketch_utils


def create_stretch_func(tf, points, stretch_type, amount):
    def f1():
        return sketch_utils.stretch_around_center(tf, points, stretch_type, amount)
    return f1

def create_rotate_func(tf, points, theta):
    def f1():
        return sketch_utils.rotate_around_center(tf, points, theta)
    return f1

def generate_shape_graph(tf, points):
    true = tf.constant(1)
    stretch_amount = tf.reshape(tf.random_uniform([1], minval = 0.2, maxval=5, name="stretch_amount"), [])
    theta = tf.reshape(tf.random_uniform([1], minval=-2 * math.pi, maxval=2 * math.pi, name="rotate_amount"), [])
    stretch_type = tf.random_uniform([2, 1], minval=0, maxval=2, dtype=tf.int32, seed=None, name="stretch_type")
    is_stretch = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name="is_stretch")
    is_rotate = tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name="is_rotate")

    print tf.shape(is_stretch)

    stretched = tf.cond(tf.reduce_all(tf.equal(is_stretch, true)),
                        create_stretch_func(tf, points, stretch_type, stretch_amount),
                        sketch_utils.create_no_op_func(points))
    rotated = tf.cond(tf.reduce_all(tf.equal(is_rotate, true)),
                      create_rotate_func(tf, stretched, theta),
                      sketch_utils.create_no_op_func(stretched))

    return rotated
