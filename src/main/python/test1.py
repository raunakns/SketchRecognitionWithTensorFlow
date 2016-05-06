import math

import numpy as np
import tensorflow as tf

import generate_shapes_graph

tf.reset_default_graph()

X = tf.constant([1, 0])
Y = tf.constant([0, 1])
BOTH = tf.constant([1, 1])
WORKING = tf.constant(1)
PI = tf.constant(math.pi)

example_points = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
example_point_list = tf.placeholder(tf.float32)

#result = move_to_center(tf, example_point_list)
#result = rotate_around_center(tf, example_point_list, PI)
#result = sketch_utils.stretch_around_center(tf, example_point_list, Y, tf.constant(2.))
result = generate_shapes_graph.generate_shape_graph(tf, example_point_list)

sess = tf.Session()
with tf.Session() as sess:
    writer = tf.train.SummaryWriter('../../../../log', sess.graph)
    result = sess.run(result, feed_dict={example_point_list: example_points})
    print(result)
# ==> [[ 12.]]
exit(0)
