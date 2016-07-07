import math

import numpy as np
import tensorflow as tf

from recognition.custom import feature_generation as features
from gui import plotter
from recognition import tensor_flow_utils

tf.reset_default_graph()

X = tf.constant([1, 0])
Y = tf.constant([0, 1])
BOTH = tf.constant([1, 1])
WORKING = tf.constant(1)
PI = tf.constant(math.pi)
example_points = np.array([[10, 10], [9, 10], [8, 10], [7, 10],
                           [7, 9],[7, 8],[7, 7],[8, 7],
                           [9, 7],[10, 7],[10, 6],[10, 5],
                           [10, 4],[9, 4],[8, 4],[7, 4], [7, 4.1]], dtype=np.float32)
example_point_list = tf.placeholder(tf.float32)

#result = tensor_flow_utils.move_to_center(tf, example_point_list)
#result = rotate_around_center(tf, example_point_list, PI)
#result = sketch_utils.stretch_around_center(tf, example_point_list, Y, tf.constant(2.))
#result = generate_shapes_graph.generate_shape_graph(tf, example_point_list)
#result = tensor_flow_utils.shear_around_center(tf, example_point_list, Y, tf.constant(-0.7))
#result = tensor_flow_utils.compute_extremes(tf, result)
result = features.generate_features(example_point_list)

sess = tf.Session()
with tf.Session() as sess:
    print "starting points"
    print example_points
    plt = plotter.get_plotter_instance()
    plotter.plot_point_list(plt, example_points)
    plotter.save(plt, 'pre_test.png')
    writer = tf.train.SummaryWriter('../../../../log', sess.graph)
    result = sess.run(result, feed_dict={example_point_list: example_points})
    print("printing result:")
    print(result)

    plt = plotter.get_plotter_instance()
    plotter.plot_point_list(plt, result)
    plotter.save(plt, 'post_test.png')
    plt = plotter.get_plotter_instance()
# ==> [[ 12.]]
