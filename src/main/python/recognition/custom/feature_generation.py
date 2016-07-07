import tensorflow as tf
from recognition import tensor_flow_utils as tf_utils

def generate_features(points):
    # x and y are both tensors of a single stroke
    centroid = tf_utils.compute_centroid(tf, points)
    centered_points = tf_utils.move_to_center(tf, points, centroid=centroid)
    extreme_points = tf_utils.compute_extremes(tf, centered_points)
    extreme_points = tf.reshape(extreme_points, shape=[2,2])
    print extreme_points
    # these are features!
    x_length = tf.reshape(extreme_points[1, 0] - extreme_points[0, 0], shape=[1])
    y_length = tf.reshape(extreme_points[1, 1] - extreme_points[0, 1], shape=[1])
    # x is 0 y is 1, this is a feature!
    longer_axis = tf.argmax(tf.concat(0, [x_length, y_length]), 0, name="Compute_longer_axis")
    longer_axis_value = tf.maximum(x_length, y_length)
    # this is a feature
    axis_ratio = x_length / y_length
    # scaled to between 0 and 1 by the longer axis
    scaled = centered_points / longer_axis_value

    point_shape = tf.shape(points)
    # this is a feature!
    num_points = tf.reshape(point_shape, shape=[2])[0]

    shaped_num_points = tf.reshape(num_points, shape=[1])

    # this is a feature!
    buckets = create_bucket(scaled, shaped_num_points, point_shape)
    return buckets


def create_bucket(centered_scaled, num_points, point_shape):
    # creates a list of buckets from zero to 100
    max_values = tf.constant(100.0)
    num_buckets = tf.constant(20)
    bucket_size = max_values / tf.to_float(num_buckets)
    half = tf.constant(2.0)
    between_0_and_100 = (centered_scaled * (max_values / half)) + (max_values / half)
    between_0_and_100 = tf.Print(between_0_and_100, [between_0_and_100], "scaled values")
    shaped_buckets = tf.reshape(num_buckets, shape=[1])
    tensor_shape = tf.concat(0, [shaped_buckets, shaped_buckets])
    buckets = tf.zeros(tensor_shape)

    bucketed_tensor = tf.to_int64(tf.round(between_0_and_100 / bucket_size))
    # return bucketed_tensor
    sparser = tf.SparseTensor(bucketed_tensor, tf.ones(num_points), tf.to_int64(tensor_shape))
    ordered_tensor = tf.sparse_reorder(sparser)

    return merge_dupes(ordered_tensor, num_points, point_shape)


def merge_dupes(sparse_tensor, num_points, point_shape):
    print point_shape
    indexes = tf.reshape(sparse_tensor.indices, shape=point_shape)
    indexes = sparse_tensor.indices
    print indexes
    values = sparse_tensor.values
    one = tf.reshape(tf.constant(1), shape=[1])

    #LOOP BELOW
    # for (int i = 0; i < num_points; i++) {

    def create_loop_cond(tf, max):
        def f1(i, ignore1, ignore2):
            return tf.reduce_all(tf.less(i, max))
        return f1

    def create_loop_body(tf):
        def f1(i, indexes, values):
            minus_1 = tf.reshape(tf.add(i, -1), shape=[1])
            minus_1_index = tf.slice(indexes, minus_1, one)
            index = tf.slice(indexes, i, one)

            def equal():
                print 'welp this works!'
                return tf.add(i, 1)

            i = tf.cond(minus_1_index == index, equal, lambda: tf.add(i, 1))
            return i, indexes, values
        return f1

    #}

    print "creating event sequence"
    # points = tf.Print(points, [points], 'original points')
    loop_amount = tf.maximum(1, tf.to_int32(num_points))
    i = tf.constant(1)


    loop_amount = tf.Print(loop_amount, [loop_amount], 'max loop amount')
    loop_cond = create_loop_cond(tf, loop_amount)
    loop_body = create_loop_body(tf)

    result = tf.while_loop(loop_cond, loop_body,
                           [i, indexes, values], back_prop=False)

    return result[1]

