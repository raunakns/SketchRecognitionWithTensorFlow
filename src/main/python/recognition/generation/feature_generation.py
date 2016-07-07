import tensorflow as tf
from recognition import tensor_flow_utils as tf_utils

def get_num_buckets():
    return 20

def merge_features(feature_list):
    return tf.concat(0, feature_list)

def create_axis_values(extreme_points):
    # these are features!
    x_length = tf.reshape(extreme_points[1, 0] - extreme_points[0, 0], shape=[1])
    y_length = tf.reshape(extreme_points[1, 1] - extreme_points[0, 1], shape=[1])
    # x is 0 y is 1, this is a feature!
    longer_axis = tf.argmax(tf.concat(0, [x_length, y_length]), 0, name="Compute_longer_axis")
    axis_ratio = x_length / y_length
    return x_length, y_length, longer_axis, axis_ratio

def generate_columns():
    feature_columns = []
    feature_columns.append(tf.contrib.layers.real_valued_column("x_length"))
    feature_columns.append(tf.contrib.layers.real_valued_column("y_length"))
    feature_columns.append(tf.contrib.layers.real_valued_column("long_axis"))
    feature_columns.append(tf.contrib.layers.real_valued_column("axis_ratio"))
    feature_columns.append(tf.contrib.layers.real_valued_column("num_points"))
    for i in range(get_num_buckets() * get_num_buckets()):
        feature_columns.append(tf.contrib.layers.real_valued_column("bucket" + str(i)))
    return feature_columns

def match_features_columns(features, columns):
    num_features = len(columns)
    map = {}
    for i in range(num_features):
        feature = tf.reshape(features[:, i], shape=[1])
        map[columns[i].name] = feature
    return map

def generate_features(points):
    """feature order:
     x_length
     y_length
     longer_axis
     axis_ratio
     num_points
     buckets
    """
    # x and y are both tensors of a single stroke
    centroid = tf_utils.compute_centroid(tf, points)
    centered_points = tf_utils.move_to_center(tf, points, centroid=centroid)
    extreme_points = tf_utils.compute_extremes(tf, centered_points)
    extreme_points = tf.reshape(extreme_points, shape=[2,2])

    x_length, y_length, longer_axis, axis_ratio = create_axis_values(extreme_points)
    feature_list = merge_features([x_length, y_length,
                                   tf.reshape(tf.to_float(longer_axis), shape=[1]),
                                   axis_ratio])

    longer_axis_value = tf.maximum(x_length, y_length)
    # this is a feature

    # scaled to between 0 and 1 by the longer axis
    scaled = centered_points / longer_axis_value
    point_shape = tf.shape(points)
    # this is a feature!
    num_points = tf.reshape(point_shape, shape=[2])[0]

    shaped_num_points = tf.reshape(num_points, shape=[1])

    feature_list = merge_features([feature_list, tf.to_float(shaped_num_points)])

    # this is a feature!
    buckets = create_bucket(scaled, shaped_num_points, point_shape)
    flat_buckets = tf.reshape(buckets, shape=[get_num_buckets() * get_num_buckets()])
    feature_list = merge_features([feature_list, flat_buckets])
    feature_list_shape = tf.concat(0, [tf.constant(1, shape=[1]), tf.shape(feature_list)])
    reshaped = tf.reshape(feature_list, shape=feature_list_shape)
    return reshaped


def create_bucket(centered_scaled, num_points, point_shape):
    # creates a list of buckets from zero to 100
    max_values = tf.constant(100.0)
    num_buckets = tf.constant(get_num_buckets())
    bucket_size = max_values / tf.to_float(num_buckets)
    half = tf.constant(2.0)
    between_0_and_100 = (centered_scaled * (max_values / half)) + (max_values / half)
    shaped_buckets = tf.reshape(num_buckets, shape=[1])
    tensor_shape = tf.concat(0, [shaped_buckets, shaped_buckets])

    bucketed_tensor = tf.to_int64(tf.round(between_0_and_100 / bucket_size))
    tensor_shape = tf.to_int64(tensor_shape)
    # return bucketed_tensor
    sparser = tf.SparseTensor(bucketed_tensor, tf.ones(num_points), tf.to_int64(tensor_shape))
    ordered_tensor = tf.sparse_reorder(sparser)

    indexes, values = merge_dupes(ordered_tensor, num_points, point_shape)
    tensors = tf.SparseTensor(indexes, values, tensor_shape)
    return tf.sparse_tensor_to_dense(tensors)


def merge_dupes(sparse_tensor, num_points, point_shape):
    indexes = sparse_tensor.indices
    values = sparse_tensor.values
    zero = tf.constant(0, shape=[1])
    one = tf.constant(1, shape=[1])
    two = tf.constant(2, shape=[1])
    row = tf.concat(0, [one, two])

    #LOOP BELOW
    # for (int i = 0; i < num_points; i++) {

    def create_loop_cond(tf):
        def f1(i, ignore1, ignore2, num_points):
            return tf.reduce_all(tf.less(i, num_points))
        return f1

    def create_loop_body(tf):
        def f1(i, indexes, values, num_points):
            i_reshaped = tf.reshape(i, shape=[1])
            i_minus_1 = tf.add(i_reshaped, -1)
            i_slice_minus_1 = tf.concat(0, [i_minus_1, zero])
            i_slice = tf.concat(0, [i_reshaped, zero])
            minus_1_index = tf.slice(indexes, i_slice_minus_1, row)
            index = tf.slice(indexes, i_slice, row)

            def equal():
                value_1 = tf.slice(values, i_minus_1, one)
                value_2 = tf.slice(values, i_reshaped, one)
                merged_value = value_1 + value_2
                one_more = tf.add(i_reshaped, 1)

                first_half_values = tf.slice(values, zero, i_minus_1)
                second_half_values = tf.slice(values, one_more, num_points - one_more)
                merged_values = tf.concat(0, [first_half_values, merged_value, second_half_values])


                first_half_indices = tf.slice(indexes,
                                              tf.concat(0, [zero, zero]),
                                              tf.concat(0, [i_minus_1, two]))
                second_half_indices = tf.slice(indexes,
                                               tf.concat(0, [i_reshaped, zero]),
                                               tf.concat(0, [num_points - i_reshaped, two]))
                merged_indexes = tf.concat(0, [first_half_indices, second_half_indices])
                return i, merged_indexes, merged_values, tf.add(num_points, -1)

            def not_equal():
                return tf.add(i, 1), indexes, values, num_points

            condition = tf.reduce_all(tf.equal(minus_1_index, index))
            i, indexes, values, num_points = tf.cond(condition, equal, not_equal)
            return i, indexes, values, num_points
        return f1

    #}

    # points = tf.Print(points, [points], 'original points')
    i = tf.constant(1)

    loop_cond = create_loop_cond(tf)
    loop_body = create_loop_body(tf)

    result = tf.while_loop(loop_cond, loop_body,
                           [i, indexes, values, num_points], back_prop=False)

    return result[1], result[2]

