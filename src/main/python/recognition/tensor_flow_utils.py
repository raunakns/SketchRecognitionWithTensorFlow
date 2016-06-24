
def compute_centroid(tf, points):
    # points {[2, ?]} tensor of float_64
    """Computes the centroid of the points."""
    return tf.reduce_mean(points, 0)

def move_to_center(tf, points):
    # points {[2, ?]} tensor of float_64
    "moves the list of points to the origin"
    return points - compute_centroid(tf, points, name="compute center")

def rotate(tf, points, theta):
    # points {[2, ?]} tensor of float_64
    # theta a tensor containing an angle
    top = tf.pack([tf.cos(theta), -tf.sin(theta)])
    bottom = tf.pack([tf.sin(theta), tf.cos(theta)])
    rotation_matrix = tf.pack([top, bottom])
    return tf.matmul(points, rotation_matrix, name="rotate_matrices")

def rotate_around_center(tf, points, theta):
    centroid = compute_centroid(tf, points)
    return rotate(tf, (points - centroid), theta) + centroid

def create_mult_func(tf, amount, list):
    def f1():
        return tf.scalar_mul(amount, list)
    return f1

def create_no_op_func(tensor):
    def f1():
        return tensor
    return f1


def callForEachDimension(tf, points, dim, callback):
    """dim is which dimenision is active (currently only 2 x, y)
       callback is a function that takes in the list of points and returns function that returns a list of points"""
    yes = tf.constant(1)
    x_list, y_list = tf.split(1, 2, points)
    x_dim, y_dim = tf.split(0, 2, dim)
    is_dim_X = tf.reduce_all(tf.equal(x_dim, yes, name="is_dim_x"))
    is_dim_Y = tf.reduce_all(tf.equal(y_dim, yes, name="is_dim_Y"))
    x_list_dimed = tf.cond(is_dim_X, callback(tf, x_list, 0), create_no_op_func(x_list))
    y_list_dimed = tf.cond(is_dim_Y, callback(tf, y_list, 1), create_no_op_func(y_list))
    return tf.concat(1, [x_list_dimed, y_list_dimed])

def create_stretch(amount):
    def f1(tf, points, dim_called):
        return create_mult_func(tf, amount, points)
    return f1

def stretch(tf, points, dim, amount):
    """points is a 2 by ??? tensor, dim is a 1 by 2 tensor, amount is tensor scalor"""
    return callForEachDimension(tf, points, dim, create_stretch(amount))

def stretch_around_center(tf, points, dim, amount):
    """Stretches the points around their center"""
    centroid = compute_centroid(tf, points)
    return stretch(tf, points- centroid, dim, amount) + centroid

def flip(tf, points, mirror_point, flip_type):
    """points is a 2 by ??? tensor, dim is a 1 by 2 tensor, amount is tensor scalor"""
    temp_point = mirror_point - points
    return points + (2 * temp_point)

def compute_flip_point(tf, centroid):
    two = tf.constant(2.)
    x, y = tf.split(0, 2, centroid)
    x_rand = tf.random_normal([], mean=x, stddev=tf.maximum(two, (x - tf.sqrt(x))))
    y_rand = tf.random_normal([], mean=y, stddev=tf.maximum(two, (y - tf.sqrt(y))))
    return tf.concat(0, [x_rand, y_rand])

def flip_around_center(tf, points, flip_type):
    """flip the points around their center"""
    centroid = compute_centroid(tf, points)
    flipPoint = compute_flip_point(tf, centroid)
    flipped_points = flip(tf, points, flipPoint, flip_type)
    flipped_center = compute_centroid(tf, flipped_points)
    offset = centroid - flipped_center
    return flipped_points + offset
    #return offset

def create_shear_array(original_tensor, amount):
    def set_shear(tf, shear_line, dim_called):
        indices = [[1 - dim_called, 0]]  # the of coordinate to update.

        values = tf.reshape(amount, [1])  # A list of values corresponding to the respective
        # coordinate in indices.

        def f1():
            shape = tf.shape(shear_line) # The shape of the corresponding dense tensor, same as `c`.
            shape = tf.to_int64(shape)
            delta = tf.SparseTensor(indices, values, shape)
            denseDelta = tf.sparse_tensor_to_dense(delta)
            combined_shear = denseDelta + shear_line
            return combined_shear
        return f1
    return set_shear

def shear_around_center(tf, points, flip_type, amount):
    """shear the points around their center"""
    centroid = compute_centroid(tf, points)
    shear = tf.constant([[1., 0.], [0., 1.]])

    shear = callForEachDimension(tf, shear, flip_type, create_shear_array(shear, amount))
    shear = tf.Print(shear, [shear], "resulting shear array")
    result = tf.matmul(points, shear)
    sheared_center = compute_centroid(tf, result)
    offset = centroid - sheared_center
    return result + offset
    return offset