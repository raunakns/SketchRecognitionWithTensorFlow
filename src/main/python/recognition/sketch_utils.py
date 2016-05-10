
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

def stretch(tf, points, dim, amount):
    """points is a 2 by ??? tensor, dim is a 1 by 2 tensor, amount is tensor scalor"""
    yes = tf.constant(1)
    x_list, y_list = tf.split(1, 2, points)
    x_stretch, y_stretch = tf.split(0, 2, dim)
    is_stretch_X = tf.reduce_all(tf.equal(x_stretch, yes, name="is_stretch_x"))
    is_stretch_Y = tf.reduce_all(tf.equal(y_stretch, yes, name="is_stretch_Y"))
    x_list_stretched = tf.cond(is_stretch_X,
                               create_mult_func(tf, amount, x_list), create_no_op_func(x_list))
    y_list_stretched = tf.cond(is_stretch_Y,
                               create_mult_func(tf, amount, y_list), create_no_op_func(y_list))
    return tf.concat(1, [x_list_stretched, y_list_stretched])

def stretch_around_center(tf, points, dim, amount):
    """Stretches the points around their center"""
    centroid = compute_centroid(tf, points)
    return stretch(tf, points- centroid, dim, amount) + centroid
