import math

import tensor_flow_utils

#########################
#START ACTIONS
#########################

# SHEAR
def create_shear_func(tf, points, shear_type, amount):
    def shear_action():
        print "creating shear action"
        return tensor_flow_utils.shear_around_center(tf, points, shear_type, amount)
    return shear_action

def create_shear_action(tf):
    def shear_callback(points):
        shear_type = tf.random_uniform([2, 1], minval=0, maxval=2, dtype=tf.int32, seed=None, name="shear_type")
        shear_amount = tf.reshape(tf.random_normal([1], stddev=.4, name="shear_amount"), [])
        # shear_type = tf.Print(shear_type, [shear_type, shear_amount], "shear action")
        return create_shear_func(tf, points, shear_type, shear_amount)
    return shear_callback

# ROTATE
def create_rotate_func(tf, points, theta):
    def rotate_action():
        print "creating stretch action"
        return tensor_flow_utils.rotate_around_center(tf, points, theta)
    return rotate_action

def create_rotate_action(tf):
    def rotate_callback(points):
        theta1 = tf.random_uniform([1], minval=-2 * math.pi, maxval=2 * math.pi, name="rotate_amount_un")
        theta2 = tf.random_normal([1], mean=0,  stddev=math.pi/4, name="rotate_amount_norm")

        #theta2 = tf.Print(theta2, [theta2], "theta norm")
        #theta1 = tf.Print(theta1, [theta1], "theta un")
        theta = tf.reduce_mean(tf.concat(0, [theta1, theta2]))
        #theta = tf.Print(theta, [theta], "rotate action")
        return create_rotate_func(tf, points, theta)
    return rotate_callback

# STRETCH
def create_stretch_func(tf, points, stretch_type, amount):
    def stretch_action():
        print "creating stretch action"
        return tensor_flow_utils.stretch_around_center(tf, points, stretch_type, amount)
    return stretch_action

def create_stretch_action(tf):
    def stretch_callback(points):
        stretch_type = tf.random_uniform([2, 1], minval=0, maxval=2, dtype=tf.int32, seed=None, name="stretch_type")
        stretch_amount = tf.maximum(tf.constant(.3), tf.reshape(tf.random_normal([1], mean=1, stddev=.8, dtype=tf.float32,
                                                     name="stretch_amount"),[]))
        # stretch_type = tf.Print(stretch_type, [stretch_type, stretch_amount], "stretch action")
        return create_stretch_func(tf, points, stretch_type, stretch_amount)
    return stretch_callback

#########################
#END ACTIONS
#########################

#CASE
def create_case_statement(tf, statementList):
    '''Creates a function that when called creates a list of case statements'''
    caseDict = {}
    def f1(actionCase, points):
        print "creating switch statement"
        for i in range(len(statementList)):
            create_case(tf, caseDict, i, actionCase, statementList[i](points))
        return tf.case(caseDict, tensor_flow_utils.create_no_op_func(points), name="switch_statement")
    return f1

def create_case(tf, caseDict, caseNumber, dynamicNumber, caseFunc):
    '''assigns a specific case statment with a comparison to a specific value'''
    print 'creating case statement: ' + caseFunc.__name__
    caseDict[tf.equal(tf.constant(caseNumber), dynamicNumber)] = caseFunc

#LOOP
def create_loop_cond(tf, max):
    def f1(i, ignore):
        return tf.reduce_all(tf.less(i, max))
    return f1

def create_loop_body(tf, case_statement, number_of_case_statments):
    def f1(i, points):
        print "building for loop body"
        actionCase = tf.reshape(tf.random_uniform([1], minval = 0, maxval=number_of_case_statments + 2, name="case"), [])
        actionCase = tf.to_int32(tf.floor(actionCase))
        # i = tf.Print(i, [i], 'loop index')
        # i = tf.Print(i, [actionCase], 'action case')
        points = case_statement(actionCase, points)
        return tf.add(i, 1), points
    return f1

# EVENT SEQUENCE
def create_event_sequence(tf, points):
    print "creating event sequence"
    # points = tf.Print(points, [points], 'original points')
    loop_amount_random = tf.reshape(tf.random_normal([1], mean=2, stddev=1.5, dtype=tf.float32), [])
    loop_amount = tf.maximum(1, tf.to_int32(tf.round(loop_amount_random)))
    i = tf.constant(0)

    case_statement_list = [create_rotate_action(tf),
                           create_stretch_action(tf),
                           create_shear_action(tf),
                           create_shear_action(tf)]
    switch_statement = create_case_statement(tf,
                                             case_statement_list)

    loop_amount = tf.Print(loop_amount, [loop_amount], 'max loop amount')
    loop_cond = create_loop_cond(tf, loop_amount)
    loop_body = create_loop_body(tf, switch_statement, len(case_statement_list))

    result = tf.while_loop(loop_cond, loop_body,
                         [i, points], parallel_iterations=1, back_prop = False)
    return result[1]



def generate_shape_graph(tf, points):
    return create_event_sequence(tf, points)
