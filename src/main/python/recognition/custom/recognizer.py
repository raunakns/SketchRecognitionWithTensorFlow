import math
import uuid
import tensorflow as tf
from recognition import sketch_utils as utils
import numpy as np
from generated_proto import sketch_pb2 as Sketch
import custom_recogntion_utilities as training_helpers

X = 0
Y = 1
ID = 2

class Recognizer:
    num_points = 32
    classifier = None
    training_bundle_features = None
    training_bundle_targets = None
    training_bundle_amount = 1000
    training_bundle_counter = 0
    X_placeholder = None
    Y_placeholder = None
    num_classes = 2
    session = None

    def __init__(self, label):
        self.label = label
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with g.name_scope(label) as scope:
                self.X_placeholder = tf.placeholder(tf.float32, shape=[self.num_points], name="X_values")
                self.Y_placeholder = tf.placeholder(tf.float32, shape=[self.num_points], name="Y_values")
                x = tf.contrib.layers.real_valued_column("X")
                y = tf.contrib.layers.real_valued_column("Y")
                first_layer = tf.contrib.layers.input_from_feature_columns(columns_to_tensors={"X":self.X_placeholder,
                                                                           "Y":self.Y_placeholder},
                                                                           feature_columns=[x, y])
                with g.name_scope('layer2') as scope1:
                    layer2 = tf.contrib.layers.fully_connected(first_layer, self.num_points * 2, scope=scope1)
                with g.name_scope('hidden1') as scope2:
                    hidden = tf.contrib.layers.fully_connected(layer2, self.num_classes, scope=scope2)
                with g.name_scope('hidden2') as scope3:
                    hidden = tf.contrib.layers.fully_connected(hidden, self.num_classes, scope=scope3)
                    print hidden
                    reshaped = tf.reshape(hidden, shape=[1, self.num_points * 2])
                    print reshaped
                with g.name_scope('last_layer') as scope4:
                    unshaped = tf.contrib.layers.fully_connected(reshaped, self.num_classes, scope=scope4)
                    output = tf.reshape(unshaped, shape=[2, 1])
                    print output
                self.output = output
                self.target = training_helpers.create_target_classes(self.num_classes)
                self.loss = training_helpers.create_loss_function(output, self.target)
                self.train_step = training_helpers.create_training(self.loss, .01)

                self.init = tf.initialize_all_variables()
        self.graph.finalize()

    @staticmethod
    def distance(p1, p2):
        dx = p2[X] - p1[X]
        dy = p2[Y] - p1[Y]
        return math.sqrt(dx * dx + dy * dy)

    def path_length(self, points):
        d = 0.0
        for i in range(1, len(points)):
            if points[i][ID] == (points[i - 1][ID]):
                d += self.distance(points[i - 1], points[i])
        return d

    def resample(self, point_list, num_result):
        if len(point_list) == num_result:
            return point_list
        # add ids to list
        path_length = self.path_length(point_list)
        interval_length = path_length / (num_result - 1)  # interval length
        combined_distance = 0.0
        new_points = [point_list[0]]

        i = 1
        while i < len(point_list):
            if point_list[i][ID] == point_list[i - 1][ID]:
                next_point_difference = self.distance(point_list[i - 1], point_list[i])
                if next_point_difference == 0.0:
                    point_list.pop(i)
                    continue
                if (combined_distance + next_point_difference) >= interval_length:
                    qx = point_list[i - 1][X] + ((interval_length - combined_distance) / next_point_difference) * (point_list[i][X] - point_list[i - 1][X])
                    qy = point_list[i - 1][Y] + ((interval_length - combined_distance) / next_point_difference) * (point_list[i][Y] - point_list[i - 1][Y])
                    q = [qx, qy, point_list[i][ID]]
                    new_points.append(q) # append new point 'q'
                    point_list.insert(i, q) # insert 'q' at position i in point_list s.t. 'q' will be the next i
                    combined_distance = 0
                else:
                    combined_distance += next_point_difference
            i += 1
        # sometimes we fall a rounding-error short of
        # adding the last point, so add it if so
        end = len(point_list) - 1
        if len(new_points) == num_result - 1:
            new_points.append([point_list[end][X], point_list[end][Y], point_list[end][ID]])
        return new_points

    def create_features(self, point_list):
        points = self.resample(point_list, self.num_points)
        while len(points) != self.num_points:
            print 'less than 32'
            if len(points) <= self.num_points / 4:
                return None
            points = self.resample(points, self.num_points)
        utils.strip_ids_from_points(points)
        np_points = np.array(points)
        x, y = np.hsplit(np_points, 2)
        #merged_points = np.concatenate((x, y), axis=0)
        reshaped_points = np.reshape(np_points, (2, self.num_points))
        return reshaped_points

    def create_target(self, label):
        # big punishment to show difference between 0 and 1
        true_class = 1.0 if label == self.label else -1.0
        null_class = 1.0 if label != self.label else -1.0
        return [[true_class, null_class]]

    def train(self, label, features):
        target = self.create_target(label)
        if self.training_bundle_features is None:
            self.training_bundle_features = [features]
        else:
            self.training_bundle_features.append(features)

        if self.training_bundle_targets is None:
            self.training_bundle_targets = [target]
        else:
            self.training_bundle_targets.append(target)

        if self.training_bundle_counter >= self.training_bundle_amount:
            self.execute_train_bundle()
        else:
            self.training_bundle_counter += 1

    # TODO: change back to this when the code is fixed
    def single_train(self, label, features):
        target = self.create_target(label)
        self.classifier.fit(x=features, y=target, steps=1)

    def execute_train_bundle(self):
        print 'batch training: ' + self.label

        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init)
            for i in range(self.training_bundle_counter):
                feed = {self.X_placeholder: self.training_bundle_features[i][0],
                        self.Y_placeholder: self.training_bundle_features[i][1],
                        self.target: self.training_bundle_targets[i]}
                sess.run(self.train_step, feed_dict=feed)

        self.training_bundle_features = None
        self.training_bundle_targets = None
        self.training_bundle_counter = 0

    def finish_training(self):
        if self.training_bundle_counter > 0:
            self.execute_train_bundle()

    def recognize(self, features):
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init)
            feed = {self.X_placeholder: features[0],
                    self.Y_placeholder: features[1]}
            result = sess.run(self.output, feed)
            print 'result: ' + self.label
            print result
        # print self.classifier.predict_proba(features)
        interpretation = Sketch.SrlInterpretation()
        interpretation.label = self.label
        interpretation.confidence = float(0.0)
        return interpretation

