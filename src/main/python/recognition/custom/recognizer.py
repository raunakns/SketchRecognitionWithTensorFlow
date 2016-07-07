import math
import tensorflow as tf

from recognition import sketch_utils as utils
import custom_recogntion_utilities as training_helpers
from generated_proto import sketch_pb2 as Sketch
from recognition.generation import feature_generation as features

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
                self.points_placeholder = tf.placeholder(tf.float32, shape=[None, 2], name="points")

                feature_list = features.generate_features(self.points_placeholder)
                #feature_list = tf.Print(feature_list, [feature_list], "Features for recognition", summarize=500)
                column_list = features.generate_columns()

                mapping = features.match_features_columns(feature_list, column_list)
                first_layer = tf.contrib.layers.input_from_feature_columns(columns_to_tensors=mapping,
                                                                           feature_columns=column_list)
                with g.name_scope('layer2') as scope1:
                    layer2 = tf.contrib.layers.fully_connected(first_layer, 50, scope=scope1)
                with g.name_scope('hidden1') as scope2:
                    hidden = tf.contrib.layers.fully_connected(layer2, 20, scope=scope2)
                with g.name_scope('hidden2') as scope3:
                    output = tf.contrib.layers.fully_connected(hidden, self.num_classes, scope=scope3)
                    output = tf.sigmoid(output)
                    print output
                self.class_index = tf.argmax(output, 0)
                output = tf.Print(output, [output, self.class_index], "Raw output of training data")
                self.output = output
                self.target = training_helpers.create_target_classes(self.num_classes)
                lossTarget = tf.Print(self.target, [self.target], "Raw target data")
                self.loss = training_helpers.create_loss_function(output, lossTarget)
                self.train_step = training_helpers.create_training(self.loss, .01)

                self.init = tf.initialize_all_variables()
        self.graph.finalize()

    def create_features(self, point_list):
        utils.strip_ids_from_points(point_list)
        return point_list

    def create_target(self, label):
        # big punishment to show difference between 0 and 1
        true_class = 1.0 if label == self.label else 0.0
        null_class = 1.0 if label != self.label else 0.0
        return [[true_class, null_class]]

    def train(self, label, points):
        target = self.create_target(label)
        if self.training_bundle_features is None:
            self.training_bundle_features = [points]
        else:
            self.training_bundle_features.append(points)

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
                feed = {self.points_placeholder: self.training_bundle_features[i],
                        self.target: self.training_bundle_targets[i]}
                result = sess.run(self.train_step, feed_dict=feed)
                print result

        self.training_bundle_features = None
        self.training_bundle_targets = None
        self.training_bundle_counter = 0

    def finish_training(self):
        if self.training_bundle_counter > 0:
            self.execute_train_bundle()

    def recognize(self, features):
        interpretation = Sketch.SrlInterpretation()
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init)
            feed = {self.points_placeholder: features}
            raw_output, class_index = sess.run([self.output, self.class_index], feed)
            print class_index
            print 'result: ' + str(self.label if class_index == 0 else None)
            print raw_output
            interpretation.label = self.label
            interpretation.confidence = raw_output[class_index]
        return interpretation

