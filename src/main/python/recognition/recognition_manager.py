import tensorflow as tf
from generated_proto import sketch_pb2 as Sketch

import sketch_utils
from generation import generate_shapes_graph as shape_generation
from simple import recognizer

class Recognition_manager:

    classifier_list = {}
    label_list = []

    def initialize(self):
        self.generation_graph = tf.Graph()
        self.recognition_graph = tf.Graph()
        with self.generation_graph.as_default():
            self.generate_head = tf.placeholder(tf.float32)
            self.generate_result_tensor = shape_generation.generate_shape_graph(tf, self.generate_head)
            self.generation_graph.finalize()
        self.recognizers = {}

    def generate_shape(self, shape):
        points = self.create_points_from_shape(shape=shape)
        sketch_utils.strip_ids_from_points(points)
        resulting_shapes = []
        print "executing generated graph"
        with tf.Session(graph=self.generation_graph) as session:
            for i in range(0, 10):
               # print "running new shape generation graph"

               # print "input"
               # print points
                result_tensor = session.run(self.generate_result_tensor, feed_dict={self.generate_head: points})
               # print "tensor result"
               # print(result_tensor)

                result_points = sketch_utils.convert_array_to_points(result_tensor)
               # print "point result"
               # print result_points
                resulting_shapes.append(self.split_points_into_shape(point_list=result_points,
                                                                     template_shape=shape))
        return resulting_shapes

    def create_points_from_shape(self, shape):
        ''' Creates random point list
            Basically what this does is create a list of points or a list of lists of points based on random values.'''
        def stroke_func(points, stroke):
            return sketch_utils.convert_points_to_array(points, stroke), stroke

        def shape_func(sub_calls_results, shape):
            """converts list_o_points into a list of lists of points potentially... randomly merging points too"""
            result = []
            #for now lets just merge them all!

            for object in sub_calls_results:
                result.extend(object[0])
            return result, shape

        return sketch_utils.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func, srl_object=shape)[0]

    def split_points_into_shape(self, point_list, template_shape):
        current_index = [0]
        def stroke_func(points, stroke):
            new_points = point_list[current_index[0]: current_index[0] + len(points)]
            new_stroke = Sketch.SrlStroke()
            new_stroke.id = 'newId' + stroke.id
            new_stroke.time = stroke.time
            new_stroke.points.extend(new_points)
            current_index[0] += len(points)
            return new_stroke

        def shape_func(sub_objects, shape):
            new_shape = Sketch.SrlShape()
            new_shape.id = 'newId' + shape.id
            new_shape.time = 0
            for object in sub_objects:
                srl_object = Sketch.SrlObject()
                if srl_object.DESCRIPTOR.name == "SrlShape":
                    srl_object.type = Sketch.SHAPE
                else:
                    srl_object.type = Sketch.STROKE
                srl_object.object = object.SerializeToString()
                new_shape.subComponents.extend([srl_object])
            return new_shape

        return sketch_utils.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func, srl_object=template_shape)

    def recognize(self, label, shape):
        points = self.create_points_from_shape(shape)
        if len(points) < 10:
            return []
        if self.recognizers.get(label) is None:
            self.recognizers[label] = recognizer.Recognizer(label)
        features = self.recognizers[label].create_features(points)
        resultList = []
        for rec in self.label_list:
            resultList.append(self.recognizers[rec].recognize(label, features))
        print resultList
        return resultList

    def set_labels(self, label_list):
        self.label_list = label_list

    def create_classifiers(self):
        self.classifier_list = {}
        for label in self.label_list:
            print 'Creating classifier' + label
            self.recognizers[label] = recognizer.Recognizer(label)

    def add_training_data(self, label, shape):
        points = self.create_points_from_shape(shape)
        if len(points) < 10:
            return
        if self.recognizers.get(label) is None:
            self.recognizers[label] = recognizer.Recognizer(label)
        features = self.recognizers[label].create_features(points)
        for rec in self.label_list:
            print "training label " + rec
            self.recognizers[rec].train(label, features)
