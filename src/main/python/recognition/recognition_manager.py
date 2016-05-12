import tensorflow as tf
import generate_shapes_graph as shape_generation
from generated_proto import sketch_pb2 as Sketch
from gui import plotter
import sketch_utils

class Recognition_manager:
    def initialize(self):
        self.generation_graph = tf.Graph()
        with self.generation_graph.as_default():
            self.generate_head = tf.placeholder(tf.float32)
            self.generate_result_tensor = shape_generation.generate_shape_graph(tf, self.generate_head)
            self.generation_graph.finalize()

    def generate_shape(self, shape):
        points = self.create_points_from_shape(shape=shape)
        resulting_shapes = []
        with tf.Session(graph=self.generation_graph) as session:
            for i in range(0, 10):
                result_tensor = session.run(self.generate_result_tensor, feed_dict={self.generate_head: points})

                resulting_shapes.append(self.split_points_into_shape(point_list=sketch_utils.convert_array_to_points(result_tensor),
                                                                     template_shape=shape))
        return resulting_shapes

    def create_points_from_shape(self, shape):
        ''' Creates random point list
            Basically what this does is create a list of points or a list of lists of points based on random values.'''
        def stroke_func(points, stroke):
            return sketch_utils.convert_points_to_array(points), stroke

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
                    srl_object.type = Sketch.SrlObject.SHAPE
                else:
                    srl_object.type = Sketch.SrlObject.STROKE
                srl_object.object = object.SerializeToString()
                new_shape.subComponents.extend([srl_object])
            return new_shape

        return sketch_utils.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func, srl_object=template_shape)
