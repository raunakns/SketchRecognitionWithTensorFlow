import tensorflow as tf
import generate_shapes_graph as shape_generation
from generated_proto import sketch_pb2 as Sketch
import random


class Recognition_manager():
    def initialize(self):
        self.generation_graph = tf.Graph()
        with self.generation_graph.as_default():
            self.generate_head = tf.placeholder("float", [2, None])
            self.generate_result_tensor = shape_generation.generate_shape_graph(tf, self.generate_head)
            self.generation_graph.finalize()

    def generate_shape(self, shape):
        points = self.create_points_from_shape(shape=shape)
        resulting_shapes = []
        with tf.Session.__init__(graph=self.generation_graph) as session:
            func = self.recursive_generate_shape_caller(session)
            self.recursive_object_caller(shape, func)
            for i in range(0, 10):
                result_tensor = session.run(self.generate_result_tensor, feed_dict={self.generate_head: points})
                resulting_shapes.append(self.split_into_shape(self, points=self.convert_array_to_points(result_tensor), shape=shape))
        return resulting_shapes

    def recursive_object_caller(self, srl_object, func, call_as_strokes=True, top=True):
        """ Recursively calls a function on each shape using DFS
            func is a function that is only called at the lowest shape level.
            func is called with: list of points, shape/stroke it is being called on.
            func should return a list of points.
            return (result of func, shapeIt was called on)

            if call_as_strokes is true then each call is expected to return a list of strokes
            EXAMPLE:
                Shape1: [shape2, stroke1, stroke2]
                Shape2: [shape3, shape4, stroke3]
                Shape3: [stroke4, stroke5]
                Shape4: [stroke6]

            IGNORED FOR NOW
            Call Order (if call as strokes is false):
                Shape3 (stroke4, stroke5)
                Shape4 (stroke6)
                Shape2 (shape3, shape4, stroke3)
                Shape1 (shape2, stroke1, stroke2)

            IMPLEMENTED
            Call order (if call as strokes is true):
                Shape3 (stroke4) returns str4
                Shape3 (stroke5) returns str5
                Shape3 (str4 + str 5) returns str45
                Shape4 (stroke6) return (str6)
                Shape2 (stroke3) returns str3
                Shape2 (str45 + str6 + stroke3) return str[45]63
                Shape1 (stroke1) returns str1
                Shape1 (stroke2) returns str2
                Shape1 (str1 + str2 + str[45]63) return str12[[45]63]
               """
        # this is a list of a list of SrlPoint coordinates
        list_of_point_lists = []
        shape = None
        if srl_object.HasField('subComponents'):
            shape = srl_object
        else:
            object = srl_object.object
            type = srl_object.type
            if type == Sketch.SrlObject.SHAPE:
                shape = Sketch.SrlShape.parseFromString(object)
            elif type == Sketch.SrlObject.STROKE:
                stroke = Sketch.SrlStroke.parseFromString(object)
                return func(stroke.points, stroke)
        '''
        for subObject in shape.subComponents:
            list_of_point_lists.append(self.recursive_object_caller(subObject, func, call_as_strokes), top=False)

        merged_list = []
        for points in list_of_point_lists:
            merged_list.extend(points)
        point_result = func(merged_list, shape)

        if not top:
            return point_result
        else:
            return self.decode_points(point_result, srl_object)
        '''
        return None

    def create_points_from_shape(self, shape):
        ''' Creates random point list
            Basically what this does is create a list of points or a list of lists of points based on random values.'''
        def stroke_func(points, stroke):
            return self.convert_points_to_array(points), stroke

        def shape_func(sub_calls_results, shape):
            """converts list_o_points into a list of lists of points potentially... randomly merging points too"""
            result = []
            #for now lets just merge them all!

            for object in sub_calls_results:
                result.extend(object[0])
            return result, shape

        return self.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func, srl_object=shape)[0]

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
            for object in sub_objects:
                srl_object = Sketch.SrlObject()
                if object.HasField('subComponents'):
                    srl_object.type = Sketch.SrlObject.SHAPE
                else:
                    srl_object.type = Sketch.SrlObject.STROKE
                srl_object.object = object.SerializeToString()
                new_shape.subComponents.add(srl_object)
            return new_shape

        return self.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func, srl_object=template_shape)


    def call_shape_recursively(self, srl_object, stroke_func, shape_func, finished_func=None, top=True):
        ''' calls the objects recursively and calls stroke_func on strokes and then calls a shape_func on the list of results and the shape'''
        values_of_results = []
        shape = None
        if srl_object.HasField('subComponents'):
            shape = srl_object
        else:
            object = srl_object.object
            type = srl_object.type
            if type == Sketch.SrlObject.SHAPE:
                shape = Sketch.SrlShape.parseFromString(object)
            elif type == Sketch.SrlObject.STROKE:
                stroke = Sketch.SrlStroke.parseFromString(object)
                return stroke_func(stroke.points, stroke)

        for subObject in shape.subComponents:
            values_of_results.append(self.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func,
                                                                 finished_func=finished_func, srl_object=srl_object, top=False))

        if top:
            return finished_func(shape_func(values_of_results, shape), shape)
        return shape_func(values_of_results, shape)

    def convert_array_to_points(self, points):
        result = []
        for row in points:
            new_point = Sketch.SrlPoint()
            new_point.x = row[0]
            new_point.y = row[1]
            new_point.time = -1
            result.append(new_point)
        return result

    def convert_points_to_array(self, points):
        result = []
        for point in points:
            result = [point.x, point.y]
        return result

    def split_into_shape(self, points, shape):
        """recursively makes the shape from the given points"""
