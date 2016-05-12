import tensorflow as tf
import generate_shapes_graph as shape_generation
from generated_proto import sketch_pb2 as Sketch

class Recognition_manager:
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
