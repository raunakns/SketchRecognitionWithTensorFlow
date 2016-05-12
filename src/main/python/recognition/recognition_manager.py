import tensorflow as tf
import generate_shapes_graph as shape_generation
from generated_proto import sketch_pb2 as Sketch
from gui import plotter
import sketch_converters

class Recognition_manager:
    def initialize(self):
        self.generation_graph = tf.Graph()
        with self.generation_graph.as_default():
            self.generate_head = tf.placeholder(tf.float32)
            self.generate_result_tensor = shape_generation.generate_shape_graph(tf, self.generate_head)
            self.generation_graph.finalize()

    def generate_shape(self, shape):
        points = self.create_points_from_shape(shape=shape)
        plot = plotter.get_plotter_instance()
        plotter.plot_point_list(plot, points)
        plotter.save(plot, 'before.png')
        resulting_shapes = []
        with tf.Session(graph=self.generation_graph) as session:
            for i in range(0, 10):
                result_tensor = session.run(self.generate_result_tensor, feed_dict={self.generate_head: points})

                plot = plotter.get_plotter_instance()
                plotter.plot_point_list(plot, result_tensor)
                plotter.save(plot, 'after' + str(i) + '.png')

#                resulting_shapes.append(self.split_points_into_shape(self,
#                                                                     points=sketch_converters.convert_array_to_points(result_tensor), shape=shape))
        return resulting_shapes

    def create_points_from_shape(self, shape):
        ''' Creates random point list
            Basically what this does is create a list of points or a list of lists of points based on random values.'''
        def stroke_func(points, stroke):
            return sketch_converters.convert_points_to_array(points), stroke

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
        if srl_object.DESCRIPTOR.name == "SrlShape":
            shape = srl_object
        else:
            object = srl_object.object
            type = srl_object.type
            if type == Sketch.SrlObject.SHAPE:
                shape = Sketch.SrlShape()
                shape.ParseFromString(object)
            elif type == Sketch.SrlObject.STROKE:

                stroke = Sketch.SrlStroke()
                stroke.ParseFromString(object)
                return stroke_func(stroke.points, stroke)

            return shape_func(values_of_results, shape=shape)
        for sub_object in shape.subComponents:
            values_of_results.append(self.call_shape_recursively(stroke_func=stroke_func, shape_func=shape_func,
                                                                 finished_func=finished_func, srl_object=sub_object, top=False))

        if top and finished_func is not None:
            return finished_func(shape_func(values_of_results, shape), shape)
        return shape_func(values_of_results, shape)
