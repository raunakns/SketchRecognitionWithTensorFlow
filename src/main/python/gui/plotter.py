import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recognition import sketch_utils

def draw_shape(plotter_instance, shape):
    def stroke_func(points, stroke):
        plot_stroke(plotter_instance, stroke)
        return None
    def shape_func(values, shape):
        return None
    sketch_utils.call_shape_recursively(shape_func=shape_func, stroke_func=stroke_func, srl_object=shape)

def plot_stroke(plotter_instance, stroke):
    points = sketch_utils.convert_points_to_array(stroke.points)
    plot_point_list(plotter_instance=plotter_instance, points=points)

def plot_point_list(plotter_instance, points):
    plotter_instance.scatter(*zip(*points))

def save(plotter_instance, file):
    plotter_instance.ylim([0,20])
    plotter_instance.xlim([0,20])
    plotter_instance.savefig('test/' + file)

def get_plotter_instance():
    """creates a new plot instance"""
    plt.figure()
    return plt
