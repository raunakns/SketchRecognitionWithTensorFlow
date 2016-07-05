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

def plot_shape_as_points(plotter_instance, shape):
    points = sketch_utils.create_points_from_shape(shape)
    sketch_utils.strip_ids_from_points(points)
    plot_point_list(plotter_instance, points)

def plot_stroke(plotter_instance, stroke):
    points = sketch_utils.convert_points_to_array(stroke.points, stroke)
    sketch_utils.strip_ids_from_points(points)
    plot_point_list(plotter_instance=plotter_instance, points=points)

def plot_template(plotter_instance, template):
    plot_shape_as_points(plotter_instance, template.shape)

def plot_point_list(plotter_instance, points):
    if len(points) < 4:
        return
    plotter_instance.scatter(*zip(*points))
    #plotter_instance.axis([minX, maxX, minY, maxY])

def save(plotter_instance, file, path=None):
    #plotter_instance.ylim([0,20])
    #plotter_instance.xlim([0,20])
    if path is None:
        path = 'test/'
    plotter_instance.savefig(path + '/' + file)
    plotter_instance.close()

def get_plotter_instance():
    """creates a new plot instance"""
    plt.figure()
    return plt
