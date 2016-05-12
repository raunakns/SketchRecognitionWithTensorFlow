import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recognition import sketch_converters

def plot_stroke(plotter_instance, stroke):
    points = sketch_converters.convert_points_to_array(stroke.points)
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
