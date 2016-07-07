import uuid

from generated_proto import sketch_pb2 as Sketch

from gui import plotter
from proto_decoder import proto_handler

def make_point(stroke, x, y):
    point1 = Sketch.SrlPoint()
    point1.id = str(uuid.uuid1())
    point1.x = x
    point1.y = y
    point1.time = 0
    stroke.points.extend([point1])

template = Sketch.RecognitionTemplate()

shape = Sketch.SrlShape()
shape.id = str(uuid.uuid1())

stroke = Sketch.SrlStroke()
stroke.id = str(uuid.uuid1())
stroke.time = 0

make_point(stroke, 10, 10)
make_point(stroke, 9, 10)
make_point(stroke, 8, 10)
make_point(stroke, 7, 10)
make_point(stroke, 7, 9)
make_point(stroke, 7, 8)
make_point(stroke, 7, 7)
make_point(stroke, 8, 7)
make_point(stroke, 9, 7)
make_point(stroke, 10, 7)
make_point(stroke, 10, 6)
make_point(stroke, 10, 5)
make_point(stroke, 10, 4)
make_point(stroke, 9, 4)
make_point(stroke, 8, 4)
make_point(stroke, 7, 4)

srl_obj = Sketch.SrlObject()
srl_obj.object = stroke.SerializeToString()
srl_obj.type = Sketch.STROKE

shape.subComponents.extend([srl_obj])

plt = plotter.get_plotter_instance()
plotter.draw_shape(plt, shape=shape)
plotter.save(plt, 'before.png')

template.shape.CopyFrom(shape)
templateList = proto_handler.generate_shapes(template)
i = 0
for template in templateList.generatedTemplates:
    plt = plotter.get_plotter_instance()
    plotter.draw_shape(plt, shape=template.shape)
    plotter.save(plt, 'after' + str(i) + '.png')
    i += 1
