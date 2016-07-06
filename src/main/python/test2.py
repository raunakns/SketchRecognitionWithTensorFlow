import uuid

import recognition.recognition_manager
from generated_proto import sketch_pb2 as Sketch

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

print 'shape created starting recognition'
rec = recognition.recognition_manager.Recognition_manager()

print 'initializing recognition'
rec.initialize()

print 'creating recognition classifiers'
rec.set_labels(['S', 'X'])
rec.create_classifiers()

print 'training'
for i in range(0, 3000):
    rec.add_training_data('S', shape)
rec.finish_training()

print 'recognizing'
print rec.recognize('S', shape)

print 'done'