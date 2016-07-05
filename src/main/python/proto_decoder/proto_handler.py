from generated_proto import pythonRecognitionService_pb2 as rec_proto
import recognition.recognition_manager
from generated_proto import sketch_pb2 as Sketch
from gui import plotter
import os

rec = recognition.recognition_manager.Recognition_manager()
rec.initialize()

def generate_shapes(recognition_template):
    """Returns {GeneratedTemplates}"""
    print "generating shapes form recognition template"
    template_holder = rec_proto.GeneratedTemplates()
    shape_list = rec.generate_shape(recognition_template.shape)

    for shape in shape_list:
        template = Sketch.RecognitionTemplate()
        template.templateId = recognition_template.templateId
        template.shape.CopyFrom(shape)
        template_holder.generatedTemplates.extend([template])
    return template_holder

def train_shape(recognition_template):
    #print recognition_template
    instance = plotter.get_plotter_instance()
    plotter.plot_template(instance, recognition_template)
    path = '../resources/images/' + recognition_template.interpretation.label
    try:
        os.mkdir(path, 0755)
    except OSError:
        #do notbhing
        i = 1
    plotter.save(instance, recognition_template.templateId + '.png', path=path)
    return rec_proto.Noop()
    # return rec.add_training_data(recognition_template.interpretation.label, recognition_template.shape)

def init(labels):
    no_op = rec_proto.Noop()
    rec.set_labels(labels)
    rec.create_classifiers()
    return no_op

def finish_training():
    no_op = rec_proto.Noop()
    rec.finish_training()
    return no_op

def test(recognition_template):
    result = Sketch.RecognitionTemplate()
    result.templateId = recognition_template.templateId
    interpretations = rec.recognize(recognition_template.interpretation.label, recognition_template.shape)
    print interpretations
    for interp in interpretations:
        result.interpretations.extend([interp])

    return result

def message_processor(general_proto):
    request_type = general_proto.requestType
    #print "request type"
    #print request_type
    if request_type == rec_proto.GENERATE_SHAPES:
        return generate_shapes(general_proto.template)
    elif request_type == rec_proto.TRAIN:
        return train_shape(general_proto.template)
    elif request_type == rec_proto.INIT:
        return init(general_proto.labels)
    elif request_type == rec_proto.FINISH_TRAINING:
        return finish_training()
    elif request_type == rec_proto.TEST:
        return test(general_proto.template)
    return rec_proto.Noop()
