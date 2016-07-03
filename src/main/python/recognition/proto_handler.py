from generated_proto import pythonRecognitionService_pb2 as rec_proto
import recognition_manager
from generated_proto import sketch_pb2 as Sketch

rec = recognition_manager.Recognition_manager()
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
    no_op = rec_proto.Noop()
    rec.add_training_data(recognition_template.interpretation.label, recognition_template.shape)
    return no_op

def init(labels):
    no_op = rec_proto.Noop()
    rec.set_labels(labels)
    rec.create_classifiers()
    return no_op

def test(recognition_template):
    result = Sketch.RecognitionTemplate()
    result.templateId = recognition_template.templateId
    interpretations = rec.recognize(recognition_template.interpretation.label, recognition_template.shape)
    for interp in interpretations:
        result.interpretations.extend([interp])

    return result

def message_processor(general_proto):
    request_type = general_proto.requestType
    print "request type"
    print request_type
    if request_type == rec_proto.GENERATE_SHAPES:
        return generate_shapes(general_proto.template)
    elif request_type == rec_proto.TRAIN:
        return train_shape(general_proto.template)
    elif request_type == rec_proto.INIT:
        return init(general_proto.labels)
    elif request_type == rec_proto.TEST:
        return test(general_proto.template)
    return rec_proto.Noop()
