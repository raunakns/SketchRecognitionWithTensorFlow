from generated_proto import pythonRecognitionService_pb2 as rec_proto
from recognition import recognition_manager
from generated_proto import sketch_pb2 as Sketch

rec = recognition_manager.Recognition_manager()
rec.initialize()

def generate_shapes(recognition_template):
    """Returns {GeneratedTemplates}"""
    template_holder = rec_proto.GeneratedTemplates()
    shape_list = rec.generate_shape(recognition_template.shape)

    for shape in shape_list:
        template = Sketch.RecognitionTemplate()
        template.templateId = recognition_template.templateId
        template.shape.CopyFrom(shape)
        template_holder.generatedTemplates.extend([template])
    return template_holder

def message_processor(general_proto):
    request_type = general_proto.requestType
    print "request type"
    print request_type
    if request_type == rec_proto.GENERATE_SHAPES:
        return generate_shapes(general_proto.template)
    return rec_proto.Noop()
