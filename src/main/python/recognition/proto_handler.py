from generated_proto import pythonRecognitionService_pb2 as rec_proto

def generate_shapes(recognition_template):
    """Returns {GeneratedTemplates}"""
    return rec_proto.GeneratedTemplates()

def message_processor(general_proto):
    request_type = general_proto.requestType
    print "request type"
    print request_type
    if request_type == rec_proto.GENERATE_SHAPES:
        return generate_shapes(general_proto.template)
    return rec_proto.Noop()
