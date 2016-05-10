from connection import python_socket_server as connection

print "STARTING PYTHON SERVER"

def callback(general_proto):
    print general_proto.message
    return "message was read!"


connection.start_socket(8089, callback=callback)