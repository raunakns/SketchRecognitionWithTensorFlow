from connection import python_socket_server as connection
from proto_decoder import proto_handler as handler

print "STARTING PYTHON SERVER"

connection.start_socket(8089, callback=handler.message_processor)
