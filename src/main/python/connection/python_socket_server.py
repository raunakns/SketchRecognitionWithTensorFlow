import socket
import struct
from generated_proto import pythonRecognitionService_pb2 as rec_proto

def start_socket(port, callback):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('localhost', port))
    serversocket.listen(5) # become a server socket, maximum 5 connections

    while True:
        connection, address = serversocket.accept()
        while True:
            receive_message(connection)
            print "read one message"

def receive_message(connection, callback):
    # how long the binary message is
    totallen = connection.recv(4)
    totallenRecv = struct.unpack('>I', totallen)[0]

    # the protobuf message
    message = connection.recv(totallenRecv)
    general_proto = rec_proto.GeneralRecognitionRequest()
    general_proto.ParseFromString(message)
    result = callback(general_proto)
    print result