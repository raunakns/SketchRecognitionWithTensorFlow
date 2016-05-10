import socket
import struct
from generated_proto import pythonRecognitionService_pb2 as rec_proto

def start_socket(port, callback):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('localhost', port))
    serversocket.listen(1) # become a server socket, maximum 5 connections

    while True:
        connection, address = serversocket.accept()
        while True:
            process_message(connection, callback=callback)
            print "read one message"

def recvall(socket, length):
    # Helper function to recv n bytes or return None if EOF is hit
    data = ''
    while len(data) < length:
        packet = socket.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

def process_message(connection, callback):
    # how long the binary message is
    totallen = connection.recv(4)

    length2 = struct.unpack("<I", totallen)[0]
    print "value of message"
    print length2

    # the protobuf message
    message = recvall(connection, length2)
    general_proto = rec_proto.GeneralRecognitionRequest()
    general_proto.ParseFromString(message)
    result = callback(general_proto)
    send_message(connection=connection, msg=result)

def send_message(connection, msg):
    print 'sending data back!'
    data = msg.SerializeToString()
    length = len(data)
    print 'data has length ' + str(len(data))
    pack1 = struct.pack('>I', length) # the first part of the message is length
    connection.sendall(pack1)
    connection.sendall(data)
    print 'data has been sent'
