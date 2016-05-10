package connection;

import protobuf.srl.services.recognition.PythonRecognitionService;

import java.io.IOException;

/**
 * Created by david on 5/8/16.
 */
public class Test {
    public static void main(String args[]) throws IOException {
        String message = "MY CUSTOM MESSAGE";
        SocketConnection connection = new SocketConnection();
        connection.createSocket(8089);
        connection.writeOut(PythonRecognitionService.GeneralRecognitionRequest.newBuilder().setMessage(message).setRequestType(
                PythonRecognitionService.RecognitionRequestType.ADD_TEMPLATE).build().toByteArray());
        System.out.println("Sent message:" + message);
        final PythonRecognitionService.GeneralRecognitionRequest generalRecognitionRequest = PythonRecognitionService.GeneralRecognitionRequest
                .parseFrom(connection.readIn());
        System.out.println(generalRecognitionRequest.getMessage());
    }
}
