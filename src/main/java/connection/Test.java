package connection;

import protobuf.srl.services.recognition.PythonRecognitionService;
import protobuf.srl.sketch.Sketch;

import java.io.IOException;

/**
 * Created by david on 5/8/16.
 */
public class Test {
    public static void main(String args[]) throws IOException {
        String message = "MY CUSTOM MESSAGE";
        SocketConnection connection = new SocketConnection();
        connection.createSocket(8089);
        connection.writeOut(PythonRecognitionService.GeneralRecognitionRequest.newBuilder().setMessage(message).setTemplate(
                Sketch.RecognitionTemplate.newBuilder()
                        .setShape(Sketch.SrlShape.newBuilder().setTime(0).setId("id").addSubComponents(
                            Sketch.SrlObject.newBuilder()
                                .setType(Sketch.SrlObject.ObjectType.STROKE)
                                .setObject(makeStroke().toByteString())
                        ))
                        .setTemplateId("id"))
                .setRequestType(PythonRecognitionService.RecognitionRequestType.GENERATE_SHAPES)
                .build().toByteArray());
        System.out.println("Sent message:" + message);
        final PythonRecognitionService.GeneratedTemplates generalRecognitionRequest = PythonRecognitionService.GeneratedTemplates
                .parseFrom(connection.readIn());
        System.out.println(generalRecognitionRequest.isInitialized());
    }

    public static Sketch.SrlStroke makeStroke() {
        final Sketch.SrlStroke.Builder builder = Sketch.SrlStroke.newBuilder()
                .setId("id")
                .setTime(0)
                .addPoints(
                        Sketch.SrlPoint.newBuilder()
                                .setTime(0)
                                .setId("hi")
                                .setX(5)
                                .setY(5));

        for(int i = 0; i < 200; i++) {
            builder.addPoints(
                    Sketch.SrlPoint.newBuilder()
                            .setTime(0)
                            .setId("hi")
                            .setX(5)
                            .setY(5));
        }
        return builder.build();
    }
}
