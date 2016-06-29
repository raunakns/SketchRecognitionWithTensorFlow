import com.google.protobuf.InvalidProtocolBufferException;
import com.mongodb.ServerAddress;
import coursesketch.database.RecognitionDatabaseClient;
import coursesketch.recognition.BasicRecognition;
import coursesketch.recognition.RecognitionInitializationException;
import coursesketch.recognition.defaults.DefaultRecognition;
import protobuf.srl.commands.Commands;
import protobuf.srl.sketch.Sketch;
import recognition.TensorFlowRecognition;

import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by turnerd on 6/29/16.
 */
public class RecognitionTest {

    public static void main(String args[]) throws UnknownHostException, RecognitionInitializationException {
        final List<ServerAddress> databaseUrl = new ArrayList<>();
        databaseUrl.add(new ServerAddress());

        RecognitionDatabaseClient client = new RecognitionDatabaseClient(databaseUrl, "RecognitionServer");
        TensorFlowRecognition rec1 = new TensorFlowRecognition(client);
        rec1.train();
        BasicRecognition rec2 = new BasicRecognition(client);
        rec2.initialize();
    }

    public static void testRecognitionSystems(RecognitionDatabaseClient client, DefaultRecognition... systems) {
        boolean[][] correct = new boolean[systems.length][];
        List<Sketch.RecognitionTemplate> allTemplates = client.getAllTemplates();
        int numTemplates = allTemplates.size() / 100;
        for (int i = 0; i < systems.length; i++) {
            correct[i] = new boolean[numTemplates];
        }
        Random r = new Random();
        for (int i = 0; i < numTemplates; i++) {
            Sketch.RecognitionTemplate template = allTemplates.get(r.nextInt());

            for (int j = 0; j < systems.length; j++) {
                systems[j].recognize()
            }
        }
    }

    public static List<Sketch.SrlStroke> convert(Sketch.RecognitionTemplate template) {
        final List<Sketch.SrlStroke> strokes = new ArrayList<Sketch.SrlStroke>();
        if (template.hasStroke()) {
            // LOG.debug("Loading Template {}", template);
            strokes.add(template.getStroke());
        } else if (template.hasShape()) {
            final Sketch.SrlShape shape = template.getShape();
            for (Sketch.SrlObject object: shape.getSubComponentsList()) {
                if (object.getType() == Sketch.ObjectType.STROKE) {
                    try {
                        strokes.add(Sketch.SrlStroke.parseFrom(object.getObject()));
                    } catch (InvalidProtocolBufferException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return strokes;
    }

    public static Commands.SrlUpdateList convert(List<Sketch.SrlStroke> strokes) {

    }
}
