import com.google.protobuf.InvalidProtocolBufferException;
import com.mongodb.ServerAddress;
import coursesketch.database.RecognitionDatabaseClient;
import coursesketch.recognition.BasicRecognition;
import coursesketch.recognition.RecognitionInitializationException;
import coursesketch.recognition.defaults.DefaultRecognition;
import coursesketch.recognition.framework.exceptions.TemplateException;
import coursesketch.recognition.test.RecognitionScoreMetrics;
import coursesketch.recognition.test.RecognitionTesting;
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

    public static void main(String args[]) throws UnknownHostException, RecognitionInitializationException, TemplateException {
        final List<ServerAddress> databaseUrl = new ArrayList<>();
        databaseUrl.add(new ServerAddress());

        RecognitionDatabaseClient client = new RecognitionDatabaseClient(databaseUrl, "RecognitionServer");
        TensorFlowRecognition rec1 = new TensorFlowRecognition(client);
        rec1.initialize();
        BasicRecognition rec2 = new BasicRecognition(client);
        rec2.initialize();

        client.onStartDatabase();

        RecognitionTesting tester = new RecognitionTesting(client, rec1, rec2);
        List<RecognitionScoreMetrics> recognitionScoreMetrics = tester.testAgainstAllTemplates();
        for (RecognitionScoreMetrics scoreMetrics : recognitionScoreMetrics) {
            System.out.println(scoreMetrics);
        }

    }
}
