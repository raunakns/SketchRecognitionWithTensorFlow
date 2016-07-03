import com.mongodb.ServerAddress;
import coursesketch.database.RecognitionDatabaseClient;
import coursesketch.recognition.BasicRecognition;
import coursesketch.recognition.RecognitionInitializationException;
import coursesketch.recognition.framework.exceptions.TemplateException;
import coursesketch.recognition.test.RecognitionScoreMetrics;
import coursesketch.recognition.test.RecognitionTesting;
import protobuf.srl.sketch.Sketch;
import recognition.TensorFlowRecognition;

import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by turnerd on 6/29/16.
 */
public class RecognitionTest {

    public static void main(String args[]) throws UnknownHostException, RecognitionInitializationException, TemplateException {
        final List<ServerAddress> databaseUrl = new ArrayList<>();
        databaseUrl.add(new ServerAddress());

        RecognitionDatabaseClient client = new RecognitionDatabaseClient(databaseUrl, "Recognition");
        client.onStartDatabase();
        TensorFlowRecognition rec1 = new TensorFlowRecognition(client);
        BasicRecognition rec2 = new BasicRecognition(client);

        rec1.initialize();
        rec2.initialize();

        System.out.println("Running on recognition templates");
        RecognitionTesting tester = new RecognitionTesting(client, rec1, rec2);
        final List<Sketch.SrlInterpretation> allInterpretations = client.getAllInterpretations();
        List<RecognitionScoreMetrics> recognitionScoreMetrics = tester.testAgainstAllTemplates();
        System.out.println("done recognizer");
        for (RecognitionScoreMetrics scoreMetrics : recognitionScoreMetrics) {
            scoreMetrics.computeRecognitionMetrics();
            System.out.println(scoreMetrics.getRecognitionMetric());
            System.out.println(scoreMetrics.getTrainingMetric());
        }
    }
}
