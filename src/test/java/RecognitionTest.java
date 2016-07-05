import com.mongodb.ServerAddress;
import coursesketch.database.RecognitionDatabaseClient;
import coursesketch.recognition.BasicRecognition;
import coursesketch.recognition.RecognitionInitializationException;
import coursesketch.recognition.framework.exceptions.TemplateException;
import coursesketch.recognition.test.RecognitionScore;
import coursesketch.recognition.test.RecognitionScoreMetrics;
import coursesketch.recognition.test.RecognitionTesting;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

    private static final Logger LOG = LoggerFactory.getLogger(RecognitionTest.class);
    public static void main(String args[]) throws UnknownHostException, RecognitionInitializationException, TemplateException {
        final List<ServerAddress> databaseUrl = new ArrayList<>();
        databaseUrl.add(new ServerAddress());

        RecognitionDatabaseClient client = new RecognitionDatabaseClient(databaseUrl, "Recognition");
        client.onStartDatabase();
        TensorFlowRecognition rec1 = new TensorFlowRecognition(client);

        rec1.initialize();

        System.out.println("Running on recognition templates");
        RecognitionTesting tester = new RecognitionTesting(client, rec1);
        List<Sketch.SrlInterpretation> allInterpretations = client.getAllInterpretations();
        List<Sketch.RecognitionTemplate> templatse = client.getTemplate(allInterpretations.get(2));
        templatse.addAll(client.getTemplate(allInterpretations.get(2)));
        List<RecognitionScoreMetrics> recognitionScoreMetrics =
                tester.testAgainstAllTemplates();
                // tester.testAgainstTemplates(templatse);
        System.out.println("done recognizer");
        for (RecognitionScoreMetrics scoreMetrics : recognitionScoreMetrics) {
            System.out.println(scoreMetrics.getSimpleName());
            scoreMetrics.computeRecognitionMetrics();
            System.out.println(scoreMetrics.getRecognitionMetric());
            System.out.println(scoreMetrics.getTrainingMetric());
            for (RecognitionScore recognitionScore : scoreMetrics.getScores()) {
                LOG.info("Correct Label: {}", recognitionScore.getCorrectInterpretations());
                LOG.info("Template ID {}", recognitionScore.getTemplateId());
                if (recognitionScore.getCorrectInterpretations() != null) {
                    for (Sketch.SrlInterpretation srlInterpretation : recognitionScore.getRecognizedInterpretations()) {
                        LOG.info("\t\t{}", srlInterpretation);
                    }
                } else {
                    LOG.info("No interpretations were returned for this defice");
                }
            }
        }
    }
}
