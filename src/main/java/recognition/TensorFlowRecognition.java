package recognition;

import com.google.protobuf.InvalidProtocolBufferException;
import connection.SocketConnection;
import coursesketch.recognition.defaults.DefaultRecognition;
import coursesketch.recognition.framework.TemplateDatabaseInterface;
import coursesketch.recognition.framework.exceptions.RecognitionException;
import coursesketch.recognition.framework.exceptions.TemplateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import protobuf.srl.commands.Commands;
import protobuf.srl.services.recognition.PythonRecognitionService;
import protobuf.srl.sketch.Sketch;

import java.io.IOException;
import java.util.List;

/**
 * Created by David Windows on 5/10/2016.
 */
public class TensorFlowRecognition extends DefaultRecognition {

    public static final int PYTHON_PORT = 8089;

    /**
     * Declaration and Definition of Logger.
     */
    private static final Logger LOG = LoggerFactory.getLogger(TensorFlowRecognition.class);

    private final SocketConnection pythonConnection;

    public TensorFlowRecognition(final TemplateDatabaseInterface templateDatabase) {
        super(templateDatabase);
        pythonConnection = new SocketConnection();
        try {
            pythonConnection.createSocket(PYTHON_PORT);
        } catch (IOException e) {
            LOG.error("Error opening socket at port: " + PYTHON_PORT, e);
        }
    }

    @Override public Commands.SrlUpdateList addUpdate(final String s, final Commands.SrlUpdate srlUpdate) throws RecognitionException {
        return null;
    }

    @Override public Commands.SrlUpdateList setUpdateList(final String s, final Commands.SrlUpdateList srlUpdateList) throws RecognitionException {
        return null;
    }

    @Override public Sketch.SrlSketch setSketch(final String s, final Sketch.SrlSketch srlSketch) throws RecognitionException {
        return null;
    }

    @Override
    public void trainTemplate(Sketch.RecognitionTemplate recognitionTemplate) throws TemplateException {

    }

    @Override public Commands.SrlUpdateList recognize(final String s, final Commands.SrlUpdateList srlUpdateList) throws RecognitionException {
        return null;
    }

    @Override public Sketch.SrlSketch recognize(final String s, final Sketch.SrlSketch srlSketch) throws RecognitionException {
        return null;
    }

    @Override
    public List<Sketch.SrlInterpretation> recognize(String s, Sketch.RecognitionTemplate recognitionTemplate) throws RecognitionException {
        return null;
    }

    @Override public List<Sketch.RecognitionTemplate> generateTemplates(final Sketch.RecognitionTemplate recognitionTemplate)
            throws RecognitionException {
        final PythonRecognitionService.GeneralRecognitionRequest recognitionRequest = PythonRecognitionService.GeneralRecognitionRequest.newBuilder()
                .setRequestType(PythonRecognitionService.RecognitionRequestType.GENERATE_SHAPES)
                .setTemplate(recognitionTemplate).build();
        try {
            pythonConnection.writeOut(recognitionRequest.toByteArray());
        } catch (IOException e) {
            throw new RecognitionException("Error writing out message to python while generating shapes", e);
        }
        PythonRecognitionService.GeneratedTemplates generatedTemplates;
        try {
            generatedTemplates = PythonRecognitionService.GeneratedTemplates
                    .parseFrom(pythonConnection.readIn());
        } catch (InvalidProtocolBufferException e) {
            throw new RecognitionException("Error parsing response from python while generating shapes", e);
        } catch (IOException e) {
            throw new RecognitionException("Error reading response from python while generating shapes", e);
        }

        return generatedTemplates.getGeneratedTemplatesList();
    }

    @Override
    public void initialize() {

    }

    public synchronized void train() {

    }
}
