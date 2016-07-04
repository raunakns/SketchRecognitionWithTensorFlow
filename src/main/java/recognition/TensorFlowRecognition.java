package recognition;

import com.google.common.collect.Lists;
import com.google.protobuf.InvalidProtocolBufferException;
import connection.SocketConnection;
import coursesketch.database.ShapeConverter;
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
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Created by David Windows on 5/10/2016.
 */
public class TensorFlowRecognition extends DefaultRecognition {

    public static final int PYTHON_PORT = 8053;

    /**
     * Declaration and Definition of Logger.
     */
    private static final Logger LOG = LoggerFactory.getLogger(TensorFlowRecognition.class);

    private final SocketConnection pythonConnection;
    private boolean initialized = false;

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
    public synchronized void trainTemplate(Sketch.RecognitionTemplate recognitionTemplate) throws TemplateException {
        final Sketch.RecognitionTemplate.Builder builder = recognitionTemplate.toBuilder();
        if (recognitionTemplate.hasStroke()) {
            Sketch.SrlStroke stroke = recognitionTemplate.getStroke();
            Sketch.SrlObject object = Sketch.SrlObject.newBuilder().setType(Sketch.ObjectType.STROKE)
                    .setObject(stroke.toByteString()).build();
            Sketch.SrlShape shape = Sketch.SrlShape.newBuilder().setId(UUID.randomUUID().toString())
                    .setTime(0).addSubComponents(object).build();
            builder.clearStroke();
            builder.setShape(shape);
        }
        final PythonRecognitionService.GeneralRecognitionRequest recognitionRequest = PythonRecognitionService.GeneralRecognitionRequest.newBuilder()
                .setRequestType(PythonRecognitionService.RecognitionRequestType.TRAIN)
                .setTemplate(builder).build();
        try {
            pythonConnection.writeOut(recognitionRequest.toByteArray());
        } catch (IOException e) {
            throw new TemplateException("Error writing out message to python while generating shapes", e);
        }
        PythonRecognitionService.Noop noop;
        try {
            noop = PythonRecognitionService.Noop
                    .parseFrom(pythonConnection.readIn());
        } catch (InvalidProtocolBufferException e) {
            throw new TemplateException("Error parsing response from python while generating shapes", e);
        } catch (IOException e) {
            throw new TemplateException("Error reading response from python while generating shapes", e);
        }
        if (noop == null) {
            throw new TemplateException("No return value created");
        }
        /*
        if (noop.getSuccess() == false) {
            throw new TemplateException("Invalid data for the given template");
        }
        */
    }

    @Override public Commands.SrlUpdateList recognize(final String s, final Commands.SrlUpdateList srlUpdateList) throws RecognitionException {
        return null;
    }

    @Override public Sketch.SrlSketch recognize(final String s, final Sketch.SrlSketch srlSketch) throws RecognitionException {
        return null;
    }

    @Override
    public synchronized List<Sketch.SrlInterpretation> recognize(String s, Sketch.RecognitionTemplate recognitionTemplate) throws RecognitionException {
        final Sketch.RecognitionTemplate.Builder builder = recognitionTemplate.toBuilder();
        if (recognitionTemplate.hasStroke()) {
            Sketch.SrlStroke stroke = recognitionTemplate.getStroke();
            Sketch.SrlObject object = Sketch.SrlObject.newBuilder().setType(Sketch.ObjectType.STROKE)
                    .setObject(stroke.toByteString()).build();
            Sketch.SrlShape shape = Sketch.SrlShape.newBuilder().setId(UUID.randomUUID().toString())
                    .setTime(0).addSubComponents(object).build();
            builder.clearStroke();
            builder.setShape(shape);
        }
        final PythonRecognitionService.GeneralRecognitionRequest recognitionRequest = PythonRecognitionService.GeneralRecognitionRequest.newBuilder()
                .setRequestType(PythonRecognitionService.RecognitionRequestType.TEST)
                .setTemplate(builder).build();
        try {
            pythonConnection.writeOut(recognitionRequest.toByteArray());
        } catch (IOException e) {
            throw new RecognitionException("Error writing out message to python while generating shapes", e);
        }
        Sketch.RecognitionTemplate recognitionResults;
        try {
            recognitionResults = Sketch.RecognitionTemplate
                    .parseFrom(pythonConnection.readIn());
        } catch (InvalidProtocolBufferException e) {
            throw new RecognitionException("Error parsing response from python while generating shapes", e);
        } catch (IOException e) {
            throw new RecognitionException("Error reading response from python while generating shapes", e);
        }
        return recognitionResults.getInterpretationsList();
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

    private void loadLabels() throws TemplateException {
        final List<Sketch.SrlInterpretation> allInterpretations = super.getTemplateDatabase().getAllInterpretations();
        List<String> labels = new ArrayList<>();
        for (Sketch.SrlInterpretation allInterpretation : allInterpretations) {
            labels.add(allInterpretation.getLabel());
        }

        LOG.debug("ADDING {} LABELS", labels.size());
        final PythonRecognitionService.GeneralRecognitionRequest recognitionRequest = PythonRecognitionService.GeneralRecognitionRequest.newBuilder()
                .setRequestType(PythonRecognitionService.RecognitionRequestType.INIT)
                .addAllLabels(labels).build();
        try {
            pythonConnection.writeOut(recognitionRequest.toByteArray());
        } catch (IOException e) {
            throw new TemplateException("Error writing out message to python while generating shapes", e);
        }
        PythonRecognitionService.Noop noop;
        try {
            noop = PythonRecognitionService.Noop
                    .parseFrom(pythonConnection.readIn());
        } catch (InvalidProtocolBufferException e) {
            throw new TemplateException("Error parsing response from python while generating shapes", e);
        } catch (IOException e) {
            throw new TemplateException("Error reading response from python while generating shapes", e);
        }
        LOG.debug("LABELS HAVE BEEN ADDED", noop);
    }

    @Override
    public void finishTraining() throws RecognitionException {
        LOG.debug("telling training to finish up");
        final PythonRecognitionService.GeneralRecognitionRequest recognitionRequest = PythonRecognitionService.GeneralRecognitionRequest.newBuilder()
                .setRequestType(PythonRecognitionService.RecognitionRequestType.FINISH_TRAINING).build();
        try {
            pythonConnection.writeOut(recognitionRequest.toByteArray());
        } catch (IOException e) {
            throw new TemplateException("Error writing out message to python while generating shapes", e);
        }
        PythonRecognitionService.Noop noop;
        try {
            noop = PythonRecognitionService.Noop
                    .parseFrom(pythonConnection.readIn());
        } catch (InvalidProtocolBufferException e) {
            throw new TemplateException("Error parsing response from python while generating shapes", e);
        } catch (IOException e) {
            throw new TemplateException("Error reading response from python while generating shapes", e);
        }
        LOG.debug("Training has been completed", noop);
    }

    @Override
    public void initialize() {
        if (initialized) {
            return;
        }
        try {
            loadLabels();
        } catch (TemplateException e) {
            LOG.error("UNABLE TO CREATE LABELS", e);
        }
        initialized = true;
    }
}
