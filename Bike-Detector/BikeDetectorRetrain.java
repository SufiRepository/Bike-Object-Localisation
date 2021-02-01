package ai.certifai.solution.object_detection.BikeDetector;

import ai.certifai.solution.object_detection.ActorsDetector.tinyyolo.BikeDetector;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ColorConversionTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;
import java.util.List;

import static ai.certifai.solution.object_detection.BikeDetector.BikeDataSetIterator.*;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_core.RGB;

public class BikeDetectorRetrain {

    private static final Logger log = LoggerFactory.getLogger(BikeDetector.class);

    private static int seed = 123;
    private static double detectionThreshold = 0.1; // 0.5
    private static int nBoxes = 5;
    private static double lambdaNoObj = 0.5;
    private static double lambdaCoord = 5.0; // 5.0
    private static double[][] priorBoxes = { { 1, 3 }, { 2.5, 6 }, { 3, 4 }, { 3.5, 8 }, { 4, 9 } }; // {{1, 3}, {2.5,
                                                                                                     // 6}, {3, 4},
                                                                                                     // {3.5, 8}, {4,
                                                                                                     // 9}},{{2, 5},
                                                                                                     // {2.5, 6}, {3,
                                                                                                     // 7}, {3.5, 8},
                                                                                                     // {4, 9}}
    public static int batchSize = 2; // 2
    private static int nEpochs = 2; // 40
    private static double learningRate = 1e-4;
    private static int nClasses = 1;
    private static List<String> labels;

    private static File modelFilename = new File(System.getProperty("user.dir"),
            "generated-models/BikeDetector_yolov2.zip");
    private static ComputationGraph model;
    private static Frame frame = null;
    private static final Scalar GREEN = RGB(0, 255.0, 0);
    private static final Scalar YELLOW = RGB(255, 255, 0);
    private static Scalar[] colormap = { GREEN, YELLOW };
    private static String labeltext = null;

    public static void main(String[] args) throws Exception {

        // STEP 1 : Create iterators
        BikeDataSetIterator.setup();
        RecordReaderDataSetIterator trainIter = BikeDataSetIterator.trainIterator(batchSize);
        RecordReaderDataSetIterator testIter = BikeDataSetIterator.testIterator(1);
        labels = trainIter.getLabels();

        // If model does not exist, train the model, else directly go to model
        // evaluation and then run real time object detection inference.
        if (modelFilename.exists()) {
            // STEP 2 : Load trained model from previous execution
            Nd4j.getRandom().setSeed(seed);
            log.info("Load model...");
            model = ModelSerializer.restoreComputationGraph(modelFilename);
            System.out.println(model.summary(InputType.convolutional(yoloheight, yolowidth, nClasses)));
        } else {
            Nd4j.getRandom().setSeed(seed);
            INDArray priors = Nd4j.create(priorBoxes);
            // STEP 2 : Train the model using Transfer Learning
            // STEP 2.1: Transfer Learning steps - Load TinyYOLO prebuilt model.
            log.info("Build model...");
            ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();

            // STEP 2.2: Transfer Learning steps - Model Configurations.
            FineTuneConfiguration fineTuneConf = getFineTuneConfiguration();

            // STEP 2.3: Transfer Learning steps - Modify prebuilt model's architecture
            model = getComputationGraph(pretrained, priors, fineTuneConf);
            System.out.println(model.summary(InputType.convolutional(yoloheight, yolowidth, nClasses)));

            // STEP 2.4: Training and Save model.
            log.info("Train model...");
            UIServer server = UIServer.getInstance();
            StatsStorage storage = new InMemoryStatsStorage();
            server.attach(storage);
            model.setListeners(new ScoreIterationListener(1), new StatsListener(storage));

            for (int i = 1; i < nEpochs + 1; i++) {
                trainIter.reset();
                while (trainIter.hasNext()) {
                    model.fit(trainIter.next());
                }
                log.info("*** Completed epoch {} ***", i);
            }
            ModelSerializer.writeModel(model, modelFilename, true);
            System.out.println("Model saved.");
        }

        // STEP 3: Evaluate the model's accuracy by using the test iterator.
        OfflineValidationWithTestDataset(testIter);
    }

    private static ComputationGraph getComputationGraph(ComputationGraph pretrained, INDArray priors,
            FineTuneConfiguration fineTuneConf) {
        return new TransferLearning.GraphBuilder(pretrained).fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_23").removeVertexKeepConnections("outputs")
                .addLayer("conv2d_23",
                        new ConvolutionLayer.Builder(1, 1).nIn(1024).nOut(nBoxes * (5 + nClasses)).stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same).weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY).build(),
                        "leaky_re_lu_22")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder().lambdaNoObj(lambdaNoObj).lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors.castTo(DataType.FLOAT)).build(),
                        "conv2d_23")
                .setOutputs("outputs").build();
    }

    private static FineTuneConfiguration getFineTuneConfiguration() {

        return new FineTuneConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(learningRate).build()).l2(0.00001)
                .activation(Activation.IDENTITY).trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED).build();
    }

    private static void OfflineValidationWithTestDataset(RecordReaderDataSetIterator test)
            throws InterruptedException, IOException {

        File videoPath = new ClassPathResource("/bicycle/mountainbike_test.mp4").getFile();
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoPath.getAbsolutePath());
        grabber.setFormat("mp4");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        grabber.start();

        String winName = "Object Detection";
        CanvasFrame canvas = new CanvasFrame(winName);

        int w = grabber.getImageWidth();
        int h = grabber.getImageHeight();

        canvas.setCanvasSize(w, h);
        NativeImageLoader loader = new NativeImageLoader(yolowidth, yoloheight, 3,
                new ColorConversionTransform(COLOR_BGR2RGB));
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model
                .getOutputLayer(0);
        System.out.println("Start running video");

        while ((grabber.grab()) != null) {
            Frame frame = grabber.grabImage();

            // if a thread is null, create new thread
            Mat rawImage = converter.convert(frame);
            Mat resizeImage = new Mat();// rawImage);
            resize(rawImage, resizeImage, new Size(yolowidth, yoloheight));
            INDArray inputImage = loader.asMatrix(resizeImage);
            scaler.transform(inputImage);
            INDArray results = model.outputSingle(inputImage);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);

            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());

                double proba = obj.getConfidence();

                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
                putText(rawImage, label + " " + String.format("%.2f", proba * 100) + "%",
                        new Point((x1 + 2), (y1 + y2) / 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
            }
            canvas.showImage(converter.convert(rawImage));

            KeyEvent t = canvas.waitKey(33);

            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
                break;
            }
        }
        canvas.dispose();

    }
}