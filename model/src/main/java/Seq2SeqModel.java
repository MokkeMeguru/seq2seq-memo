import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class Seq2SeqModel {
    private Seq2SeqCustomDataSetIterator dataSetIterator;
    private int dictSize;
    private int featureMaxLength;
    private File networkFile;
    private File backupFile;

    private ComputationGraph net;

    public Seq2SeqModel(Seq2SeqCustomDataSetIterator dataSetIterator,
                        int dictSize,
                        int featureMaxLength,
                        File networkFile,
                        File backupFile) {
        this.dataSetIterator = dataSetIterator;
        this.dictSize = dictSize;
        this.featureMaxLength = featureMaxLength;
        this.networkFile = networkFile;
        this.backupFile = backupFile;
    }

    public void loadNetWork(File file) throws IOException {
        net = ModelSerializer.restoreComputationGraph(file);
    }

    public void initNetWork(boolean showUI) {
        final NeuralNetConfiguration.Builder builder =
                new NeuralNetConfiguration.Builder()
                        .seed(246)
                        .updater(new RmsProp(5e-2))
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final ComputationGraphConfiguration.GraphBuilder graphBuilder =
                builder.graphBuilder()
                        .pretrain(true)
                        .backprop(true)
                        .backpropType(BackpropType.Standard)
                        .tBPTTBackwardLength(25)
                        .tBPTTForwardLength(25)
                        .addInputs("encoderLine", "decoderLine")
                        .setInputTypes(InputType.recurrent(dictSize),
                                InputType.recurrent(dictSize))
                        .addLayer("embeddingEncoder",
                                new EmbeddingLayer.Builder()
                                        .nIn(dictSize)
                                        .nOut(128 * 2)
                                        .build(),
                                "encoderLine")
                        .addLayer("encoder",
                                new LSTM.Builder()
                                        .nIn(128 * 2)
                                        .nOut(512 * 2)
                                        .activation(Activation.TANH)
                                        .build(),
                                "embeddingEncoder")
                        .addVertex("thoughtVector",
                                new LastTimeStepVertex("encoderLine"),
                                "encoder")
                        .addVertex("dup",
                                new DuplicateToTimeSeriesVertex("decoderLine"),
                                "thoughtVector")
                        .addVertex("merge",
                                new MergeVertex(),
                                "decoderLine","dup")
                        .addLayer("decoder",
                                new LSTM.Builder()
                                    .nIn(dictSize + 512 * 2)
                                    .nOut(512 * 2)
                                    .activation(Activation.TANH)
                                    .build(),
                                "merge")
                        .addLayer("output",
                                new RnnOutputLayer.Builder()
                                    .nIn(512 * 2)
                                    .nOut(dictSize)
                                    .activation(Activation.SOFTMAX)
                                    .lossFunction(LossFunctions.LossFunction.MCXENT)
                                    .build(),
                                "decoder")
                        .setOutputs("output");


        net = new ComputationGraph(graphBuilder.build());
        net.init();

        StatsStorage statsStorage = new FileStatsStorage(new File("resources/UIStorage.bin"));
        statsStorage.removeAllListeners();
        if (showUI) {
            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);
        }
        net.setListeners(new StatsListener(statsStorage));
    }

    public void train(int offset) throws IOException {
        long lastSaveTime = System.currentTimeMillis();
        for(int epoch = 1; epoch < 600; ++epoch) {
            System.out.println("Epoch " + epoch);
            if (epoch == 1) this.dataSetIterator.setCurrentBatch(offset);
            else this.dataSetIterator.reset();
            while(this.dataSetIterator.hasNextMacrobatch()) {
                net.fit(this.dataSetIterator);
                this.dataSetIterator.nextMacroBatch();
                System.out.println("Batch = " + this.dataSetIterator.getCurrentBatch());
                if (System.currentTimeMillis() - lastSaveTime > 100000) {
                    saveModel();
                    lastSaveTime = System.currentTimeMillis();
                }
            }
        }
    }

    private void saveModel() throws IOException {
        System.out.println("Saving the model");
        ModelSerializer.writeModel(this.net, this.backupFile, true);
        System.out.println("Done.");
    }

    public static void main(String... args) throws IOException, InterruptedException {
        int offset = 0;
        int dictSize = 780;
        int rowSize = 22;
        Seq2SeqModel seq2SeqModel =
                new Seq2SeqModel(
                        new Seq2SeqCustomDataSetIterator(
                                new File("resources/features.csv"),
                                new File("resources/label.csv"),
                                1,
                                200,
                                2,
                                dictSize,
                                rowSize),
                        dictSize,
                        rowSize,
                        new File("resources/network.bin"),
                        new File("resources/model.bin"));
        seq2SeqModel.initNetWork(true);
        seq2SeqModel.train(offset);
        seq2SeqModel.saveModel();
        return;
    }
}

