import org.apache.commons.lang.ArrayUtils;
import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@SuppressWarnings("serial")
public class Seq2SeqCustomDataSetIterator implements MultiDataSetIterator {
    private List<List<Double>> features;
    private List<List<Double>> labels;

    private int batchSize;
    private int batchesPerMacrobatch;
    private int totalBatches;
    private int totalMacroBatches;
    private int currentBatch = 0;
    private int currentMacroBatch = 0;
    private int dictSize;
    private int rowSize;
    private MultiDataSetPreProcessor preProcessor;

    public Seq2SeqCustomDataSetIterator(List<List<Double>> features,
                                        List<List<Double>> labels,
                                        int batchSize,
                                        int batchesPerMacrobatch,
                                        int dictSize,
                                        int rowSize) {
        this.features = features;
        this.labels = labels;
        this.batchSize = batchSize;
        this.batchesPerMacrobatch = batchesPerMacrobatch;
        this.dictSize = dictSize;
        this.rowSize = rowSize;

        this.totalBatches =
                (int) Math.ceil((double) this.features.size() / this.batchSize);
        this.totalMacroBatches =
                (int) Math.ceil((double) this.totalBatches / this.batchesPerMacrobatch);
    }

    public Seq2SeqCustomDataSetIterator(File featuresFile,
                                        File labelsFile,
                                        int offset,
                                        int batchSize,
                                        int batchesPerMacrobatch,
                                        int dictSize,
                                        int rowSize) throws IOException, InterruptedException {

        CSVLineSequenceRecordReader featuresReader = new CSVLineSequenceRecordReader(offset,',');
        featuresReader.initialize(new FileSplit(featuresFile));
        List<List<Double>> listList = new ArrayList<>();
        while(featuresReader.hasNext()){
            listList.add(featuresReader.next().stream().map(w -> w.toDouble()).collect(Collectors.toList()));
        }

        CSVLineSequenceRecordReader labelsReader = new CSVLineSequenceRecordReader(offset, ',');
        labelsReader.initialize(new FileSplit(labelsFile));
        List<List<Double>> listList1 = new ArrayList<>();
        while(labelsReader.hasNext()){
            listList1.add(labelsReader.next().stream().map(w -> w.toDouble()).collect(Collectors.toList()));
        }

        this.features = listList;
        this.labels = listList1;
        this.batchSize = batchSize;
        this.batchesPerMacrobatch = batchesPerMacrobatch;
        this.dictSize = dictSize;
        this.rowSize = rowSize;

        this.totalBatches =
                (int) Math.ceil((double) this.features.size() / this.batchSize);
        this.totalMacroBatches =
                (int) Math.ceil((double) this.totalBatches / this.batchesPerMacrobatch);
    }

    private int getMacroBatchByCurrentBatch() {
        return this.currentBatch / this.batchesPerMacrobatch;
    }

    @Override
    public MultiDataSet next(int i) {
        int pos = this.currentBatch * this.batchSize;
        int currentBatchSize = Math.min(this.batchSize, this.features.size() - pos);

        INDArray input = Nd4j.zeros(currentBatchSize, 1, this.rowSize);
        INDArray prediction = Nd4j.zeros(currentBatchSize, this.dictSize, this.rowSize);
        INDArray decode = Nd4j.zeros(currentBatchSize, this.dictSize, this.rowSize);
        INDArray inputMask = Nd4j.zeros(currentBatchSize, this.rowSize);
        INDArray predictionMask = Nd4j.zeros(currentBatchSize, this.rowSize);

        for (int j = 0; j < currentBatchSize; j++) {
            List<Double> rowIn = new ArrayList<>(this.features.get(pos));
            List<Double> rowPred = new ArrayList<>(this.labels.get(pos));

            rowPred.add(1.0); // add <eos> WARNING:  <eos> == 1.0 in the dictionary.
            input.put(new INDArrayIndex[]{
                            NDArrayIndex.point(j),
                            NDArrayIndex.point(0),
                            NDArrayIndex.interval(0, rowIn.size())},
                    Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0]))));
            inputMask.put(new INDArrayIndex[]{
                            NDArrayIndex.point(j),
                            NDArrayIndex.interval(0, rowIn.size())},
                    Nd4j.ones(rowIn.size()));
            predictionMask.put(new INDArrayIndex[]{
                            NDArrayIndex.point(j),
                            NDArrayIndex.interval(0, rowPred.size())},
                    Nd4j.ones(rowPred.size()));

            double predOneHot[][] = new double[this.dictSize][rowPred.size()];
            double decodeOneHot[][] = new double[this.dictSize][rowPred.size()];
            decodeOneHot[2][0] = 1; // add <go> WARNING: <go> == 2.0 in the dictionary.

            // pred  :     A   B   C   ... Z  <eos>
            // decode:   <go>  A   B   C  ...    Z

            int predIdx = 0;
            for (Double pred :
                    rowPred) {
                predOneHot[pred.intValue()][predIdx] = 1;
                if (predIdx < rowPred.size() - 1)
                    decodeOneHot[pred.intValue()][predIdx + 1] = 1;
                ++predIdx;
            }

            decode.put(new INDArrayIndex[]{
                            NDArrayIndex.point(j),
                            NDArrayIndex.interval(0, this.dictSize),
                            NDArrayIndex.interval(0, rowPred.size())},
                    Nd4j.create(decodeOneHot));

            prediction.put(new INDArrayIndex[]{
                            NDArrayIndex.point(j),
                            NDArrayIndex.interval(0, this.dictSize),
                            NDArrayIndex.interval(0, rowPred.size())},
                    Nd4j.create(predOneHot));

            ++pos;
        }
        ++this.currentBatch;
        return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{ input, decode},
                new INDArray[] { prediction},
                new INDArray[] { inputMask, predictionMask},
                new INDArray[] { predictionMask});
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        this.currentBatch = 0;
        this.currentMacroBatch = 0;
    }

    @Override
    public boolean hasNext() {
        return (this.currentBatch < this.totalBatches)
                && (getMacroBatchByCurrentBatch() == this.currentMacroBatch);
    }

    @Override
    public MultiDataSet next() {
        return next(this.batchSize);
    }

    public int getCurrentBatch() {
        return this.currentBatch;
    }

    public int getTotalBatches() {
        return this.totalBatches;
    }

    public void setCurrentBatch(int currentBatch) {
        this.currentBatch = currentBatch;
        this.currentMacroBatch = getMacroBatchByCurrentBatch();
    }

    public boolean hasNextMacrobatch() {
        return (getMacroBatchByCurrentBatch() < this.totalMacroBatches)
                && (this.currentMacroBatch < this.totalMacroBatches);
    }

    public void nextMacroBatch() {
        ++currentMacroBatch;
    }
}
