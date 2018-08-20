import org.datavec.api.records.reader.impl.csv.CSVLineSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class SimpleWordSequenceSequenceCSVParser implements BaseWordSequenceParser {
    private CSVLineSequenceRecordReader recordReader;
    private BaseTextParser textParser;
    private FileSplit inputFile;

    private Map<String, Integer> wordIdDict;
    private Map<Integer, String> idWordDict;
    private int currentId;

    private int featureMaxLength;
    private int labelMaxLength;

    private List<List<String>> featuresList;
    private List<List<String>> labelsList;

    public SimpleWordSequenceSequenceCSVParser(CSVLineSequenceRecordReader recordReader,
                                               BaseTextParser textParser,
                                               FileSplit inputFile) throws IOException, InterruptedException {
        this.recordReader = recordReader;
        this.textParser = textParser;
        this.inputFile = inputFile;

        this.wordIdDict = new HashMap<>();
        this.idWordDict = new HashMap<>();
        this.recordReader.initialize(inputFile);

        this.currentId = 0;
        this.inputFile.reset();
        this.featuresList = new ArrayList<>();
        this.labelsList = new ArrayList<>();

        addWord("<unk>");
        addWord("<eos>");
        addWord("<go>");
    }

    @Override
    public void setInputFile(FileSplit file) throws IOException, InterruptedException {
        this.inputFile = file;
        this.inputFile.reset();
        this.recordReader.initialize(inputFile);
    }

    @Override
    public void run(int featureIndex, int labelIndex) {
        while(this.recordReader.hasNext()) {
            List<Writable> writables = this.recordReader.next();
            String text = writables.get(featureIndex).toString();
            String label = writables.get(labelIndex).toString();
            List<String> features = this.textParser.parse(text);
            if(features.size() > this.featureMaxLength) this.featureMaxLength = features.size();
            features.forEach(str -> addWord(str));
            List<String> labels = this.textParser.parse(label);
            if(labels.size() > this.featureMaxLength) this.labelMaxLength = labels.size();
            labels.forEach(str -> addWord(str));
            featuresList.add(features);
            labelsList.add(labels);
        }
    }

    @Override
    public void save(File featureFile, File labelFile) throws FileNotFoundException {
        PrintStream featureStream = new PrintStream(featureFile);
        AtomicBoolean notfirst = new AtomicBoolean(false);
        saveOneSide(notfirst, featureStream, featuresList);
        featureStream.flush();
        featureStream.close();

        PrintStream labelStream = new PrintStream(labelFile);
        notfirst.set(false);
        saveOneSide(notfirst, labelStream, labelsList);
        labelStream.flush();
        labelStream.close();
    }

    private void saveOneSide(AtomicBoolean notfirst, PrintStream labelStream, List<List<String>> labelsList) {
        labelsList.forEach(labels -> {
            labelStream.println();
            notfirst.set(false);
            labels.forEach(str -> {
                if (notfirst.getAndSet(true)) {
                    labelStream.print(',');
                }
                labelStream.print(wordIdDict.get(str));
            });
        });
    }

    @Override
    public void runAndSave(int featureIndex, int labelIndex, File featureFile, File labelFile) throws FileNotFoundException {
        this.run(featureIndex, labelIndex);
        this.save(featureFile, labelFile);
    }

    @Override
    public Map<String, Integer> getwordIdDict() {
        return this.wordIdDict;
    }

    @Override
    public Map<Integer, String> getIdWordDict() {
        return this.idWordDict;
    }

    @Override
    public void setwordIdDict(Map<String, Integer> wordIdDict, Map<Integer, String> idWordDict) {
        this.wordIdDict = wordIdDict;
        this.idWordDict = idWordDict;
        this.currentId = wordIdDict.size();
    }

    @Override
    public int getFeatureMaxLength() {
        return this.featureMaxLength;
    }

    @Override
    public void setFeatureMaxLength(int featureMaxLength) {
        this.featureMaxLength = featureMaxLength;
    }

    @Override
    public int getLabelSize() {
        return 0;
    }

    @Override
    public void setLabelSize(int labelSize) {

    }

    @Override
    public int getDictSize() {
        return currentId;
    }

    @Override
    public int getLabelMaxLength() {
        return this.labelMaxLength;
    }

    @Override
    public void setLabelMaxLength(int labelMaxLength) {
        this.labelMaxLength = labelMaxLength;
    }

    @Override
    public void addWord(String word) {
        if(!wordIdDict.containsKey(word)) {
            wordIdDict.put(word, currentId);
            idWordDict.put(currentId, word);
            ++currentId;
        }
    }

    @Override
    public List<Integer> text2vecs(String text, String unknown) {
        return this.textParser.parse(text)
                .stream()
                .map(str -> {
                    if (wordIdDict.containsKey(str)) {
                        return wordIdDict.get(str);
                    } else {
                        return wordIdDict.get("<unk>");
                    }
                })
                .collect(Collectors.toList());
    }

    public static void main (String... args) throws IOException, InterruptedException {
        System.out.print("Start!");
        BaseWordSequenceParser sequenceParser =
                new SimpleWordSequenceSequenceCSVParser(
                                new CSVLineSequenceRecordReader(0,','),
                                new SimpleJapaneseTextParser(),
                                new FileSplit(new File("resources/japanese-corpus/usually"))
                );
        sequenceParser.runAndSave(0,1,
                new File("resources/features.csv"),
                new File("resources/label.csv"));
        System.out.print("Finish!");
        System.out.println("Information: \n" +
                "DictSize" + sequenceParser.getDictSize() +
                "RowSize" + sequenceParser.getFeatureMaxLength());
    }
}
