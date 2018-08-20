import org.datavec.api.split.FileSplit;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface BaseWordSequenceParser {

    void setInputFile(FileSplit file) throws IOException, InterruptedException;
    void run(int featureIndex, int labelIndex);
    void save(File featureFile, File labelFile) throws FileNotFoundException;
    void runAndSave(int featureIndex, int labelIndex, File featureFile, File labelFile) throws FileNotFoundException;
    Map<String, Integer> getwordIdDict();
    Map<Integer, String> getIdWordDict();
    void setwordIdDict(Map<String, Integer> wordIdDict, Map<Integer, String> idWordDict);
    int getFeatureMaxLength ();
    void setFeatureMaxLength (int featureMaxLength);
    int getLabelSize ();
    void setLabelSize (int labelSize);
    int getDictSize();
    int getLabelMaxLength ();
    void setLabelMaxLength (int labelMaxLength);
    void addWord(String word);
    List<Integer> text2vecs(String text, String unknown);
}
