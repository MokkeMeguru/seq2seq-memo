import net.reduls.igo.Tagger;

import java.io.IOException;
import java.util.List;

public class SimpleJapaneseTextParser implements BaseTextParser {
    private Tagger tagger;

    public SimpleJapaneseTextParser() throws IOException {
        tagger  = new Tagger("lib/ipadic");
    }

    @Override
    public List<String> parse(String text) {
        return tagger.wakati(text);
    }

    public static void main (String... args) throws IOException {
        BaseTextParser textParser = new SimpleJapaneseTextParser();
        textParser.parse("こんにちは、紲星あかりです。").forEach(str -> System.out.println(str));
    }
}
