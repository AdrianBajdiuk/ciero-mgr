import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;

public class ResultSerializer {

    private static final char DEFAULT_SEPARATOR = ',';
    private String outputFileName;

    public ResultSerializer(String outputFileName) {
        this.outputFileName = outputFileName;
    }

    public void serialize(List<Future<SingleRunResult>> result) throws IOException, ExecutionException, InterruptedException {

        FileWriter writer = new FileWriter(outputFileName);

        List<String> row = new LinkedList<String>();
        row.add("classifier");
        row.add("probe");
        row.add("singleClassifier");
        for (Class<? extends AbstractClassifier> ensemble : result.get(0).get().getEnsembleMeanSquareError().keySet()) {
            row.add(ensemble.getSimpleName());
        }
        //write header row:
        writeLine(writer,row,DEFAULT_SEPARATOR,' ');

        for (Future<SingleRunResult> f : result) {
            row = new LinkedList<String>();
            SingleRunResult s = f.get();
            row.add(s.getClassifier().getSimpleName());
            row.add(new Integer(s.getProbe()).toString());
            row.add(new Double(s.getMeanSquareError()).toString());
            for (Class<? extends AbstractClassifier> ensemble : s.getEnsembleMeanSquareError().keySet()) {
                row.add(new Double(s.getEnsembleMeanSquareError().get(ensemble)).toString());
            }
            writeLine(writer,row,DEFAULT_SEPARATOR,' ');
        }

        writer.flush();
        writer.close();
    }

    public  void  serializeOptions(String[] options, String classifier){

    }

    private void writeLine(Writer w, List<String> values, char separators, char customQuote) throws IOException {

        boolean first = true;

        //default customQuote is empty

        if (separators == ' ') {
            separators = DEFAULT_SEPARATOR;
        }

        StringBuilder sb = new StringBuilder();
        for (String value : values) {
            if (!first) {
                sb.append(separators);
            }
            if (customQuote == ' ') {
                sb.append(followCVSformat(value));
            } else {
                sb.append(customQuote).append(followCVSformat(value)).append(customQuote);
            }

            first = false;
        }
        sb.append("\n");
        w.append(sb.toString());

    }
    private String followCVSformat(String value) {

        String result = value;
        if (result.contains("\"")) {
            result = result.replace("\"", "\"\"");
        }
        return result;

    }
}
