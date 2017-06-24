import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import weka.classifiers.AbstractClassifier;

public class ResultSerializer {

    private static final char DEFAULT_SEPARATOR = ',';
    private String outputFileName;

    public ResultSerializer(String outputFileName) {
        this.outputFileName = outputFileName;
    }

    public void result(Map<String, Map<Class, Map<Class, Double>>> toSerialize) throws IOException, ExecutionException, InterruptedException {

        FileWriter writer = new FileWriter(outputFileName);

        List<String> header = new LinkedList<String>();
        header.add("classifier");
        header.add("probe");
        header.add("single classifier RMSE");
        for(Class c : Configuration.getEnsembleClassifiers()){
            header.add(c.getSimpleName() + " RMSE");
        }
        //write header row:
        writeLine(writer,header,DEFAULT_SEPARATOR,' ');

        for(String souirceName : toSerialize.keySet()){
            for(Class simpleClass : toSerialize.get(souirceName).keySet()){
                List<String> row = new LinkedList<String>();
                row.add(simpleClass.getSimpleName());
                row.add(souirceName);
                row.add(toSerialize.get(souirceName).get(simpleClass).get(simpleClass).toString());
                for(Class e : Configuration.getEnsembleClassifiers()){
                    row.add(toSerialize.get(souirceName).get(simpleClass).get(e).toString());
                }
                writeLine(writer, row, DEFAULT_SEPARATOR, ' ');
            }
        }
        writer.flush();
        writer.close();
    }

   public  void serializeReferenceIterations(Map<Class<? extends AbstractClassifier>, Map<Integer,Double>> referenceMap)
           throws IOException {
       File f = new File(outputFileName);
       File dir = f.getParentFile();
       String referenceFileName = dir.getPath() + "/reference.csv";

        FileWriter writer = new FileWriter(referenceFileName);

       List<String> row = new LinkedList<String>();
       row.add("classifier");
       row.add("iterations");
       row.add("rmse");

       //write header row:
       writeLine(writer,row,DEFAULT_SEPARATOR,' ');

       for (Class clazz : referenceMap.keySet()) {
           for(Integer i:referenceMap.get(clazz).keySet()) {
               row = new LinkedList<String>();
               row.add(clazz.getSimpleName());
               row.add(i.toString());
               row.add(referenceMap.get(clazz).get(i).toString());
               writeLine(writer, row, DEFAULT_SEPARATOR, ' ');
           }
       }
       writer.flush();
       writer.close();
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
