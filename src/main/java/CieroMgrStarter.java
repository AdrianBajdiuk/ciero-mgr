import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class CieroMgrStarter {

    public static void main(String[] args){
        Option inputFile = new Option("i","input folder");
        Option outputFile = new Option("o", "output folder");
        Options options = new Options();
        options.addOption(inputFile);
        options.addOption(outputFile);
        DefaultParser parser = new DefaultParser();
        try {
            CommandLine cli = parser.parse(options, args);
        } catch (ParseException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.out.println("ciero start!");
    }
}
