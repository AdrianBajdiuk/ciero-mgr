import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class CieroMgrStarter {

    public static void main(String[] args){
        Option inputFolder = Option.builder("i").argName("input").hasArg().required(true).build();
        Option outputFile = Option.builder("o").argName("output").hasArg().required(true).build();
        Option referenceFile = Option.builder("r").argName("referenceFile").hasArg().required(true).build();
        Options options = new Options();
        options.addOption(inputFolder);
        options.addOption(outputFile);
        options.addOption(referenceFile);
        DefaultParser parser = new DefaultParser();
        String input;
        String output;
        String reference;

        try {
            CommandLine cli  = parser.parse(options, args);
            input = cli.getOptionValue(inputFolder.getOpt());
            output =  cli.getOptionValue(outputFile.getOpt());
            reference = cli.getOptionValue(referenceFile.getOpt());
            Configuration configuration  = new Configuration(input, output, reference);
            ResultCalculator calculator = new ResultCalculator(configuration.getDestination(),configuration.getReferenceInstances(),configuration.getInstancesSet());
            calculator.calculateMetrics();

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.out.println("ciero start!");
        System.exit(0);
    }
}
