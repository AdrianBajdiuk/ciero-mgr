import weka.core.Instances;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

public class CieroMgrStarter {

    public static void main(String[] args){
        Option inputFile = Option.builder("i").argName("input").hasArg().required(true).build();
        Option outputFile = Option.builder("o").argName("output").hasArg().required(true).build();
        Options options = new Options();
        options.addOption(inputFile);
        options.addOption(outputFile);
        DefaultParser parser = new DefaultParser();
        String inputFolder;
        String outputFolder;

        try {
            CommandLine cli  = parser.parse(options, args);
            inputFolder = cli.getOptionValue(inputFile.getOpt());
            outputFolder =  cli.getOptionValue(outputFile.getOpt());
            Configuration configuration  = new Configuration(inputFolder, outputFolder);
            Instances testDataPool = configuration.getTestData();
            Instances learningData = configuration.getLearningData();
            int workers = 1;
            int probes = 2;

            ResultCalculator calculator = new ResultCalculator(configuration.getDestination(),configuration.getLearningData(),configuration.getTestData(),workers,probes);
            calculator.calculateMetrics();

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        System.out.println("ciero start!");
    }
}
