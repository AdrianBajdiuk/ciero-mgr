import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javafx.util.Pair;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;

public class ResultCalculator {

    private Set<AbstractClassifier> classifiers;
    Set<Pair<Instances,Instances>> learningToTestInstances;
    private int workers;
    CVParameterSelection parameterSelection = new CVParameterSelection();
    Class<? extends ResultSerializer> resultSerializerClass;

    public void calculateMetrics() {

        ExecutorService executor = Executors.newFixedThreadPool(workers);
        Set<SingleRunCalculator> singleRuns = new HashSet<SingleRunCalculator>();
        try {
            for (AbstractClassifier classifier : classifiers) {
                for(Pair<Instances,Instances> classificationPair: learningToTestInstances) {
                    ResultSerializer serializer = new ResultSerializer();
                    singleRuns.add(new SingleRunCalculator(classifier,classificationPair.getValue(),classificationPair.getKey(),serializer));
                }
            }
            List<Future<SingleRunResult>> singleRunResults = executor.invokeAll(singleRuns);
        }catch (IOException ioEx) {
            System.out.println("ups, sorry Ciero ... IOException");
            ioEx.printStackTrace();
        }catch (InterruptedException iE) {
            System.out.println("ups, sorry Ciero ... Interrupted");
            iE.printStackTrace();
        } catch (Exception e) {
            System.out.println("ups, sorry Ciero ... dupa, nie wiem");
            e.printStackTrace();
        }

    }

    private class SingleRunCalculator implements Callable<SingleRunResult> {

        private AbstractClassifier classifier;
        private Instances learningData;
        private Evaluation eval;
        private ResultSerializer serializer;

        public SingleRunCalculator(
                AbstractClassifier classifier,
                Instances testData,
                Instances learningData,
                ResultSerializer serializer) throws Exception {
            this.classifier = classifier;
            this.learningData = learningData;
            this.eval = new Evaluation(testData);
            this.serializer = serializer;
        }

        public SingleRunResult call() throws Exception {
            classifier.buildClassifier(learningData);
            return null;
        }
    }
}
