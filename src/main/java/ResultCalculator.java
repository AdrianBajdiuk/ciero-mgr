import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;

public class ResultCalculator {

    private Set<AbstractClassifier>  classifiers;
    private int workers;

    public void calculateMetrics(ResultSerializer resultSerializer){

        ExecutorService executor = Executors.newFixedThreadPool(workers);
        Set<SingleRunCalculator> singleRuns = new HashSet<SingleRunCalculator>();
        for (AbstractClassifier classifier : classifiers) {
            singleRuns.add(new SingleRunCalculator(classifier));
        }
        try {
            List<Future<SingleRunResult>> singleRunResults = executor.invokeAll(singleRuns);
        } catch (InterruptedException e) {
            System.out.println("ups, sorry Ciero ...");
            e.printStackTrace();
        }

    }

    private class SingleRunCalculator implements Callable<SingleRunResult>{

        private AbstractClassifier classifier;

        public SingleRunCalculator(AbstractClassifier classifier) {
            this.classifier = classifier;
        }

        public SingleRunResult call() throws Exception {
            return null;
        }
    }
}
