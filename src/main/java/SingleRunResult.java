import java.util.Map;

import weka.classifiers.AbstractClassifier;

public class SingleRunResult {

    private int probe;
    private Class<? extends AbstractClassifier> classifier;
    private double meanSquareError;
    private Map<Class<? extends AbstractClassifier>,Double> ensembleMeanSquareErrors;

    public SingleRunResult(int probe,
            Class<? extends AbstractClassifier> classifier,double meanSquareError, Map<Class<? extends AbstractClassifier>, Double> ensembleMeanSquareErrors) {
        this.meanSquareError = meanSquareError;
        this.ensembleMeanSquareErrors = ensembleMeanSquareErrors;
        this.classifier = classifier;
        this.probe = probe;
    }

    public double getMeanSquareError() {
        return meanSquareError;
    }

    public Map<Class<? extends AbstractClassifier>, Double> getEnsembleMeanSquareError() {
        return ensembleMeanSquareErrors;
    }

    public int getProbe() {
        return probe;
    }

    public Class<? extends AbstractClassifier> getClassifier() {
        return classifier;
    }
}
