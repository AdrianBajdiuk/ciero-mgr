import weka.classifiers.AbstractClassifier;

public class SingleRunResult {

    private String sourceName;
    private Class<? extends AbstractClassifier> sc;
    private Class<? extends AbstractClassifier> ec;
    private double meanSquareError;

    public SingleRunResult(
            String sourceName,
            Class<? extends AbstractClassifier> sc,
            Class<? extends AbstractClassifier> ec,
            double meanSquareError) {
        this.sourceName = sourceName;
        this.sc = sc;
        this.ec = ec;
        this.meanSquareError = meanSquareError;
    }

    public String getSourceName() {
        return sourceName;
    }

    public Class<? extends AbstractClassifier> getSc() {
        return sc;
    }

    public Class<? extends AbstractClassifier> getEc() {
        return ec;
    }

    public double getMeanSquareError() {
        return meanSquareError;
    }
}
