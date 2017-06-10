import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.Stacking;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.trees.M5P;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.RemoveType;

public class Configuration {

    private CSVLoader csvLoader = new CSVLoader();
    private Instances learningData;
    private Instances testDataPool;
    private String sourceFileDirectory;
    private static final int START_ENSAMBLE_AGGREGATE_COUNTER = 1;
    private static final int END_ENSAMBLE_AGGREGATE_COUNTER = 10;
    private static final double TEST_DATA_PERCENTAGE = 0.3;
    private String destinationDirectory;
    private static Set<Class<? extends AbstractClassifier>> singleClassifiers;
    private static Set<Class<? extends AbstractClassifier>> ensembleClassifiers;

    static {
        singleClassifiers = new HashSet<Class<? extends AbstractClassifier>>() {

            {
                add(M5P.class);
                add(M5Rules.class);
                add(MultilayerPerceptron.class);
            }
        };

        ensembleClassifiers = new HashSet<Class<? extends AbstractClassifier>>() {

            {
                add(Bagging.class);
                add(AdditiveRegression.class);
                add(Stacking.class);
                add(RandomSubSpace.class);
                add(RotationForest.class);
//                add(RandomCommittee.class);
            }
        };

    }

    public Configuration(String sourceFileDirectory, String destinationDirectory) throws IOException {
        this.sourceFileDirectory = sourceFileDirectory;
        this.destinationDirectory = destinationDirectory;
//        csvLoader.setFieldSeparator(";");
//        csvLoader.setSource(new File(sourceFileDirectory));
//        csvLoader.setNumericAttributes("first-last");
        Instances instantes  = new Instances(new FileReader(sourceFileDirectory));
        divideTestDataPool(instantes);
    }

    private void divideTestDataPool(Instances dataSet) {

        int testDataCount = (int) Math.floor(dataSet.size() * TEST_DATA_PERCENTAGE);
        int addedTestData = 0;
        Set<Integer> testDataAdded = new HashSet<Integer>();
        Instances toLearnInstances = new Instances(dataSet);
        Instances toTestInstances = new Instances(dataSet);
        toTestInstances.delete();
        Random rand = new Random();
        while (addedTestData <= testDataCount) {
            int generatedIndex = rand.nextInt(testDataCount);
            double generateProb = rand.nextDouble();
            if (!testDataAdded.contains(generatedIndex)) {
                RemoveType removeType = new RemoveType();
                String[] options = new String[1];
                Instance instance = dataSet.get(generatedIndex);
                if (generatedIndex <= TEST_DATA_PERCENTAGE) {
                    toTestInstances.add(instance);
                    toLearnInstances.remove(generatedIndex);
                    addedTestData++;
                }
            }
        }
        this.learningData = toLearnInstances;
        this.learningData.setClassIndex(this.learningData.numAttributes()-1);
        this.testDataPool = toTestInstances;
        this.testDataPool.setClassIndex(this.testDataPool.numAttributes()-1);
    }

    public Instances getSourceData() throws IOException {
        return this.csvLoader.getDataSet();
    }

    public String getDestination() {
        return destinationDirectory;
    }

    public static Set<Class<? extends AbstractClassifier>> getSingleClassifiers() {
        return singleClassifiers;
    }

    public static Set<Class<? extends AbstractClassifier>> getEnsembleClassifiers() {
        return ensembleClassifiers;
    }

    public Instances getLearningData() {
        return learningData;
    }

    public Instances getTestData() {
        return this.testDataPool;
    }
}
