import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.rules.M5Rules;
import weka.classifiers.trees.M5P;
import weka.core.Instances;

public class Configuration {

    private final String referenceFile;
    private final String sourceDirectory;
    private final String destinationFile;
    private Map<String,Instances> instancesSet;
    private Instances referenceInstances;
    public  static final int executorWorkers;
    public static final int iteratorsStartingValue;
    public static final int iteratorsValueStep;
    public static final int iteratorsEndingValue;
    private static Set<Class<? extends AbstractClassifier>> singleClassifiers;
    private static Set<Class<? extends AbstractClassifier>> ensembleClassifiers;
    private static Class<? extends AbstractClassifier> referenceClassifier;
    static {
        executorWorkers = 8;
        iteratorsStartingValue = 25;
        iteratorsValueStep = 25;
        iteratorsEndingValue = 150;
        singleClassifiers = new HashSet<Class<? extends AbstractClassifier>>() {

            {
                add(M5P.class);
                add(M5Rules.class);
                add(MultilayerPerceptron.class);
                add(LinearRegression.class);
            }
        };

        ensembleClassifiers = new HashSet<Class<? extends AbstractClassifier>>() {

            {
                add(Bagging.class);
                add(AdditiveRegression.class);
                add(RandomSubSpace.class);
                add(RotationForest.class);
            }
        };
        referenceClassifier = M5Rules.class;

    }

    public Configuration(String sourceFileDirectory, String destinationFile, String referenceFile) throws IOException {
        this.sourceDirectory = sourceFileDirectory;
        this.destinationFile = destinationFile;
        this.referenceFile = referenceFile;
        loadData();
    }

    private void loadData() throws IOException {
        File sourceFile = new File(this.sourceDirectory);
        if (!(sourceFile.exists() && sourceFile.isDirectory())) {
            throw new RuntimeException("Source file must exist and be directory!");
        }
        this.instancesSet = new HashMap<String,Instances>();
        File[] sourceFiles = sourceFile.listFiles();
        for(File source : sourceFiles){
            String sourceName = source.getName();
            Instances sourceInstances = new Instances(new FileReader(source));
            sourceInstances.setClassIndex(sourceInstances.numAttributes()-1);
            instancesSet.put(sourceName,sourceInstances);
            if (source.getPath().equals(referenceFile)){
                referenceInstances = new Instances(new FileReader(source));
                referenceInstances.setClassIndex(referenceInstances.numAttributes()-1);
            }
        }
    }

    public Instances getReferenceInstances() {
        return referenceInstances;
    }

    public Map<String, Instances> getInstancesSet() {
        return instancesSet;
    }

    public String getDestination() {
        return destinationFile;
    }

    public static Set<Class<? extends AbstractClassifier>> getSingleClassifiers() {
        return singleClassifiers;
    }

    public static Set<Class<? extends AbstractClassifier>> getEnsembleClassifiers() {
        return ensembleClassifiers;
    }
    public static Class<? extends AbstractClassifier> getReferenceClassifier() {
        return referenceClassifier;
    }


}