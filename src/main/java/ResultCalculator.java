import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Stacking;
import weka.core.Instance;
import weka.core.Instances;

public class ResultCalculator {

    private Instances learningData;
    private Instances testData;
    private int workers;
    private int dataSetCount;
    private ResultSerializer resultSerializer;
    private static final int TEST_DATA_COUNT = 50;

    public ResultCalculator(String destinationFile, Instances learningData, Instances testData, int workers, int dataSetCount) {
        this.learningData = learningData;
        this.testData = testData;
        this.workers = workers;
        this.resultSerializer = new ResultSerializer(destinationFile);
        this.dataSetCount = dataSetCount;
    }

    public void calculateMetrics() {

        ExecutorService executor = Executors.newFixedThreadPool(workers);
        Set<SingleRunCalculator> singleRuns = new HashSet<SingleRunCalculator>();
        Set<EnsembleClassifierBuildCallable> ensembleRuns = new HashSet<EnsembleClassifierBuildCallable>();
        Set<SingleRunCalculator> runCalculators = new HashSet<SingleRunCalculator>();
        Set<SingleClassifierBuildCallable> singleClassifierBuildCallables = new HashSet<SingleClassifierBuildCallable>();

        Set<AbstractClassifier> singleClassifiers = new HashSet<AbstractClassifier>();
        try {
            for (Class<? extends AbstractClassifier> classifier : Configuration.getSingleClassifiers()) {
                AbstractClassifier classifierInstance = classifier.newInstance();
                singleClassifierBuildCallables.add(new SingleClassifierBuildCallable(classifierInstance, learningData));
            }

            // build all simple classifiers
            List<Future<AbstractClassifier>> allSingleClassifiers = executor.invokeAll(singleClassifierBuildCallables);
            for (Future<AbstractClassifier> builtClassifier : allSingleClassifiers) {
                singleClassifiers.add(builtClassifier.get());
            }

            // build all ensemble classifiers
            for (AbstractClassifier singleClassifier : singleClassifiers) {
                for (Class<? extends AbstractClassifier> classifier : Configuration.getEnsembleClassifiers()) {
                    ensembleRuns.add(new EnsembleClassifierBuildCallable(singleClassifier, classifier, learningData));
                }
            }
            List<Future<AbstractClassifier>> ensembleRunResults = executor.invokeAll(ensembleRuns);
            Set<AbstractClassifier> allEnsembleClassifiers = new HashSet<AbstractClassifier>();
            for (Future<AbstractClassifier> builtClassifier : ensembleRunResults) {
                allEnsembleClassifiers.add(builtClassifier.get());
            }

            // test data
            List<Instances> testDataInstances = new ArrayList<Instances>();
            for (int i = 0; i < dataSetCount; i++) {
                Instances instances = new Instances(testData);
                instances.delete();
                Set<Integer> selectedInstances = new HashSet<Integer>();
                while (instances.size() < TEST_DATA_COUNT) {
                    Random generator = new Random();
                    Integer random = generator.nextInt(TEST_DATA_COUNT);
                    if (!selectedInstances.contains(random)) {
                        selectedInstances.add(random);
                        instances.add(testData.get(random));
                    }
                }
                testDataInstances.add(instances);
            }

            for (AbstractClassifier simpleClassifier : singleClassifiers) {
                int counter = 0;
                for (Instances testInstance : testDataInstances) {
                    counter++;
                    runCalculators.add(new SingleRunCalculator(counter, simpleClassifier, allEnsembleClassifiers, testInstance));
                }
            }
            List<Future<SingleRunResult>> singleRunCalculatorsResult = executor.invokeAll(runCalculators);


            // serialize results
                resultSerializer.serialize(singleRunCalculatorsResult);

        } catch (InterruptedException iE) {
            System.out.println("ups, sorry Ciero ... Interrupted");
            iE.printStackTrace();
        } catch (Exception e) {
            System.out.println("ups, sorry Ciero ... dupa, nie wiem");
            e.printStackTrace();
        }

    }

    private class SingleClassifierBuildCallable implements Callable<AbstractClassifier> {

        private AbstractClassifier classifier;
        private Instances learningData;
        private CVParameterSelection parameterSelector;

        public SingleClassifierBuildCallable(AbstractClassifier classifier, Instances learningData) {
            this.classifier = classifier;
            this.learningData = learningData;
            AbstractClassifier toRunSingleClassifier = classifier;
            parameterSelector = new CVParameterSelection();
        }

        public AbstractClassifier call() throws Exception {
            parameterSelector.setClassifier(classifier);
            parameterSelector.buildClassifier(learningData);
            String[] options = parameterSelector.getBestClassifierOptions();
            classifier.setOptions(options);
            classifier.buildClassifier(learningData);
            return this.classifier;
        }
    }

    private class EnsembleClassifierBuildCallable implements Callable<AbstractClassifier> {

        private AbstractClassifier simpleClassifier;
        private AbstractClassifier ensembleClassifier;
        private Instances learningData;
        private CVParameterSelection parameterSelector;

        public EnsembleClassifierBuildCallable(
                AbstractClassifier simpleClassifier,
                Class<? extends AbstractClassifier> ensembleClassifier,
                Instances learningData) throws IllegalAccessException, InstantiationException {
            this.simpleClassifier = simpleClassifier;
            this.ensembleClassifier = ensembleClassifier.newInstance();
            this.learningData = learningData;
            this.parameterSelector = new CVParameterSelection();
        }

        public AbstractClassifier call() throws Exception {
            if (ensembleClassifier instanceof Stacking) {
                Classifier[] classifiers = new Classifier[1];
                classifiers[0] = simpleClassifier;
                ((Stacking) ensembleClassifier).setClassifiers(classifiers);
            } else {
                ((SingleClassifierEnhancer) ensembleClassifier).setClassifier(simpleClassifier);
            }
            parameterSelector.setClassifier(ensembleClassifier);
            parameterSelector.buildClassifier(learningData);
            String[] options = parameterSelector.getBestClassifierOptions();
            ensembleClassifier.setOptions(options);
            ensembleClassifier.buildClassifier(learningData);
            return this.ensembleClassifier;
        }
    }

    private class SingleRunCalculator implements Callable<SingleRunResult> {

        private AbstractClassifier classifier;
        private Set<AbstractClassifier> ensembleClassifiers;
        private Instances testData;
        private int probe;

        public SingleRunCalculator(int probe,
                AbstractClassifier simpleClassifier,
                Set<AbstractClassifier> ensembleClassifiers,
                Instances testData) throws Exception {
            this.probe = probe;
            this.classifier = simpleClassifier;
            this.ensembleClassifiers = ensembleClassifiers;
            this.testData = testData;
        }

        public SingleRunResult call() throws Exception {
            Double simpleClassifierError = getMeanSquaredError(classifier);
            Map<Class<? extends AbstractClassifier>, Double> ensembleMeanSquareErrors = new HashMap<Class<? extends AbstractClassifier>, Double>();

            for (AbstractClassifier ensembleClassifier : ensembleClassifiers) {
                ensembleMeanSquareErrors.put(ensembleClassifier.getClass(), getMeanSquaredError(ensembleClassifier));
            }

            return new SingleRunResult(probe,classifier.getClass(), simpleClassifierError, ensembleMeanSquareErrors);
        }

        private Double getMeanSquaredError(AbstractClassifier classifier) throws Exception {
            Double error = 0D;
            for (Instance instance : testData) {
                Double result = classifier.classifyInstance(instance);
                Double realValue = instance.value(instance.numAttributes() - 1);
                error += Math.pow(realValue - result, 2);
            }
            error = error / testData.size();
            return Math.sqrt(error);
        }
    }
}
