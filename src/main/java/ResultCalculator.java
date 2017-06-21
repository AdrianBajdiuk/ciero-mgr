import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;

public class ResultCalculator {

    private final String destination;
    private final Instances referenceInstances;
    private final Map<String, Instances> instances;
    private ResultSerializer resultSerializer;


    public ResultCalculator(
            String destination, Instances referenceInstances, Map<String, Instances> instancesSet) {
        this.destination = destination;
        this.referenceInstances = referenceInstances;
        this.instances = instancesSet;
        this.resultSerializer = new ResultSerializer(destination);
    }

    public void calculateMetrics() {


        ExecutorService executor = Executors.newFixedThreadPool(Configuration.executorWorkers);
        Set<SingleRunCalculator> singleRuns = new HashSet<SingleRunCalculator>();
        Set<EnsembleClassifierBuildCallable> ensembleRuns = new HashSet<EnsembleClassifierBuildCallable>();
        Set<SingleRunCalculator> runCalculators = new HashSet<SingleRunCalculator>();
        Set<SingleClassifierBuildCallable> singleClassifierBuildCallables = new HashSet<SingleClassifierBuildCallable>();

        try {

            //obtain reference ensemble iterators param:
            Map<Class<? extends AbstractClassifier>, Integer> ensembleClassifiersClassForReference = new HashMap();
            Map<Class<? extends AbstractClassifier>, Map<Integer,Double>> ensembleReferenceIteratorsResult = new HashMap();

            AbstractClassifier referenceClassifier = Configuration.getReferenceClassifier().newInstance();
            CVParameterSelection parameterSelection = new CVParameterSelection();
            referenceClassifier.buildClassifier(referenceInstances);
            parameterSelection.setClassifier(referenceClassifier);
            parameterSelection.buildClassifier(referenceInstances);
            referenceClassifier.setOptions(parameterSelection.getBestClassifierOptions());
            referenceClassifier.buildClassifier(referenceInstances);
            Set<GetReferenceEnsembleIterationsCallable> iteratorsCallables = new HashSet<GetReferenceEnsembleIterationsCallable>();

            for(int i= Configuration.iteratorsStartingValue; i<Configuration.iteratorsEndingValue; i=i+Configuration.iteratorsValueStep){
                for(Class<? extends AbstractClassifier> ensebmleClassifierClass:Configuration.getEnsembleClassifiers()){
                    iteratorsCallables.add(new GetReferenceEnsembleIterationsCallable(referenceClassifier, referenceInstances, i,ensebmleClassifierClass ));
                }
            }

            List<Future<GetReferenceEnsembleIterationsGet>> iterators = executor.invokeAll(iteratorsCallables);

            for(Future<GetReferenceEnsembleIterationsGet> iterator : iterators){
                GetReferenceEnsembleIterationsGet tempGet = iterator.get();
                Class<? extends AbstractClassifier> ensembleClass = tempGet.getEnsembleClassifier();
                Integer i = tempGet.getEnsembleItarations();
                Double rmse = tempGet.getRMSE();
                if(ensembleClassifiersClassForReference.containsKey(ensembleClass)){
                    Map<Integer,Double> values = ensembleReferenceIteratorsResult.get(ensembleClass);
                    boolean bestTillNow = false;
                    for(Integer val : values.keySet()) {
                        if (rmse < values.get(val)){
                            bestTillNow = true;
                        }
                    }
                    if (bestTillNow){
                        ensembleClassifiersClassForReference.put(ensembleClass,i);
                    }
                } else {
                    ensembleClassifiersClassForReference.put(ensembleClass,i);
                }
                if(ensembleReferenceIteratorsResult.containsKey(ensembleClass)){
                    ensembleReferenceIteratorsResult.get(ensembleClass).put(i,rmse);
                } else {
                    Map<Integer, Double> t = new HashMap<Integer, Double>();
                    t.put(i,rmse);
                    ensembleReferenceIteratorsResult.put(ensembleClass,t);
                }

            }

            resultSerializer.serializeReferenceIterations(ensembleReferenceIteratorsResult);
            //done with obtaining references for every ensembleclassifierclass!
            Map<String,Set<AbstractClassifier>> singleClassifiers = new HashMap<String, Set<AbstractClassifier>>();

            for (Class<? extends AbstractClassifier> classifier : Configuration.getSingleClassifiers()) {
                for (String sourceName : instances.keySet()) {
                    singleClassifierBuildCallables.add(new SingleClassifierBuildCallable(classifier,sourceName,instances.get(sourceName) ));
                }
            }

            // build all simple classifiers
            List<Future<SingleClassifierBuildGet>> allSingleClassifiers = executor.invokeAll(singleClassifierBuildCallables);
            for (Future<SingleClassifierBuildGet> builtClassifier : allSingleClassifiers) {
                SingleClassifierBuildGet get = builtClassifier.get();
                if(!singleClassifiers.containsKey(get.getSourceName())){
                    singleClassifiers.put(get.getSourceName(),new HashSet<AbstractClassifier>());
                }
                singleClassifiers.get(get.getSourceName()).add(get.getSimpleClassifier());
            }

            // build all ensemble classifiers
            for (String sourceName : singleClassifiers.keySet()) {
                for(AbstractClassifier simple : singleClassifiers.get(sourceName)) {
                    for (Class<? extends AbstractClassifier> clazz : Configuration.getEnsembleClassifiers()) {
                        ensembleRuns.add(new EnsembleClassifierBuildCallable(
                                simple,
                                clazz,
                                sourceName,
                                ensembleClassifiersClassForReference.get(clazz),
                                instances.get(sourceName)));
                    }
                }
            }
            List<Future<EnsembleClassifierBuildGet>> ensembleRunResults = executor
                    .invokeAll(ensembleRuns);
            Map<String, Map<AbstractClassifier,Set<AbstractClassifier>>> allEnsembleClassifiers = new HashMap<String, Map<AbstractClassifier, Set<AbstractClassifier>>>();

            for (Future<EnsembleClassifierBuildGet> builtClassifier : ensembleRunResults) {
                EnsembleClassifierBuildGet t = builtClassifier.get();
                String sourceName = t.getSourceName();
                AbstractClassifier simpleClassifier = t.getSimpleClassifier();
                AbstractClassifier ensembleClassifier = t.getEnsembleClassifier();
                if (!allEnsembleClassifiers.containsKey(sourceName)) {
                    allEnsembleClassifiers.put(sourceName, new HashMap<AbstractClassifier, Set<AbstractClassifier>>());
                }
                Map<AbstractClassifier,Set<AbstractClassifier>> singleEnsembleMap = allEnsembleClassifiers.get(sourceName);
                if(! singleEnsembleMap.containsKey(simpleClassifier)){
                    singleEnsembleMap.put(simpleClassifier,new HashSet<AbstractClassifier>());
                }
                singleEnsembleMap.get(simpleClassifier).add(ensembleClassifier);

            }

            for (String sourceName : allEnsembleClassifiers.keySet()) {
                for (AbstractClassifier single : allEnsembleClassifiers.get(sourceName).keySet()) {
                    runCalculators.add(
                            new SingleRunCalculator(
                                    sourceName,
                                    single,
                                    allEnsembleClassifiers.get(sourceName).get(single),
                                    instances.get(sourceName)));
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

    private class GetReferenceEnsembleIterationsGet {
        private final Class<? extends AbstractClassifier> ensembleClassifier;
        private final Integer ensembleItarations;
        private final Double crossValidationEval;

        public GetReferenceEnsembleIterationsGet(
                Class<? extends AbstractClassifier> ensembleClassifier,
                Integer ensembleItarations,
                Double crossValidationEval) {
            this.ensembleClassifier = ensembleClassifier;
            this.ensembleItarations = ensembleItarations;
            this.crossValidationEval = crossValidationEval;
        }

        public Class<? extends AbstractClassifier> getEnsembleClassifier() {
            return ensembleClassifier;
        }

        public Integer getEnsembleItarations() {
            return ensembleItarations;
        }

        public Double getRMSE() {
            return crossValidationEval;
        }
    }

    private class GetReferenceEnsembleIterationsCallable implements Callable<GetReferenceEnsembleIterationsGet>{
        private final Class<? extends AbstractClassifier> ensembleClassifierClass;
        private final Integer ensembleIterations;
        private Instances learningData;
        private CVParameterSelection parameterSelector;
        private final AbstractClassifier referenceClassifier;

        public GetReferenceEnsembleIterationsCallable(AbstractClassifier referenceClassifier, Instances learningData, Integer ensembleIterations, Class<? extends AbstractClassifier> ensembleClassifierClass) {
            this.referenceClassifier = referenceClassifier;
            this.ensembleClassifierClass = ensembleClassifierClass;
            this.ensembleIterations = ensembleIterations;
            this.learningData = referenceInstances;
            parameterSelector = new CVParameterSelection();
        }

        public GetReferenceEnsembleIterationsGet call() throws Exception {
            IteratedSingleClassifierEnhancer ensemble = (IteratedSingleClassifierEnhancer) ensembleClassifierClass.newInstance();
            ensemble.setClassifier(referenceClassifier);
            ensemble.buildClassifier(learningData);
            parameterSelector.setClassifier(ensemble);
            parameterSelector.buildClassifier(learningData);
            String[] options = parameterSelector.getBestClassifierOptions();
            ensemble.setOptions(options);
            ensemble.setNumIterations(ensembleIterations);
            ensemble.buildClassifier(learningData);
            Evaluation eval = new Evaluation(learningData);
            eval.evaluateModel(ensemble,learningData);
            return new GetReferenceEnsembleIterationsGet(ensembleClassifierClass, ensembleIterations, eval.rootMeanSquaredError());
        }
    }
    private class SingleClassifierBuildGet {
        private final String sourceName;
        private final AbstractClassifier simpleClassifier;
        

        public SingleClassifierBuildGet(String sourceName, AbstractClassifier simpleClassifier) {
            this.sourceName = sourceName;
            this.simpleClassifier = simpleClassifier;
        }

        public String getSourceName() {
            return sourceName;
        }

        public AbstractClassifier getSimpleClassifier() {
            return simpleClassifier;
        }
    }
    private class EnsembleClassifierBuildGet extends SingleClassifierBuildGet{
        private final AbstractClassifier ensembleClassifier;


        public EnsembleClassifierBuildGet(String sourceName, AbstractClassifier simpleClassifier, AbstractClassifier ensembleClassifier) {
            super(sourceName,simpleClassifier);
            this.ensembleClassifier = ensembleClassifier;
        }

        public AbstractClassifier getEnsembleClassifier() {
            return ensembleClassifier;
        }
    }
    private class SingleClassifierBuildCallable implements Callable<SingleClassifierBuildGet> {

        private Class<? extends AbstractClassifier> classifier;
        private Instances learningData;
        private AbstractClassifier classifierInstance;
        private CVParameterSelection parameterSelector;
        private final String sourceData;

        public SingleClassifierBuildCallable(Class<? extends AbstractClassifier> classifier,String sourceData, Instances learningData) {
            this.classifier = classifier;
            this.learningData = learningData;
            this.sourceData = sourceData;
            parameterSelector = new CVParameterSelection();
        }

        public SingleClassifierBuildGet call() throws Exception {
            classifierInstance = classifier.newInstance();
            classifierInstance.buildClassifier(learningData);
            parameterSelector.setClassifier(classifierInstance);
            parameterSelector.buildClassifier(learningData);
            String[] options = parameterSelector.getBestClassifierOptions();
            classifierInstance.setOptions(options);
            classifierInstance.buildClassifier(learningData);
            return new SingleClassifierBuildGet(sourceData,classifierInstance);
        }
    }

    private class EnsembleClassifierBuildCallable implements Callable<EnsembleClassifierBuildGet> {

        private AbstractClassifier simpleClassifier;
        private AbstractClassifier ensembleClassifier;
        private Instances learningData;
        private CVParameterSelection parameterSelector;
        private final Integer interations;
        private String sourceName;

        public EnsembleClassifierBuildCallable(
                AbstractClassifier simpleClassifier,
                Class<? extends AbstractClassifier> ensembleClassifierClass,
                String sourcename,Integer iterations,
                Instances learningData) throws IllegalAccessException, InstantiationException {
            this.simpleClassifier = simpleClassifier;
            this.ensembleClassifier = ensembleClassifierClass.newInstance();
            this.learningData = learningData;
            this.sourceName = sourcename;
            this.interations = iterations;
            this.parameterSelector = new CVParameterSelection();
        }

        public EnsembleClassifierBuildGet call() throws Exception {
                IteratedSingleClassifierEnhancer ensembleClassifier = ((IteratedSingleClassifierEnhancer)this.ensembleClassifier);
                ensembleClassifier.setClassifier(simpleClassifier);
                ensembleClassifier.buildClassifier(learningData);
                parameterSelector.setClassifier(ensembleClassifier);
                parameterSelector.buildClassifier(learningData);
                String[] options = parameterSelector.getBestClassifierOptions();
                ensembleClassifier.setOptions(options);
                ensembleClassifier.setNumIterations(this.interations);
                ensembleClassifier.buildClassifier(learningData);
            return new EnsembleClassifierBuildGet(this.sourceName,simpleClassifier,ensembleClassifier);
        }
    }

    private class SingleRunCalculator implements Callable<SingleRunResult> {

        private AbstractClassifier classifier;
        private Set<AbstractClassifier> ensembleClassifiers;
        private Instances testData;
        private String probe;

        public SingleRunCalculator(
                String probe,
                AbstractClassifier simpleClassifier,
                Set<AbstractClassifier> ensembleClassifiers,
                Instances testData) throws Exception {
            this.probe = probe;
            this.classifier = simpleClassifier;
            this.ensembleClassifiers = ensembleClassifiers;
            this.testData = testData;
        }

        public SingleRunResult call() throws Exception {
            Double simpleClassifierError = getMeanSquaredError(classifier,testData);
            Map<Class<? extends AbstractClassifier>, Double> ensembleMeanSquareErrors = new HashMap<Class<? extends AbstractClassifier>, Double>();

            for (AbstractClassifier ensembleClassifier : ensembleClassifiers) {
                ensembleMeanSquareErrors.put(ensembleClassifier.getClass(), getMeanSquaredError(ensembleClassifier,testData));
            }

            return new SingleRunResult(probe, classifier.getClass(), simpleClassifierError, ensembleMeanSquareErrors);
        }

        private Double getMeanSquaredError(AbstractClassifier classifier, Instances instances) throws Exception {
            Evaluation eval = new Evaluation(instances);
            eval.evaluateModel(classifier,instances);
            return eval.rootMeanSquaredError();
        }
    }
}
