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
import weka.classifiers.Evaluation;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.core.Instances;

public class ResultCalculator {

    private final String destination;
    private final Instances referenceInstances;
    private final Map<String, Instances> instances;
    private ResultSerializer resultSerializer;

    public ResultCalculator(String destination, Instances referenceInstances, Map<String, Instances> instancesSet) {
        this.destination = destination;
        this.referenceInstances = referenceInstances;
        this.instances = instancesSet;
        this.resultSerializer = new ResultSerializer(destination);
    }

    public void calculateMetrics() {


        ExecutorService executor = Executors.newFixedThreadPool(Configuration.executorWorkers);
        Set<SingleRunCalculator> singleRuns = new HashSet<SingleRunCalculator>();
        Set<SingleRunCalculator> runCalculators = new HashSet<SingleRunCalculator>();

        try {

            //obtain reference ensemble iterators param:
            Map<Class<? extends AbstractClassifier>, Integer> ensembleClassifiersClassForReference = new HashMap();
            Map<Class<? extends AbstractClassifier>, Map<Integer,Double>> ensembleReferenceIteratorsResult = new HashMap();

            Set<GetReferenceEnsembleIterationsCallable> iteratorsCallables = new HashSet<GetReferenceEnsembleIterationsCallable>();

            for(int i= Configuration.iteratorsStartingValue; i<Configuration.iteratorsEndingValue; i=i+Configuration.iteratorsValueStep){
                for(Class<? extends AbstractClassifier> ensebmleClassifierClass:Configuration.getEnsembleClassifiers()){
                    iteratorsCallables.add(new GetReferenceEnsembleIterationsCallable(Configuration.getReferenceClassifier(), referenceInstances, i,ensebmleClassifierClass ));
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

//            resultSerializer.serializeReferenceIterations(ensembleReferenceIteratorsResult);
            //done with obtaining references for every ensembleclassifierclass!
            //build all simple classifiers && obtain RMSE
            //build all ensemble combination && obtain RMSE
            Map<String,Map<Class,Map<Class,Double>>> resultMap = new HashMap<String, Map<Class, Map<Class, Double>>>();
            for (String sourceName : instances.keySet()) {
                if(!resultMap.containsKey(sourceName)){
                    resultMap.put(sourceName, new HashMap<Class, Map<Class, Double>>());
                }
                for (Class<? extends AbstractClassifier> scClazz : Configuration.getSingleClassifiers() ) {
                    if(!resultMap.get(sourceName).containsKey(scClazz)){
                        resultMap.get(sourceName).put(scClazz,new HashMap<Class, Double>());
                        resultMap.get(sourceName).get(scClazz).put(scClazz,null);
                    }
                        runCalculators.add(new SingleRunCalculator(sourceName,
                                scClazz,
                                null,
                                instances.get(sourceName),null));
                        for(Class<? extends AbstractClassifier> ecClazz : Configuration.getEnsembleClassifiers()) {
                            resultMap.get(sourceName).get(scClazz).put(ecClazz,null);
                            runCalculators.add(new SingleRunCalculator(sourceName,
                                    scClazz,
                                    ecClazz,
                                    instances.get(sourceName),ensembleClassifiersClassForReference.get(ecClazz)));
                        }
                }
            }
            //sc,probe,single RMSE, ensemble RMSE... , Options as one string
            List<Future<SingleRunResult>> singleRunCalculatorsResult = executor.invokeAll(runCalculators);
            Map<String,Map<Class,Map<Class,String>>> toSerialize = new HashMap<String, Map<Class, Map<Class, String>>>();
            for(Future<SingleRunResult> f : singleRunCalculatorsResult) {
                SingleRunResult result = f.get();
                String sourceName = result.getSourceName();
                Class singleClass = result.getSc();
                Class ensembleClass = result.getEc();
                Double rmse = result.getMeanSquareError();
                if(ensembleClass == null) {
                    resultMap.get(sourceName).get(singleClass).put(singleClass,rmse);
                } else {
                    resultMap.get(sourceName).get(singleClass).put(ensembleClass,rmse);
                }
            }

            // result results
            resultSerializer.result(resultMap);

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

    private class GetReferenceEnsembleIterationsCallable implements Callable<GetReferenceEnsembleIterationsGet> {

        private final Class<? extends AbstractClassifier> ensembleClassifierClass;
        private final Integer ensembleIterations;
        private Instances learningData;
        private final Class<? extends AbstractClassifier> referenceClassifierClazz;
        private Random rand = new Random(191);

        public GetReferenceEnsembleIterationsCallable(
                Class<? extends AbstractClassifier> referenceClassifier,
                Instances learningData,
                Integer ensembleIterations,
                Class<? extends AbstractClassifier> ensembleClassifierClass) {
            this.referenceClassifierClazz = referenceClassifier;
            this.ensembleClassifierClass = ensembleClassifierClass;
            this.ensembleIterations = ensembleIterations;
            this.learningData = referenceInstances;
        }

        public GetReferenceEnsembleIterationsGet call() throws Exception {
            IteratedSingleClassifierEnhancer ensemble = (IteratedSingleClassifierEnhancer) ensembleClassifierClass
                    .newInstance();
            ensemble.setClassifier(referenceClassifierClazz.newInstance());
            ensemble.setNumIterations(ensembleIterations);
            Evaluation eval = new Evaluation(learningData);
            eval.crossValidateModel(ensemble,learningData,10,new Random(rand.nextInt()));
            return new GetReferenceEnsembleIterationsGet(
                    ensembleClassifierClass,
                    ensembleIterations,
                    eval.rootMeanSquaredError());
        }
    }

    private class SingleRunCalculator implements Callable<SingleRunResult> {

        private Class<? extends  AbstractClassifier> singleClassifierClazz;
        private Class<? extends  AbstractClassifier> ensembleClassifierClazz;
        private Integer ensembleClassifierIterations;
        private Instances testData;
        private String probe;
        private Random seed;

        public SingleRunCalculator(
                String probe,
                Class<? extends  AbstractClassifier> simpleClassifier,
                Class<? extends  AbstractClassifier> ensembleClassifiers,
                Instances testData, Integer iterations) throws Exception {
            this.probe = probe;
            this.singleClassifierClazz = simpleClassifier;
            this.ensembleClassifierClazz = ensembleClassifiers;
            this.testData = testData;
            this.ensembleClassifierIterations = iterations;
            seed = new Random(131);
        }

        public SingleRunResult call() throws Exception {
            Double result = 0.0d;

            if ( ensembleClassifierClazz != null) {
                //licz ensemble
                IteratedSingleClassifierEnhancer ec = (IteratedSingleClassifierEnhancer) ensembleClassifierClazz.newInstance();
                ec.setClassifier(singleClassifierClazz.newInstance());
                ec.setNumIterations(ensembleClassifierIterations);
                Evaluation eval = new Evaluation(testData);
                eval.crossValidateModel(ec,testData,10,new Random(seed.nextInt()));
                result = eval.rootMeanSquaredError();
            } else{
                //licz single
                AbstractClassifier sc = singleClassifierClazz.newInstance();
                Evaluation eval = new Evaluation(testData);
                eval.crossValidateModel(sc,testData,10,new Random(seed.nextInt()));
                result = eval.rootMeanSquaredError();
            }
            return new SingleRunResult(probe, singleClassifierClazz,ensembleClassifierClazz, result);
        }
    }
}
