import gnu.trove.map.hash.TObjectFloatHashMap;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.LinearPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.utils.ExperimentUtils;
import it.uniroma2.sag.kelp.utils.FileUtils;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.Math;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.io.*;
import java.util.List;

/**
 * Evaluating kernel function parameters. Linear kernels are not allowed here cause they don't have
 * parameter to evaluate for tuning purpose.
 * Cn and Cp set to 0.
 */
public class KernelParametrization {


    private SimpleDataset testSet;
    private SimpleDataset trainingSet;

    private Kernel getKernel(String kernelType, float param) throws Exception {

        // Read the training and test dataset
        trainingSet = new SimpleDataset();
        trainingSet.populate(DataConfiguration.getTrainFilePath());
        System.out.println("The training set is made of " + trainingSet.getNumberOfExamples() + " examples.");

        testSet = new SimpleDataset();
        testSet.populate(DataConfiguration.getTestFilePath());
        System.out.println("The test set is made of " + testSet.getNumberOfExamples() + " examples.");

        // print the number of train and test examples for each class
        for (Label l : trainingSet.getClassificationLabels()) {
            System.out.println("Positive training examples for the class " + l.toString() + " "
                    + trainingSet.getNumberOfPositiveExamples(l));
            System.out.println("Negative training examples for the class  " + l.toString() + " "
                    + trainingSet.getNumberOfNegativeExamples(l));
        }

        Kernel usedKernel = null;
        // Initialize the proper kernel function
        if (kernelType.equalsIgnoreCase("poly")) {
            String vectorRepresentationName = "bow";
            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            usedKernel = new PolynomialKernel(param, linearKernel);

        } else if (kernelType.equalsIgnoreCase("tk")) {
            String treeRepresentationName = "grct";
            usedKernel = new SubSetTreeKernel(param, treeRepresentationName);

        } else if (kernelType.equalsIgnoreCase("comb")) {
            String vectorRepresentationName = "bow";
            String treeRepresentationName = "grct";
            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel tkgrct = new SubSetTreeKernel(param, treeRepresentationName);

            LinearKernelCombination combination = new LinearKernelCombination();
            combination.addKernel(1, linearKernel);
            combination.addKernel(1, tkgrct);
            usedKernel = combination;

        } else if (kernelType.equalsIgnoreCase("comb-norm")) {
            String vectorRepresentationName = "bow";
            String treeRepresentationName = "grct";

            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
            Kernel treeKernel = new SubSetTreeKernel(param, treeRepresentationName);
            Kernel normalizedTreeKernel = new NormalizationKernel(treeKernel);

            LinearKernelCombination combination = new LinearKernelCombination();
            combination.addKernel(1, normalizedLinearKernel);
            combination.addKernel(1, normalizedTreeKernel);
            combination.normalizeWeights();

            usedKernel = combination;

        } else {
            System.err.println("The specified kernel (" + kernelType + ") is not valid.");
            System.exit(0);
        }

        return usedKernel;
    }

    private List<MulticlassClassificationEvaluator> getNFoldEvaluatorForMulticlass(Kernel usedKernel, float C) {

        // calculating the size of the gram matrix to store all the examples
        int cacheSize = trainingSet.getNumberOfExamples() + testSet.getNumberOfExamples();

        // Useful to cache the values of the norm in the kernel space
        FixIndexSquaredNormCache ncache1 = new FixIndexSquaredNormCache(6000);
        usedKernel.setSquaredNormCache(ncache1);

        // storage of the kernel computations
        KernelCache cache = new FixSizeKernelCache(cacheSize);
        usedKernel.setKernelCache(cache);

        // Instantiate the SVM learning Algorithm.
        BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
        //Set the kernel
        svmSolver.setKernel(usedKernel);

        // Set C param
        svmSolver.setC(C);

        // Instantiate the multi-class classifier that apply a One-vs-All schema
        OneVsAllLearning ovaLearner = new OneVsAllLearning();
        ovaLearner.setBaseAlgorithm(svmSolver);
        ovaLearner.setLabels(trainingSet.getClassificationLabels());

        //Building the evaluation function
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
                trainingSet.getClassificationLabels());

        // Configure nFold environment
        int nfold = DataConfiguration.getNfold();
        // Validate through 5 fold cross validation and compute the accuracy
        SimpleDataset completeDataset = new SimpleDataset();
        completeDataset.addExamples(trainingSet);
        completeDataset.addExamples(testSet);

        return ExperimentUtils.nFoldCrossValidation(nfold, ovaLearner, completeDataset, evaluator);
    }

    private List<BinaryClassificationEvaluator> getNFlodEvaluatoForHumClass(Kernel usedKernel, float C) {

        StringLabel human = new StringLabel("HUM");
        System.out.println(trainingSet.getClassificationLabels());

        // calculating the size of the gram matrix to store all the examples
        int cacheSize = trainingSet.getNumberOfExamples() + testSet.getNumberOfExamples();

        // Useful to cache the values of the norm in the kernel space
        FixIndexSquaredNormCache ncache1 = new FixIndexSquaredNormCache(6000);
        usedKernel.setSquaredNormCache(ncache1);

        // storage of the kernel computations
        KernelCache cache = new FixSizeKernelCache(cacheSize);
        usedKernel.setKernelCache(cache);

        // Instantiate the SVM learning Algorithm.
        BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
        //Set the kernel
        svmSolver.setKernel(usedKernel);
        // Set C param
        svmSolver.setC(C);
        svmSolver.setLabel(human);

        //Building the evaluation function
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator(human);

        // Configure nFold environment
        int nfold = DataConfiguration.getNfold();
        // Validate through 5 fold cross validation and compute the accuracy
        SimpleDataset completeDataset = new SimpleDataset();
        completeDataset.addExamples(trainingSet);
        completeDataset.addExamples(testSet);

        return ExperimentUtils.nFoldCrossValidation(nfold, svmSolver, completeDataset, evaluator);
    }

    /**
     * @param usedKernel "poly", "tk", "comb", "comb-norm"
     * @param params     params[0]: Exponent for polynomial kernel or lambda for tree kernet
     *                   params[1]: C value for the svm
     * @throws Exception
     */
    public void evaluateParams(String kernelType, float[] params) throws Exception {
        Kernel kernel = getKernel(kernelType, params[0]);
        List<MulticlassClassificationEvaluator> nfoldEv = getNFoldEvaluatorForMulticlass(kernel, params[1]);
        String out = String.format("%s_%s.txt", DataConfiguration.getOutputFilename(), kernelType);

        PrintWriter writer = Utils.FileUtils.openOutFile(out);
        writer.append(String.format("C=%s\n", params[1]));
        writer.append(String.format("param=%s\n", params[0]));

        float[] accuracy = new float[nfoldEv.size()];
        float[] precisions = new float[nfoldEv.size()];
        float[] recalls = new float[nfoldEv.size()];
        float[] f1s = new float[nfoldEv.size()];
        for (int i = 0; i < DataConfiguration.getNfold(); i++) {
            accuracy[i] = nfoldEv.get(i).getAccuracy();
            precisions[i] = nfoldEv.get(i).getOverallPrecision();
            recalls[i] = nfoldEv.get(i).getOverallRecall();
            f1s[i] = nfoldEv.get(i).getOverallF1();
        }

        writer.append(String.format("accuracy_mean=%s\n", Math.getMean(accuracy)));
        writer.append(String.format("accuracy_std=%s\n", Math.getStandardDeviation(accuracy)));
        writer.append(String.format("precision=%s\n", Math.getMean(precisions)));
        writer.append(String.format("recall=%s\n", Math.getMean(recalls)));
        writer.append(String.format("F1=%s\n", Math.getMean(f1s)));
        writer.append("\n");

        writer.close();
    }

    public void evaluateParamsForHumClass(String kernelType, float[] params) throws Exception {
        Kernel kernel = getKernel(kernelType, params[0]);
        List<BinaryClassificationEvaluator> nfoldEv = getNFlodEvaluatoForHumClass(kernel, params[1]);

        String out = String.format("%s_%s_hum.txt", DataConfiguration.getOutputFilename(), kernelType);

        PrintWriter writer = Utils.FileUtils.openOutFile(out);
        writer.append(String.format("C=%s\n", params[1]));
        writer.append(String.format("param=%s\n", params[0]));

        float[] f1s = new float[nfoldEv.size()];
        for (int i = 0; i < DataConfiguration.getNfold(); i++) {
            f1s[i] = nfoldEv.get(i).getF1();
        }

        writer.append(String.format("F1=%s\n", Math.getMean(f1s)));
        writer.append("\n");

        writer.close();
    }

}
