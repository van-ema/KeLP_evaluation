import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.util.Scanner;

public class WmIRQuestionClassificationExample {
    public static void main(String[] args) throws Exception {

//        if (args.length != 4) {
//            System.err
//                    .println("Usage: training_set_path test_set_path kernel[lin| poly | tk | comb | comb-norm] c_svm");
//            return;
//        }

        Scanner input = new Scanner(System.in);

        // Reading the input parameters
        String trainingSetFilePath = "/home/emanuele/workdir/KeLP_ex/qc_data/qc_train.klp";
        String testsetFilePath = "/home/emanuele/workdir/KeLP_ex/qc_data/qc_test.klp";
        System.out.print("insert kernel type: ");
        String kernelType = input.nextLine();
        System.out.print("insert C param: ");
        float c = input.nextFloat();

        // Read the training and test dataset
        SimpleDataset trainingSet = new SimpleDataset();
        trainingSet.populate(trainingSetFilePath);
        System.out.println("The training set is made of " + trainingSet.getNumberOfExamples() + " examples.");

        SimpleDataset testSet = new SimpleDataset();
        testSet.populate(testsetFilePath);
        System.out.println("The test set is made of " + testSet.getNumberOfExamples() + " examples.");

        // print the number of train and test examples for each class
        for (Label l : trainingSet.getClassificationLabels()) {
            System.out.println("Positive training examples for the class " + l.toString() + " "
                    + trainingSet.getNumberOfPositiveExamples(l));
            System.out.println("Negative training examples for the class  " + l.toString() + " "
                    + trainingSet.getNumberOfNegativeExamples(l));
        }

        // calculating the size of the gram matrix to store all the examples
        int cacheSize = trainingSet.getNumberOfExamples() + testSet.getNumberOfExamples();

        // Initialize the proper kernel function
        Kernel usedKernel = null;
        if (kernelType.equalsIgnoreCase("lin")) {
            String vectorRepresentationName = "bow";
            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            usedKernel = linearKernel;
        } else if (kernelType.equalsIgnoreCase("poly")) {
            String vectorRepresentationName = "bow";
            int exponent = 2;
            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel polynomialKernel = new PolynomialKernel(exponent, linearKernel);
            usedKernel = polynomialKernel;
        } else if (kernelType.equalsIgnoreCase("tk")) {
            String treeRepresentationName = "grct";
            float lambda = 0.4f;
            Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);
            usedKernel = tkgrct;
        } else if (kernelType.equalsIgnoreCase("comb")) {
            String vectorRepresentationName = "bow";
            String treeRepresentationName = "grct";
            float lambda = 0.4f;

            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);

            LinearKernelCombination combination = new LinearKernelCombination();
            combination.addKernel(1, linearKernel);
            combination.addKernel(1, tkgrct);
            usedKernel = combination;
        } else if (kernelType.equalsIgnoreCase("comb-norm")) {
            String vectorRepresentationName = "bow";
            String treeRepresentationName = "grct";
            float lambda = 0.4f;

            Kernel linearKernel = new LinearKernel(vectorRepresentationName);
            Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
            Kernel treeKernel = new SubSetTreeKernel(lambda, treeRepresentationName);
            Kernel normalizedTreeKernel = new NormalizationKernel(treeKernel);

            LinearKernelCombination combination = new LinearKernelCombination();
            combination.addKernel(1, normalizedLinearKernel);
            combination.addKernel(1, normalizedTreeKernel);
            usedKernel = combination;
        } else {
            System.err.println("The specified kernel (" + kernelType + ") is not valid.");
        }

        // Setting the cache to speed up the computations
        KernelCache cache = new FixIndexKernelCache(cacheSize);
        usedKernel.setKernelCache(cache);

        // Instantiate the SVM learning Algorithm.
        BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
        //Set the kernel
        svmSolver.setKernel(usedKernel);
        //Set the C parameter
        svmSolver.setCn(c);
        svmSolver.setCp(c);

        // Instantiate the multi-class classifier that apply a One-vs-All schema
        OneVsAllLearning ovaLearner = new OneVsAllLearning();
        ovaLearner.setBaseAlgorithm(svmSolver);
        ovaLearner.setLabels(trainingSet.getClassificationLabels());
        // Writing the learning algorithm and the kernel to file
        JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
        serializer.writeValueOnFile(ovaLearner, "ova_learning_algorithm.klp");

        //Learn and get the prediction function
        ovaLearner.learn(trainingSet);
        //Selecting the prediction function
        Classifier classifier = ovaLearner.getPredictionFunction();
        // Write the model (aka the Classifier for further use)
        serializer.writeValueOnFile(classifier, "model_kernel-" + kernelType + "_cp" + c + "_cn" + c + ".klp");

        //Building the evaluation function
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
                trainingSet.getClassificationLabels());

        MyMulticlassEvaluator myMulticlassEvaluator = new MyMulticlassEvaluator(
                trainingSet.getClassificationLabels());

        // Classify examples and compute the accuracy
        for (Example e : testSet.getExamples()) {
            // Predict the class
            ClassificationOutput p = classifier.predict(e);
            evaluator.addCount(e, p);
            myMulticlassEvaluator.addCount(e, p);
//            System.out.println("Question:\t" + e.getRepresentation("quest"));
//            System.out.println("Original class:\t" + e.getClassificationLabels());
//            System.out.println("Predicted class:\t" + p.getPredictedClasses());
//            System.out.println();
        }

        System.out.println("Accuracy: " + evaluator.getAccuracy());
        System.out.println("MyAccuracy: " + myMulticlassEvaluator.getAccuracy());

        System.out.println("Precision: " + evaluator.getOverallPrecision());
        System.out.println("MyPrecision: " + myMulticlassEvaluator.getOverallPrecision());

        System.out.println("Recall: " + evaluator.getOverallRecall());
        System.out.println("MyRecall: " + myMulticlassEvaluator.getOverallRecall());

        System.out.println("F1: " + evaluator.getOverallF1());
        System.out.println("MyF1: " + myMulticlassEvaluator.getOverallF1());

    }
}
