/**
 * @author emanuele
 */
public class DataConfiguration {

    private static String trainFilePath = "./qc_data/qc_train.klp";
    private static String testFilePath = "./qc_data/qc_test.klp";
    private static String outputFilename = "./qc_data/output";
    private static int nfold = 5;

    public static String getTrainFilePath() {
        return trainFilePath;
    }

    public static void setTrainFilePath(String trainFilePath) {
        DataConfiguration.trainFilePath = trainFilePath;
    }

    public static String getTestFilePath() {
        return testFilePath;
    }

    public static void setTestFilePath(String testFilePath) {
        DataConfiguration.testFilePath = testFilePath;
    }

    public static String getOutputFilename() {
        return outputFilename;
    }

    public static void setOutputFilename(String outputFilename) {
        DataConfiguration.outputFilename = outputFilename;
    }

    public static int getNfold() {
        return nfold;
    }

    public static void setNfold(int nfold) {
        DataConfiguration.nfold = nfold;
    }
}
