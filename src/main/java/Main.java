/**
 * @author emanuele
 */
public class Main {

    public static void main(String[] args) throws Exception {

        String[] kernelTypes = {"poly", "tk", "comb", "comb-norm"};

        KernelParametrization kernelParametrization = new KernelParametrization();
        for (String kernelType : kernelTypes)
            kernelParametrization.execute(kernelType);

    }
}
