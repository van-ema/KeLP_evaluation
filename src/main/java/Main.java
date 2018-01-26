import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.utils.Math;

/**
 * @author emanuele
 */
public class Main {

    public static void main(String[] args) throws Exception {

        String[] kernelTypes = {"poly", "tk", "comb", "comb-norm"};
        // Range for Cp and Cn to evaluate
        float[] c_range = new float[20];
        for (int i = -5; i < 16; i++)
            c_range[i + 5] = Math.pow(2, i);


        KernelParametrization kernelParametrization = new KernelParametrization();

        // Evaluate parameter for svm with polynomial kernel
        for (float i = 1; i < 5; i++) {
            for (float c : c_range) {
                float params[] = {i, c};
                Kernel kernel = kernelParametrization.getKernel("poly", params);
                kernelParametrization.evaluateParams(kernel, "poly", params);
            }
        }

        for (String type : kernelTypes)
            for (float i = 0.1f; i < 1f; i += 0.1) {
                for (float c : c_range) {
                    float params[] = {i, c};
                    Kernel kernel = kernelParametrization.getKernel(type, params);
                    kernelParametrization.evaluateParams(kernel, type, params);
                }
            }


    }
}
