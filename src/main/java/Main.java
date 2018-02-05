import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.utils.Math;

/**
 * @author emanuele
 */
public class Main {

    public static void main(String[] args) throws Exception {

        // Range for Cp and Cn to evaluate
        float[] c_range = new float[18];
        for (int i = -7; i < 11; i++)
            c_range[i + 7] = Math.pow(2, i);


        KernelParametrization kernelParametrization = new KernelParametrization();

        // Evaluate parameter for svm with polynomial kernel
        for (float i = 1; i < 5; i++) {
            for (float c : c_range) {
                float params[] = {i, c};
                kernelParametrization.evaluateParamsForHumClass("poly", params);
            }
        }

        // Evaluate parameters for svm with tree kernel
        for (float i = 0.1f; i < 1f; i += 0.1) {
            for (float c : c_range) {
                float params[] = {i, c};
                kernelParametrization.evaluateParamsForHumClass("tk", params);
            }
        }


    }
}
