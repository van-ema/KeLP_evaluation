package Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;

/**
 * @author emanuele
 */
public class FileUtils {

    public static PrintWriter openOutFile(String filename) throws FileNotFoundException {
        File f = new File(filename);

        PrintWriter out = null;
        if (f.exists() && !f.isDirectory()) {
            out = new PrintWriter(new FileOutputStream(new File(filename), true));
        } else {
            out = new PrintWriter(filename);
        }

        return out;
    }

}
