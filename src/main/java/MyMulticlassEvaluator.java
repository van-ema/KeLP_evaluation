import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.predictionfunction.Prediction;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.utils.evaluation.Evaluator;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class MyMulticlassEvaluator extends Evaluator {

    private List<Label> labelList;
    private HashMap<Label, Float> toBePredicted = new HashMap<Label, Float>();
    private HashMap<Label, Float> predicted = new HashMap<Label, Float>();
    private HashMap<Label, Float> correctPrediction = new HashMap<Label, Float>();

    private float overallPrecision;
    private float overallRecall;
    private float overallF1;
    private float accuracy;
    private int total;


    public MyMulticlassEvaluator(List<Label> labelList) {
        this.labelList = labelList;

        Iterator labelIterator = this.labelList.iterator();
        while (labelIterator.hasNext()) {
            Label l = (Label) labelIterator.next();
            this.correctPrediction.put(l, 0.0F);
            this.toBePredicted.put(l, 0.0F);
            this.predicted.put(l, 0.0F);
        }

        this.overallPrecision = 0f;
        this.overallRecall = 0f;
        this.overallF1 = 0f;
        this.accuracy = 0f;


    }

    public void addCount(Example example, Prediction prediction) {
        ClassificationOutput output = (ClassificationOutput) prediction;

        Iterator labels = example.getClassificationLabels().iterator();
        while (labels.hasNext()) {
            Label label = (Label) labels.next();
            this.total++;
            toBePredicted.put(label, toBePredicted.get(label) + 1.0f);
        }

        List<Label> predictedLabels = output.getPredictedClasses();
        for (Label label : predictedLabels) {
            predicted.put(label, predicted.get(label) + 1.0f);
            if (example.isExampleOf(label))
                correctPrediction.put(label, correctPrediction.get(label) + 1.0f);

        }

        this.computed = false;
    }

    protected void compute() {
        int correctTot = 0;
        int predictedTot = 0;
        int toBePredictedTot = 0;

        Iterator labelIterator = this.labelList.iterator();

        while (labelIterator.hasNext()) {
            Label l = (Label) labelIterator.next();
            correctTot = (int) ((float) correctTot + this.correctPrediction.get(l));
            predictedTot = (int) ((float) predictedTot + this.predicted.get(l));
            toBePredictedTot = (int) ((float) toBePredictedTot + this.toBePredicted.get(l));
        }

        this.overallPrecision = (float) correctTot / (float) predictedTot;
        this.overallRecall = (float) correctTot / (float) toBePredictedTot;
        this.overallF1 = 2.0F * this.overallPrecision * this.overallRecall / (this.overallPrecision + this.overallRecall);
        this.accuracy = (float) correctTot / (float) this.total;

        this.computed = true;
    }

    public void clear() {

        this.toBePredicted.clear();
        this.correctPrediction.clear();
        this.predicted.clear();

        this.total = 0;
        this.overallPrecision = 0f;
        this.overallRecall = 0f;
        this.overallF1 = 0f;
        this.accuracy = 0f;
    }

    public Evaluator duplicate() {
        return new MyMulticlassEvaluator(this.labelList);
    }

    public float getOverallPrecision() {
        return overallPrecision;
    }

    public float getOverallRecall() {
        return overallRecall;
    }

    public float getOverallF1() {
        return overallF1;
    }

    public float getAccuracy() {
        return accuracy;
    }

    public int getTotal() {
        return total;
    }
}
