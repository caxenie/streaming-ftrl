package org.streamingml.experiments;

import org.ejml.data.DMatrixRMaj;
import org.ejml.ops.MatrixIO;
import org.ejml.simple.SimpleMatrix;

import java.time.Duration;
import java.time.Instant;


public class Main{

    private static void swapLabelSort(double arr[], int i, int j) {
        double a;
        a = arr[i];
        arr[i] = arr[j];
        arr[j] = a;
    }
    
    private static void labelsSort(double[] arr, double[] brr, int n) {
        int jStack = - 1;
        int l = 0;
        int nStack = 64;
        int dim = 7;
        int[] iStack = new int[nStack];
        int ir = n - 1;

        int i, j, k;
        double a, b;
        for (; ; ) {
            if (ir - l < dim) {
                for (j = l + 1; j <= ir; j++) {
                    a = arr[j];
                    b = brr[j];
                    for (i = j - 1; i >= l; i--) {
                        if (arr[i] <= a) {
                            break;
                        }
                        arr[i + 1] = arr[i];
                        brr[i + 1] = brr[i];
                    }
                    arr[i + 1] = a;
                    brr[i + 1] = b;
                }
                if (jStack < 0) {
                    break;
                }
                ir = iStack[jStack--];
                l = iStack[jStack--];
            } else {
                k = (l + ir) >> 1;
                swapLabelSort(arr, k, l + 1);
                swapLabelSort(brr, k, l + 1);
                if (arr[l] > arr[ir]) {
                    swapLabelSort(arr, l, ir);
                    swapLabelSort(brr, l, ir);
                }
                if (arr[l + 1] > arr[ir]) {
                    swapLabelSort(arr, l + 1, ir);
                    swapLabelSort(brr, l + 1, ir);
                }
                if (arr[l] > arr[l + 1]) {
                    swapLabelSort(arr, l, l + 1);
                    swapLabelSort(brr, l, l + 1);
                }
                i = l + 1;
                j = ir;
                a = arr[l + 1];
                b = brr[l + 1];
                for (; ; ) {
                    do {
                        i++;
                    } while (arr[i] < a);
                    do {
                        j--;
                    } while (arr[j] > a);
                    if (j < i) {
                        break;
                    }
                    swapLabelSort(arr, i, j);
                    swapLabelSort(brr, i, j);
                }
                arr[l + 1] = arr[j];
                arr[j] = a;
                brr[l + 1] = brr[j];
                brr[j] = b;
                jStack += 2;

                if (jStack >= nStack) {
                    throw new IllegalStateException("nStack too small in sort.");
                }

                if (ir - i + 1 >= j - l) {
                    iStack[jStack] = ir;
                    iStack[jStack - 1] = i;
                    ir = j - 1;
                } else {
                    iStack[jStack] = j - 1;
                    iStack[jStack - 1] = l;
                    l = i;
                }
            }
        }
    }

    // In statistics, a receiver operating characteristic (ROC), or ROC curve,
    // is a graphical plot that illustrates the performance of a binary classifier
    // system as its discrimination threshold is varied. The curve is created by
    // plotting the true positive rate (TPR) against the false positive rate (FPR)
    // at various threshold settings.

    // AUC is quite noisy as a classification measure and has some other
    // significant problems in model comparison.
    private static double computeFtrlAUC(SimpleMatrix prediction, SimpleMatrix truth) {

        // for large sample size, overflow may happen for pos * neg.
        // switch to double to prevent it.
        double pos = 0;
        double neg = 0;

        for (int i = 0; i < truth.getNumElements(); i++) {
            if (truth.get(i) == 0.0) {
                neg++;
            } else if (truth.get(i) == 1.0) {
                pos++;
            } else {
                throw
                        new IllegalArgumentException("AUC is only for binary classification. Invalid label: " +
                                truth.get(i));
            }
        }

        SimpleMatrix labels = truth.copy();
        SimpleMatrix outputs = prediction.copy();

        // sort simultaneously the two vectors
        labelsSort(outputs.getDDRM().data, labels.getDDRM().data, outputs.getDDRM().data.length);

        // calculate the rank of the values in the prediction and true labels
        double[] rank = new double[labels.getNumElements()];
        for (int i = 0; i < outputs.getNumElements(); i++) {
            if (i == outputs.getNumElements() - 1 || outputs.get(i) != outputs.get(i+1)) {
                rank[i] = i + 1;
            } else {
                int j = i + 1;
                for (; j < outputs.getNumElements() && outputs.get(j) == outputs.get(i); j++);
                double r = (i + 1 + j) / 2.0;
                for (int k = i; k < j; k++) rank[k] = r;
                i = j - 1;
            }
        }

        double auc = 0.0;
        for (int i = 0; i < labels.getNumElements(); i++) {
            if (labels.get(i) == 1.0)
                auc += rank[i];
        }

        auc = (auc - (pos * (pos+1) / 2.0)) / (pos * neg);
        return auc;
    }

    public static void main(String[] args) throws Exception {

        ClassLoader classLoader = Main.class.getClassLoader();

        // H. B. McMahan, G. Holt, D. Sculley, et al. "Ad click prediction: a view from the trenches".
        // In: _The 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining,
        // KDD 2013, Chicago, IL, USA, August 11-14, 2013_. Ed. by I. S.Dhillon, Y. Koren, R. Ghani,
        // T. E. Senator, P. Bradley, R. Parekh, J. He, R. L. Grossman and R. Uthurusamy. ACM, 2013, pp. 1222-1230.
        // The data is from the paper repo: iPinYou Real-Time Bidding Dataset for Computational Advertising Research
        //
        // This is a sample from the iPinYou Real-Time Bidding dataset. The data.frame named ipinyou.train
        // is a sample from the data of 2013-10-19 and the data.frame named ipinyou.test
        // is a sample from the data of 2013-10-20.

        String inputDatasetFile = args[0];      // "train-dataset.csv" "train-dataset-full.csv"
        String testDatasetFile  = args[1];      // "test-dataset.csv" "test-dataset-full.csv"
        String inputDatasetLabelsFile = args[2];      // "train-dataset-labels.csv" "train-dataset-labels-full.csv"
        String testDatasetLabelsFile  = args[3];      // "test-dataset-labels.csv" "test-dataset-labels-full.csv"


        // Execution time profiling
        Instant start;
        Instant finish;

        try {

//             the loaded data is from the original ipinyou dataset but hashed
//             Feature hashing, also called as the hashing trick, is a method to
//             transform features to vector. Without looking the indices up in
//             an associative array, it applies a hash function to the features
//             and uses their hash values as indices directly.
//
//             The dataset has 18 features out of which:
//             IP + Region + City + AdExchange + Domain +
//              URL + AdSlotId + AdSlotWidth + AdSlotHeight +
//              AdSlotVisibility + AdSlotFormat + CreativeID +
//              Adid + isClick
//             the isCLick is the label and the other the features
            DMatrixRMaj inputDataRead = MatrixIO
                    .loadCSV(classLoader.getResource(inputDatasetFile)
                            .getFile(), true);
            SimpleMatrix trainData = SimpleMatrix.wrap(inputDataRead);
            DMatrixRMaj testDataRead = MatrixIO
                    .loadCSV(classLoader.getResource(testDatasetFile)
                            .getFile(), true);
            SimpleMatrix testData = SimpleMatrix.wrap(testDataRead);

            DMatrixRMaj inputDataLabelsRead = MatrixIO
                    .loadCSV(classLoader.getResource(inputDatasetLabelsFile)
                            .getFile(), true);
            SimpleMatrix trainDataLabels = SimpleMatrix.wrap(inputDataLabelsRead);
            DMatrixRMaj testDataLabelsRead = MatrixIO
                    .loadCSV(classLoader.getResource(testDatasetLabelsFile)
                            .getFile(), true);
            SimpleMatrix testDataLabels = SimpleMatrix.wrap(testDataLabelsRead);

            // run algorithm

            StreamingFTRL ftrlModel = new StreamingFTRL(trainData,
                    trainDataLabels,
                    "binomial", // gaussian, binomial, poisson
                    0.01,
                    0.1,
                    1.0,
                    1.0,
                    100);

            // Time execution
            start = Instant.now();

            // switch between batch mode and streaming mode, for testing
            boolean batchMode = false;
            // sliding window size
            int slide_size = trainData.numCols() - 150; // small value bad AUC (<50evs), large value good AUC (>= 50evs)

            if (batchMode == false) {

                for (int dId = 0; dId < trainData.numCols() - slide_size; dId++) {
                    // update FTRL GLM
                    SimpleMatrix values = trainData.extractMatrix(0, trainData.numRows(), dId, slide_size + dId);
                    SimpleMatrix valuesLabels = trainDataLabels.extractMatrix(dId, slide_size + dId, 0, trainDataLabels.numCols());
                    ftrlModel.setFtrlModel(ftrlModel.trainFTRLModel(ftrlModel.getFtrlModel(),
                            values,
                            valuesLabels));
                }
            }
            else {
                // training the model, supervised learning
                ftrlModel.setFtrlModel(ftrlModel.trainFTRLModel(ftrlModel.getFtrlModel(), trainData, trainDataLabels));
            }
            finish = Instant.now();

            System.out.println("Prediction matrix after training");
            ftrlModel.getFtrlModel().getP().print("%f");

            long timeElapsed = Duration.between(start, finish).toMillis();
            System.out.println("Execution time for "+ (batchMode==false ? " sliding ":" batch ") + " FTRL for GLM " +
                    (trainData.numCols()) +  " values is " + timeElapsed + " ms" );

            System.out.println("Trained " + (batchMode==false ? " sliding ":" batch ") + " FTRL for GLM experiment (Logistic Regression Classifier)");
            System.out.println("Weights");
            ftrlModel.getFtrlModel().getW().print("%f");

            // prediction of the trained model on testing data
            // accuracy analysis (AUC) given the testDataLabels
            double AUC = 0.0;
            for (int dId = 0; dId < testData.numCols() - slide_size; dId++) {
                // update FTRL GLM
                SimpleMatrix testValues = testData.extractMatrix(0, testData.numRows(), dId, slide_size + dId);
                SimpleMatrix testValuesLabels = testDataLabels.extractMatrix(dId, slide_size + dId, 0, testDataLabels.numCols());
                ftrlModel.setFtrlModel(ftrlModel.predictFTRLModel(ftrlModel.getFtrlModel(), testValues));
                AUC = computeFtrlAUC(ftrlModel.getFtrlModel().getP(), testValuesLabels);
            }
            System.out.println("Prediction");
            ftrlModel.getFtrlModel().getP().print("%f");

            System.out.println("AUC: " + AUC);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}