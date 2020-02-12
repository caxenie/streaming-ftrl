package org.streamingml.experiments;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.ops.ConvertDMatrixStruct;
import org.ejml.simple.SimpleMatrix;

class StreamingFTRLModelState {

    private SimpleMatrix x;     // features
    private SimpleMatrix y;     // labels
    private String family;      // link function to be used in the model
    private double alpha;       // alpha in the per-coordinate learning rate
    private double beta;        // beta in the per-coordinate learning rate
    private double L1;          // L1 regularization term
    private double L2;          // L2 regularization term
    private SimpleMatrix w;     // weights to learn, init xDim[0]
    private SimpleMatrix p;     // prediction, xDim[1]
    private int epochs;         // number of epochs to train
    private SimpleMatrix z;     // augmented state, init xDim[0]
    private SimpleMatrix n;     // learning increment, init xDim[0]
    // design matrix
    private SimpleMatrix xDim; // IntegerVector::create(x.n_rows, x.n_cols)
    private SimpleMatrix xP;   // (x.col_ptrs, x.col_ptrs + x.n_cols + 1);
    private SimpleMatrix xI;   // (x.row_indices, x.row_indices + x.n_nonzero);
    private SimpleMatrix xX;   // (x.values, x.values + x.n_nonzero);

    void setX(SimpleMatrix xIn) {
        this.x = xIn;
    }

    SimpleMatrix getX() {
        return this.x;
    }

    void setY(SimpleMatrix yIn) {
        this.y = yIn;
    }

    SimpleMatrix getY() {
        return this.y;
    }

    void setFamily(String f) {
        this.family = f;
    }

    String getFamily() {
        return this.family;
    }

    void setAlpha(double a) {
        this.alpha = a;
    }

    double getAlpha() {
        return this.alpha;
    }

    void setBeta(double b) {
        this.beta = b;
    }

    double getBeta() {
        return this.beta;
    }

    void setL1(double l1) {
        this.L1 = l1;
    }

    double getL1() {
        return this.L1;
    }

    void setL2(double l2) {
        this.L2 = l2;
    }

    double getL2() {
        return this.L2;
    }

    void setW(SimpleMatrix wIn) {
        this.w = wIn;
    }

    SimpleMatrix getW() {
        return this.w;
    }

    void setP(SimpleMatrix p) {
        this.p = p;
    }

    SimpleMatrix getP() {
        return this.p;
    }

    void setEpochs(int e) {
        this.epochs = e;
    }

    int getEpochs() {
        return this.epochs;
    }

    void setZ(SimpleMatrix zIn) {
        this.z = zIn;
    }

    SimpleMatrix getZ() {
        return this.z;
    }

    void setN(SimpleMatrix nIn) {
        this.n = nIn;
    }

    SimpleMatrix getN() {
        return this.n;
    }

    void setxDim(SimpleMatrix xD) {
        this.xDim = xD;
    }

    SimpleMatrix getxDim() {
        return this.xDim;
    }

    void setxP(SimpleMatrix xp) {
        this.xP = xp;
    }

    SimpleMatrix getxP() {
        return this.xP;
    }

    void setxI(SimpleMatrix xi) {
        this.xI = xi;
    }

    SimpleMatrix getxI() {
        return this.xI;
    }

    void setxX(SimpleMatrix x) {
        this.xX = x;
    }

    SimpleMatrix getxX() {
        return this.xX;
    }

}

class StreamingFTRL {

    private StreamingFTRLModelState ftrlModel = new StreamingFTRLModelState();

    void setFtrlModel(StreamingFTRLModelState ftrlModel) {
        this.ftrlModel = ftrlModel;
    }

    StreamingFTRLModelState getFtrlModel(){
        return this.ftrlModel;
    }

    StreamingFTRL(SimpleMatrix dX, SimpleMatrix dY, String f, double a, double b,
                  double l1, double l2, int epochs) {

        this.ftrlModel.setX(dX);
        this.ftrlModel.setY(dY);
        this.ftrlModel.setFamily(f);
        this.ftrlModel.setAlpha(a);
        this.ftrlModel.setBeta(b);
        this.ftrlModel.setL1(l1);
        this.ftrlModel.setL2(l2);

        // design matrix
        // In some situations the speed improvement of using a sparse matrix can be substantial.
        // Do note that if your system isn't sparse enough or if its structure isn't advantageous
        // it could run even slower using sparse operations!
        // Make sparse matrix iterators correspondence
        DMatrixSparseCSC sparseX = ConvertDMatrixStruct.convert(dX.getDDRM(), new DMatrixSparseCSC(dX.numRows(), dX.numCols()), 0.01);
        int nonZero = sparseX.nz_length;
        int[] colP = sparseX.col_idx;
        double[] colPtrs = new double[colP.length];
        for (int id = 0; id < colPtrs.length; id++){
            colPtrs[id] = (double) colP[id];
        }
        int[] rowI = sparseX.nz_rows;
        double[] rowIndices = new double[rowI.length];
        for (int id = 0; id < rowIndices.length; id++){
            rowIndices[id] = (double) rowI[id];
        }
        double[] values = sparseX.nz_values;
        // copy the first dX.numCols() + 1 elements from the sparse matrix colPtrs vector member
        this.ftrlModel.setxP(new SimpleMatrix(1, dX.numCols() + 1, true, colPtrs));
        // copy the first nonZero elements from the sparse matrix rowIndices vector member
        this.ftrlModel.setxI(new SimpleMatrix(1, nonZero, true, rowIndices));
        // copy the first nonZero elements from the sparse matrix values vector member
        this.ftrlModel.setxX(new SimpleMatrix(1, nonZero, true, values));
        // model initialization
        this.ftrlModel.setZ(new SimpleMatrix(1, dX.numRows()));
        this.ftrlModel.setN(new SimpleMatrix(1, dX.numRows()));
        this.ftrlModel.setW(new SimpleMatrix(1, dX.numRows()));
        // model prediction
        this.ftrlModel.setP(new SimpleMatrix(1, dX.numCols()));
        // set nr of training epochs
        this.ftrlModel.setEpochs(epochs);
    }

    private double ftrlPredictionTransform(double x, String family) {
        switch (family) {
            case "gaussian":
                return x;
            case "binomial":
                return 1.0 / (1.0 + Math.exp(-x));
            case "poisson":
                return Math.exp(x);
            default:
                return 0.0;
        }
    }

    private SimpleMatrix ftrlWeightsUpdate(double alpha, double beta, double l1, double l2, SimpleMatrix z, SimpleMatrix n) {
        SimpleMatrix eta = new SimpleMatrix(n.numRows(), n.numCols());
        // z should be a numeric vector
        for (int id = 0; id < eta.getNumElements(); id++){
            eta.set(id, Math.sqrt(n.get(id)));
        }
        eta = eta.plus(beta).scale(1.0 / alpha).plus(l2);
        SimpleMatrix zSign = new SimpleMatrix(z.numRows(), z.numCols());
        for (int id = 0; id < zSign.getNumElements(); id++){
            zSign.set(id, Math.signum(z.get(id)));
        }
        SimpleMatrix w = eta.elementPower(-1.0).scale(-1.0).elementMult(z.minus(zSign.scale(l1)));
        // set the sub regularized values to 0
        // w[abs(z) <= l1] = 0;
        int[] sparseIndex = new int[z.getNumElements()];
        for (int id = 0; id < z.getNumElements(); id++){
            if(z.get(id) <= l1){
                sparseIndex[id] = id;
            }
        }
        for (int id = 0; id < sparseIndex.length; id++){
            if(sparseIndex[id] != 0) {
                w.set(sparseIndex[id], 0.0);
            }
        }
        return w;
    }

    // Diff adjacent elements, for a vector, return a vector of the same orientation,
    // containing the differences between consecutive elements
    private SimpleMatrix diffElemPred(SimpleMatrix arr){
        SimpleMatrix diffArr = new SimpleMatrix(arr.numRows(), arr.numCols() - 1);
        for(int id = 0; id < arr.getNumElements() - 1; id++){
            diffArr.set(id, arr.get(id+1) - arr.get(id));
        }
        return diffArr;
    }

    // element sqrt
    private SimpleMatrix elementSqrt(SimpleMatrix in){
        SimpleMatrix res = new SimpleMatrix(in.numRows(), in.numCols());
        for(int id = 0; id < in.numRows(); id++){
            for (int jd = 0; jd < in.numCols(); jd++){
                res.set(id, jd, Math.sqrt(in.get(id, jd)));
            }
        }
        return res;
    }

    StreamingFTRLModelState predictFTRLModel(StreamingFTRLModelState state, SimpleMatrix xData){
        // get state components
        // update state components
        SimpleMatrix p = state.getW().mult(xData);
        switch (state.getFamily()) {
            case "gaussian":
                state.setP(p);
                break;
            case "binomial":
                for(int id = 0; id < p.getNumElements(); id++){
                    p.set(id, 1.0 / (1.0 + Math.exp(-p.get(id))));
                }
                state.setP(p);
                break;
            case "poisson":
                for(int id = 0; id < p.getNumElements(); id++){
                    p.set(id, Math.exp(p.get(id)));
                }
                state.setP(p);
                break;
        }
        return state;
    }

    StreamingFTRLModelState trainFTRLModel(StreamingFTRLModelState state, SimpleMatrix xData, SimpleMatrix yData){
        // recover hyperparameters
        double alpha = state.getAlpha();
        double beta = state.getBeta();
        double l1 = state.getL1();
        double l2 = state.getL2();
        // recover design matrix
        SimpleMatrix x_p = state.getxP();
        SimpleMatrix x_i = state.getxI();
        SimpleMatrix x_x = state.getxX();
        // recover model
        SimpleMatrix z = state.getZ();
        SimpleMatrix n = state.getN();
        SimpleMatrix w = state.getW();
        // recover last model prediction
        SimpleMatrix p = state.getP();

        String distFamily = state.getFamily();

        // update for the current window data
        state.setX(xData);
        state.setY(yData);

        // compute non-zero feature count in the sparse matrix
        SimpleMatrix non_zero_count = diffElemPred(x_p);

        // model updating
        for (int r = 0; r < state.getEpochs(); r++) {
            for (int t = 0; t < yData.getNumElements(); t++) {
                // Non-Zero Feature Index in sparse matrix
                SimpleMatrix non_zero_index =
                        new SimpleMatrix(1, (int) non_zero_count.get(t));
                for (int id = 0; id < non_zero_index.getNumElements(); id++) {
                    non_zero_index.set(id, id + 1);
                }
                non_zero_index = non_zero_index.plus(x_p.get(t)).minus(1);
                // non-zero feature index for each sample
                SimpleMatrix i = new SimpleMatrix(1,
                        non_zero_index.getNumElements());
                for (int id = 0; id < non_zero_index.getNumElements(); id++) {
                    int index = (int) non_zero_index.get(id);
                    double val = x_i.get(index);
                    i.set(id, val);
                }
                // non-zero feature value for each sample
                SimpleMatrix x_si = new SimpleMatrix(1,
                        non_zero_index.getNumElements());
                for (int id = 0; id < non_zero_index.getNumElements(); id++) {
                    x_si.set(id, x_x.get((int) non_zero_index.get(id)));
                }
                // model parameters
                SimpleMatrix z_i = new SimpleMatrix(i.numRows(), i.numCols());
                SimpleMatrix n_i = new SimpleMatrix(i.numRows(), i.numCols());

                for (int id = 0; id < i.numCols(); id++) {
                    z_i.set(id, z.get((int) i.get(id)));
                    n_i.set(id, n.get((int) i.get(id)));
                }
                // computing weight and prediction
                SimpleMatrix w_i = ftrlWeightsUpdate(alpha, beta, l1, l2, z_i, n_i);
                double p_t = ftrlPredictionTransform((x_si.elementMult(w_i)).elementSum(), distFamily);
                // updating weight and prediction
                for (int id = 0; id < i.numCols(); id++) {
                    w.set((int) i.get(id), w_i.get(id));
                }
                p.set(t, p_t);

                // computing model parameter of next round
                SimpleMatrix g_i = x_si.scale(p.get(t) - (yData.get(t)));
                SimpleMatrix s_i = (elementSqrt(n_i.plus(g_i.elementPower(2))).minus(elementSqrt(n_i))).scale(1.0 / alpha);
                SimpleMatrix z_i_next = z_i.plus(g_i).minus(s_i.elementMult(w_i));
                SimpleMatrix n_i_next = n_i.plus(g_i.elementPower(2));
                // Updating Model Parameter
                for (int id = 0; id < i.numCols(); id++) {
                    z.set((int) i.get(id), z_i_next.get(id));
                    n.set((int) i.get(id), n_i_next.get(id));
                }
            }
        }

        // update state for next iteration
        state.setZ(z);
        state.setN(n);
        state.setW(w);
        state.setP(p);
        state.setxI(x_i);
        state.setxP(x_p);
        state.setxX(x_x);

        return state;
    }
}
