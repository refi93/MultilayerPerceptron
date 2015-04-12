/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

import java.util.ArrayList;

/**
 *
 * @author raf
 */
public class MLPModel {
    double alpha, momentum;
    int hiddenLayerSize;
    int epochCount;
    boolean recordErrors;
    
    public MLPModel (double alpha, double momentum, int hiddenLayerSize, int epochCount, boolean recordErrors) {
        this.alpha = alpha;
        this.momentum = momentum;
        this.hiddenLayerSize = hiddenLayerSize;
        this.epochCount = epochCount;
        this.recordErrors = recordErrors; // ci sa maju zaznamenavat estimacna a validacna chyba
    }
    
    public double crossValidate(ArrayList<Matrix> trainSets, ArrayList<Matrix> testSets) {
        
        double cv = 0.0; // cross validation coefficient
        double averageEstimationError = 0.0;
        for (int k = 0; k < trainSets.size(); k++) {
            NeuronMemento nm = this.train(trainSets.get(k), testSets.get(k));
            cv += nm.validationError;
            averageEstimationError += nm.estimationError;
        }
        cv /= trainSets.size();
        averageEstimationError /= trainSets.size();
        System.out.println(alpha + ", " + momentum + ", " + hiddenLayerSize + ", " + (cv * 100) + ", " + (averageEstimationError * 100));
        return cv;
    }
    
    // natrenujeme na trainSet, otestujeme na testSet, chyba na testSet nema vplyv na proces trenovania, vyuziva sa to iba pri k-cross validacii
    public NeuronMemento train(Matrix trainSet, Matrix testSet) {
        int trainDataCount = trainSet.numRows();
        int dimensionsCount = trainSet.numCols() - 1;
        int inputSize = dimensionsCount + 1;
        int classCount = Variables.classCount;
        ArrayList<Double> estimationErrors = new ArrayList<>();
        ArrayList<Double> validationErrors = new ArrayList<>();
        
        Matrix weightsHidden = Helpers.randMatrix(Variables.hiddenLayersCount, inputSize);
        Matrix weightsOut = Helpers.randMatrix(classCount, Variables.hiddenLayersCount + 1);
        
        // zapamatame si predoslu zmenu vah, aby sme mohli aplikovat momentum
        Matrix previousDeltaOut = null; 
        Matrix previousDeltaHidden = null;
                
        double bestValidationError = 1.0;
        NeuronMemento bestWeights = null;
        
        for (int ep = 0; ep < epochCount; ep++) {
            //System.out.println(ep);
            // natrenujeme na estimacnej mnozine
            for (int j = 0; j < trainSet.numRows(); j++) {
                Matrix x = Helpers.vectorToMatrix(new ArrayList(trainSet.get(j)));
                x = Helpers.MatrixComponentDivision(Helpers.matrixSubstract(x, Variables.trainDataAverage), Variables.trainDataStdDev);
                x.get(x.numRows() - 1).set(0, -1.0);

                Matrix net = Helpers.matrixProduct(weightsHidden, x);
                Matrix hBiased = Helpers.appendBias(Helpers.matrixSigmoid(net)); // pridame bias k matici
                Matrix y = Helpers.matrixSigmoid(Helpers.matrixProduct(weightsOut, hBiased));

                // urcime target
                Matrix target = Helpers.numberMatrix(classCount, 1, 0);
                int targetClass = (int)Math.round(trainSet.get(j).get(inputSize - 1)) - 1;
                target.get(targetClass).set(0, 1.0);

                Matrix sigmaOut = Helpers.matrixComponentProduct(Helpers.matrixComponentProduct(Helpers.matrixSubstract(target, y), y), Helpers.matrixSubstract(Helpers.numberMatrix(y.numRows(), y.numCols(), 1.0), y));

                Matrix weightsOutUnbiased = Helpers.removeLastColumn(weightsOut);
                Matrix hUnbiased = hBiased.subMatrix(0, hBiased.numRows() - 1);

                Matrix sigmaHidden = Helpers.matrixComponentProduct(Helpers.matrixComponentProduct(Helpers.matrixProduct(Helpers.transpose(weightsOutUnbiased), sigmaOut), hUnbiased), Helpers.matrixSubstract(Helpers.numberMatrix(hUnbiased.numRows(), hUnbiased.numCols(), 1), hUnbiased)) ; 

                Matrix deltaOut = Helpers.scalarProduct(Helpers.matrixProduct(sigmaOut, Helpers.transpose(hBiased)), Variables.alpha);
                weightsOut = Helpers.matrixSum(weightsOut, deltaOut);
                // pridame momentum
                if (previousDeltaOut != null) {
                    weightsOut = Helpers.matrixSum(weightsOut, Helpers.scalarProduct(previousDeltaOut, Variables.momentum));
                }
                previousDeltaOut = deltaOut;

                Matrix deltaHidden = Helpers.scalarProduct(Helpers.matrixProduct(sigmaHidden, Helpers.transpose(x)), Variables.alpha);
                weightsHidden = Helpers.matrixSum(weightsHidden, deltaHidden);
                // pridame momentum
                if (previousDeltaHidden != null) {
                    weightsHidden = Helpers.matrixSum(weightsHidden, Helpers.scalarProduct(previousDeltaHidden, Variables.momentum));
                }
                previousDeltaHidden = deltaHidden;
            }
            // otestujeme na validacnej mnozine
            double validationError = this.test(weightsHidden, weightsOut, testSet, false);
            if (this.recordErrors) {
                validationErrors.add(validationError);
                estimationErrors.add(this.test(weightsHidden, weightsOut, trainSet, false));
            }
            
            if (validationError < bestValidationError) {
                bestValidationError = validationError;
                // otestujeme na estimacnej mnozine
                double estimationError = 0;
                if (!this.recordErrors) {
                    estimationError = this.test(weightsHidden, weightsOut, trainSet, false);
                }
                bestWeights = new NeuronMemento(weightsHidden, weightsOut, validationError, estimationError);
            }    
        }
        if (this.recordErrors) {
            System.out.println(estimationErrors);
            System.out.println(validationErrors);
        }
        return bestWeights;
    }
    
    // vrati to validacnu chybu na test mnozine
    public double test(Matrix weightsHidden, Matrix weightsOut, Matrix testSet, boolean display) {
        // testovanie
        int inputSize = testSet.numCols();
        int errorCount = 0;
        if (display) System.out.print('[');
        int[][] confusionMatrix = new int[3][3];
        for (int j = 0; j < testSet.numRows(); j++) {
            Matrix x = Helpers.vectorToMatrix(new ArrayList(testSet.get(j)));
            x = Helpers.MatrixComponentDivision(Helpers.matrixSubstract(x, Variables.trainDataAverage), Variables.trainDataStdDev);
            x.get(x.numRows() - 1).set(0, -1.0);

            Matrix hBiased = Helpers.appendBias(Helpers.matrixSigmoid(Helpers.matrixProduct(weightsHidden, x)));
            Matrix net = Helpers.matrixSigmoid(Helpers.matrixProduct(weightsOut, hBiased));

            // urcime target
            Matrix target = Helpers.numberMatrix(Variables.classCount, 1, 0);
            int targetClass = (int)Math.round(testSet.get(j).get(inputSize - 1)) - 1;
            target.get(targetClass).set(0, 1.0);
            if (Helpers.getCategory(net) != Helpers.getCategory(target)) {
                errorCount++;
                if (display) System.out.print("1,");
                confusionMatrix[(int)(Helpers.getCategory(net))][(int)Helpers.getCategory(target)]++;
            }
            else{
                confusionMatrix[(int)(Helpers.getCategory(net))][(int)Helpers.getCategory(target)]++;
                if (display) System.out.print("0,");
            }
        }if (display) {
            System.out.println("]");
        
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    System.out.print(confusionMatrix[i][j] + ", ");
                }
                System.out.println();
            }
        }
        //System.out.println(errorCount + " " + testSet.numRows());
        //System.out.println(testSet);
        double errorPercentage = (0.0 + errorCount) / testSet.numRows(); // validacna chyba
        
        return errorPercentage;
    }
}
