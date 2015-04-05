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
    
    public MLPModel (double alpha, double momentum, int hiddenLayerSize, int epochCount) {
        this.alpha = alpha;
        this.momentum = momentum;
        this.hiddenLayerSize = hiddenLayerSize;
        this.epochCount = epochCount;
    }
    
    public double crossValidate(ArrayList<Matrix> trainSets, ArrayList<Matrix> testSets) {
        
        double cv = 0.0; // cross validation coefficient
        for (int k = 0; k < trainSets.size(); k++) {
            cv += this.train(trainSets.get(k), testSets.get(k)).validationError;
        }
        cv /= trainSets.size();
        return cv;
    }
    
    public NeuronMemento train(Matrix trainSet, Matrix testSet) {
        int trainDataCount = trainSet.numRows();
        int dimensionsCount = trainSet.numCols() - 1;
        int inputSize = dimensionsCount + 1;
        int classCount = Variables.classCount;
        
        Matrix weightsHidden = Helpers.randMatrix(Variables.hiddenLayersCount, inputSize);
        Matrix weightsOut = Helpers.randMatrix(classCount, Variables.hiddenLayersCount + 1);
        
        // zapamatame si predoslu zmenu vah, aby sme mohli aplikovat momentum
        Matrix previousDeltaOut = null; 
        Matrix previousDeltaHidden = null;
                
        double bestValidationError = 1.0;
        NeuronMemento bestWeights = null;
        
        for (int ep = 0; ep < epochCount; ep++) {
            //System.out.println("epocha " + ep);
            
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
            double validationError = this.test(weightsHidden, weightsOut, testSet);
  
            if (validationError < bestValidationError) {
                bestValidationError = validationError;
                bestWeights = new NeuronMemento(weightsHidden, weightsOut, validationError);
            }    
        }
        return bestWeights;
    }
    
    public double test(Matrix weightsHidden, Matrix weightsOut, Matrix testSet) {
        // testovanie
        int inputSize = testSet.numCols();
        int errorCount = 0;
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
            }
        }
        double validationError = (0.0 + errorCount) / testSet.numRows(); // validacna chyba
        
        return validationError;
    }
}
