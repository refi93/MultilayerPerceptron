/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author raf
 */
public class MultilayerPerceptron {

    /**
     * @param args the command line arguments
     */
    
    public static Matrix loadData(String fileName) throws FileNotFoundException{
        Scanner in = new Scanner(new FileReader(fileName));
        Matrix ret = new Matrix();
        while(in.hasNext()){
            ret.addRow();
            for (int i = 0; i < 2; i++){
                Double input = Double.parseDouble(in.next());
                ret.get(ret.numRows() - 1).add(input);
            }
            ret.get(ret.numRows() - 1).add(Helpers.letterToNumber(in.next().charAt(0)));
        }
        
        return ret;
    }
    
    public static Matrix loadCsvData(String csvFile) throws FileNotFoundException, IOException{
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line;
        Matrix ret = new Matrix();
        
        while ((line = br.readLine()) != null) {
            ArrayList<Double> pom = new ArrayList<>();
            String[] inputs = line.split(",");
            for(String s : inputs){
                pom.add(Double.parseDouble(s)); 
            }
            ret.addRow(pom);
        }
        ret = Helpers.transpose(ret);
        return ret;
    }
    
    public static void main(String[] args) throws FileNotFoundException, IOException {
        // TODO code application logic here
        Matrix trainData = loadData("2d.trn.dat");
        Matrix testData = loadData("2d.tst.dat");
        
        Matrix trainDataAverage = Helpers.matrixAverage(trainData);
        Matrix trainDataStdDev = Helpers.matrixStdDev(trainData);

        
        /*Matrix data2 = Helpers.transpose(data);
        for (int r = 0; r < data2.numRows(); r++) {
            for (int c = 0; c < data2.numCols(); c++) {
                System.out.print(data2.get(r).get(c));
                if (c != data2.numCols() - 1) {
                    System.out.print(",");
                }
            }
            System.out.println();
        }*/
        
        //data = loadCsvData("test.csv");
        
        int trainDataCount = trainData.numRows();
        int dimensionsCount = trainData.get(0).size() - 1;
        int inputSize = dimensionsCount + 1;
        int classCount = Variables.classCount;
        
        int trainUnitSize = trainDataCount / Variables.validationUnitsCount;
        // rozdelime data na trenovaci a testovaci set
        trainData.shuffleRows();
        ArrayList<Matrix> trainSets = new ArrayList<>();
        ArrayList<Matrix> testSets = new ArrayList<>();
        ArrayList<Matrix> trainUnits = new ArrayList<>();
        
        for (int i = 0; i < Variables.validationUnitsCount; i++) {
            trainUnits.add(trainData.subMatrix(i * trainUnitSize, (i + 1) * trainUnitSize));
        }
        
        for (int i = 0; i < Variables.validationUnitsCount; i++) {
            Matrix trainSet = trainUnits.get(i);
            Matrix testSet = new Matrix();
            for (int j = 0; j < Variables.validationUnitsCount; j++) {
                if (i != j) {
                    testSet.appendMatrix(trainUnits.get(i));
                }
            }
            trainSets.add(trainSet);
            testSets.add(testSet);
        }
        
        Matrix weightsHidden = Helpers.randMatrix(Variables.hiddenLayersCount, inputSize);
        Matrix weightsOut = Helpers.randMatrix(classCount, Variables.hiddenLayersCount + 1);
        
        ArrayList<Double> errors = new ArrayList<>();
        double bestGoodPercentage = 0.0;
        NeuronMemento bestWeights = null;
        
        // zapamatame si predoslu zmenu vah, aby sme mohli aplikovat momentum
        Matrix previousDeltaOut = null; 
        Matrix previousDeltaHidden = null; 
        
        for (int ep = 0; ep < Variables.epochCount; ep++) {
            System.out.println("epocha " + ep);
            double averageGoodPercentage = 0.0;
            
            for (int k = 0; k < Variables.validationUnitsCount; k++) {
                Matrix trainSet = trainSets.get(k);
                Matrix testSet = testSets.get(k);
                for (int j = 0; j < trainSet.numRows(); j++) {
                    Matrix x = Helpers.vectorToMatrix(new ArrayList(trainSet.get(j)));
                    x = Helpers.MatrixComponentDivision(Helpers.matrixSubstract(x, trainDataAverage), trainDataStdDev);
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

                // testovanie
                int goodCount = 0;
                for (int j = 0; j < testSet.numRows(); j++) {
                    Matrix x = Helpers.vectorToMatrix(new ArrayList(testSet.get(j)));
                    x = Helpers.MatrixComponentDivision(Helpers.matrixSubstract(x, trainDataAverage), trainDataStdDev);
                    x.get(x.numRows() - 1).set(0, -1.0);

                    Matrix hBiased = Helpers.appendBias(Helpers.matrixSigmoid(Helpers.matrixProduct(weightsHidden, x)));
                    Matrix net = Helpers.matrixSigmoid(Helpers.matrixProduct(weightsOut, hBiased));

                    // urcime target
                    Matrix target = Helpers.numberMatrix(classCount, 1, 0);
                    int targetClass = (int)Math.round(testSet.get(j).get(inputSize - 1)) - 1;
                    target.get(targetClass).set(0, 1.0);

                    if (Helpers.getCategory(net) == Helpers.getCategory(target)) {
                        goodCount++;
                    }
                }
                double goodPercentage = (0.0 + goodCount) / testSet.numRows();
                averageGoodPercentage += goodPercentage;
            }
            averageGoodPercentage /= Variables.validationUnitsCount;
            System.out.println(averageGoodPercentage);
            if (averageGoodPercentage > bestGoodPercentage) {
                bestWeights = new NeuronMemento(weightsHidden, weightsOut, ep);
                bestGoodPercentage = averageGoodPercentage;
            }
        }
        
        // otestujeme najlepsiu neuronovu siet na ostrych datach
        weightsHidden = bestWeights.weightsHidden;
        weightsOut = bestWeights.weightsOut;
        Matrix confusionMatrix = Helpers.numberMatrix(3, 3, 0);
        
        int goodCount = 0;
        for (int j = 0; j < testData.numRows(); j++) {
            Matrix x = Helpers.vectorToMatrix(new ArrayList(testData.get(j)));
            x = Helpers.MatrixComponentDivision(Helpers.matrixSubstract(x, trainDataAverage), trainDataStdDev); // normalizejeme data
            x.get(x.numRows() - 1).set(0, -1.0);

            Matrix hBiased = Helpers.appendBias(Helpers.matrixSigmoid(Helpers.matrixProduct(weightsHidden, x)));
            Matrix net = Helpers.matrixSigmoid(Helpers.matrixProduct(weightsOut, hBiased));

            // urcime target
            Matrix target = Helpers.numberMatrix(classCount, 1, 0);
            int targetClass = (int)Math.round(testData.get(j).get(inputSize - 1)) - 1;
            target.get(targetClass).set(0, 1.0);
            
            double confusionMatrixValue = confusionMatrix.get(Helpers.getCategory(net)).get(Helpers.getCategory(target));
            confusionMatrix.get(Helpers.getCategory(net)).set(Helpers.getCategory(target), confusionMatrixValue + 1);
            if (Helpers.getCategory(net) == Helpers.getCategory(target)) {
                goodCount++;
            }
        }
        double goodPercentage = (0.0 + goodCount) / testData.numRows();
        System.out.println("Najlepsia siet je z epochy " + bestWeights.epoch + " a ma na ostrych datach uspesnost " + goodPercentage);
        System.out.println("Confusion matrix: " + confusionMatrix);
    }
    
}
