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
        
        Variables.trainDataAverage = Helpers.matrixAverage(trainData);
        Variables.trainDataStdDev = Helpers.matrixStdDev(trainData);

        
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
            Matrix testSet = trainUnits.get(i);
            Matrix trainSet = new Matrix();
            for (int j = 0; j < Variables.validationUnitsCount; j++) {
                if (i != j) {
                    trainSet.appendMatrix(trainUnits.get(j));
                }
            }
            trainSets.add(trainSet);
            testSets.add(testSet);
        }
        
        double bestAlpha = 0;
        double bestMomentum = 0; 
        int bestHiddenLayerSize = 0;
        
        double bestCv = 1.0;
        for (double alpha = 0.05; alpha < 0.31; alpha += 0.05) {
            for (double momentum = 0; momentum < 0.51; momentum += 0.1) {
                for (int hiddenLayerSize = 16; hiddenLayerSize < 25; hiddenLayerSize += 8) {
                    MLPModel mlp = new MLPModel(alpha, momentum, hiddenLayerSize, 1000);
                    double cv = mlp.crossValidate(trainSets, testSets);
                    if (cv < bestCv) {
                        bestCv = cv;
                        bestAlpha = alpha;
                        bestMomentum = momentum;
                        bestHiddenLayerSize = hiddenLayerSize;
                    }
                }
            }
        }

        MLPModel bestModel = new MLPModel(bestAlpha, bestMomentum, bestHiddenLayerSize, 1000);
        NeuronMemento nm = bestModel.train(trainData, trainData);
        double testError = bestModel.test(nm.weightsHidden, nm.weightsOut, testData);
        System.out.println("najlepsi model: " + bestModel.alpha + " " + bestModel.momentum + " " + bestModel.hiddenLayerSize + " error: " + testError);
    }
    
}
