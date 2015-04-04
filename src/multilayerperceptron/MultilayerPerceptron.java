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
    
    static Matrix data;
    
    public static void loadData() throws FileNotFoundException{
        Scanner in = new Scanner(new FileReader("2d.trn.dat"));
        data = new Matrix();
        while(in.hasNext()){
            data.addRow();
            for (int i = 0; i < 2; i++){
                Double input = Double.parseDouble(in.next());
                data.get(data.numRows() - 1).add(input);
            }
            data.get(data.numRows() - 1).add(Helpers.letterToNumber(in.next().charAt(0)));
        }
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
        //loadData();
        
        data = loadCsvData("test.csv");
        
        int dataCount = data.numRows();
        int dimensionsCount = data.get(0).size() - 1;
        int inputSize = dimensionsCount + 1;
        int classCount = 3;
        
        int trainCount = 120;
        int testCount = dataCount - trainCount;
        // rozdelime data na trenovaci a testovaci set
        data.shuffleRows();
        Matrix trainSet = data.subMatrix(0, trainCount);
        Matrix testSet = data.subMatrix(trainCount, dataCount);
        
        double alpha = 0.1;
        int hiddenLayersCount = 4;
        
        Matrix weightsHidden = Helpers.randMatrix(hiddenLayersCount, inputSize);
        Matrix weightsOut = Helpers.randMatrix(classCount, hiddenLayersCount + 1);
        
        int epochCount = 1000;
        ArrayList<Double> errors = new ArrayList<>();
        
        for (int i = 0; i < epochCount; i++) {
            
            // trenovanie
            trainSet.shuffleRows();
            for (int j = 0; j < trainSet.numRows(); j++) {
                Matrix x = Helpers.vectorToMatrix(new ArrayList(trainSet.get(j)));
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
                
                weightsOut = Helpers.matrixSum(weightsOut, Helpers.scalarProduct(Helpers.matrixProduct(sigmaOut, Helpers.transpose(hBiased)), alpha));
                
                weightsHidden = Helpers.matrixSum(weightsHidden, Helpers.scalarProduct(Helpers.matrixProduct(sigmaHidden, Helpers.transpose(x)), alpha));
            }
            
            // testovanie
            int goodCount = 0;
            for (int j = 0; j < testCount; j++) {
                Matrix x = Helpers.vectorToMatrix(new ArrayList(testSet.get(j)));
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
            double goodPercentage = (0.0 + goodCount) / testCount;
            System.out.println(goodPercentage);
        }
        
    }
    
}
