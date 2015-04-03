/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

/**
 *
 * @author raf
 */
public class MultilayerPerceptron {

    /**
     * @param args the command line arguments
     */
    
    static ArrayList<ArrayList<Double> > data;
    
    public static void loadData() throws FileNotFoundException{
        Scanner in = new Scanner(new FileReader("2d.trn.dat"));
        data = new ArrayList<>();
        while(in.hasNext()){
            data.add(new ArrayList<Double>());
            for (int i = 0; i < 2; i++){
                Double input = Double.parseDouble(in.next());
                data.get(data.size() - 1).add(input);
            }
            data.get(data.size() - 1).add(Helpers.letterToNumber(in.next().charAt(0)));
        }
    }
    
    public static void main(String[] args) throws FileNotFoundException {
        // TODO code application logic here
        loadData();
        int dataCount = data.size();
        int dimensionsCount = data.get(0).size() - 1;
        int inputSize = dimensionsCount + 1;
        int classCount = 3;
        
        int trainCount = 120;
        int testCount = dataCount - trainCount;
        // rozdelime data na trenovaci a testovaci set
        ArrayList<ArrayList<Double> > trainSet = new ArrayList(data.subList(0, trainCount));
        ArrayList<ArrayList<Double> > testSet = new ArrayList(data.subList(trainCount, dataCount));
        
        double alpha = 0.1;
        int hiddenLayersCount = 4;
        
        ArrayList<ArrayList<Double> > weightsHidden = Helpers.randMatrix(hiddenLayersCount, inputSize);
        ArrayList<ArrayList<Double> > weightsOut = Helpers.randMatrix(classCount, hiddenLayersCount + 1);
        
        int epochCount = 200;
        ArrayList<Double> errors = new ArrayList<>();
        
        for (int i = 0; i < epochCount; i++) {
            
            // trenovanie
            java.util.Collections.shuffle(trainSet);
            for (int j = 0; j < trainSet.size(); j++) {
                ArrayList<ArrayList<Double> > x = Helpers.vectorToMatrix(new ArrayList(trainSet.get(j)));
                x.get(x.size() - 1).set(0, -1.0);
             
                ArrayList<ArrayList<Double> > net = Helpers.matrixProduct(weightsHidden, x);
                ArrayList<ArrayList<Double> > hBiased = Helpers.appendBias(Helpers.matrixSigmoid(net)); // pridame bias k matici
                ArrayList<ArrayList<Double> > y = Helpers.matrixSigmoid(Helpers.matrixProduct(weightsOut, hBiased));
       
                // urcime target
                ArrayList<ArrayList<Double> > target = Helpers.numberMatrix(classCount, 1, 0);
                int targetClass = (int)Math.round(trainSet.get(j).get(inputSize - 1));
                target.get(targetClass).set(0, 1.0);
                
                ArrayList<ArrayList<Double> > sigmaOut = Helpers.matrixProduct(Helpers.matrixProduct(Helpers.matrixSubstract(target, y), Helpers.transpose(y)), Helpers.matrixSubstract(y, Helpers.numberMatrix(y.size(), y.get(0).size(), 1.0)));
                
                ArrayList<ArrayList<Double> > weightsOutUnbiased = Helpers.removeLastColumn(weightsOut);
                ArrayList<ArrayList<Double> > hUnbiased = new ArrayList<>(hBiased.subList(0, hBiased.size() - 1));
                
                ArrayList<ArrayList<Double> > sigmaHidden = Helpers.matrixComponentProduct(Helpers.matrixComponentProduct(Helpers.matrixProduct(Helpers.transpose(weightsOutUnbiased), sigmaOut), hUnbiased), Helpers.matrixSubstract(Helpers.numberMatrix(hUnbiased.size(), hUnbiased.get(0).size(), 1), hUnbiased)) ; 
                
                weightsOut = Helpers.matrixSum(weightsOut, Helpers.scalarProduct(Helpers.matrixProduct(sigmaOut, Helpers.transpose(hBiased)), alpha));
                
                weightsHidden = Helpers.matrixSum(weightsHidden, Helpers.scalarProduct(Helpers.matrixProduct(sigmaHidden, Helpers.transpose(net)), alpha));
            }
            
            // testovanie
            int errorCount = 0;
            for (int j = 0; j < testCount; j++) {
                ArrayList<ArrayList<Double> > x = Helpers.vectorToMatrix(new ArrayList(testSet.get(j)));
                x.get(x.size() - 1).set(0, -1.0);
                
                ArrayList<ArrayList<Double> > hBiased = Helpers.appendBias(Helpers.matrixSigmoid(Helpers.matrixProduct(weightsHidden, x)));
                ArrayList<ArrayList<Double> > net = Helpers.matrixProduct(weightsOut, hBiased);
                
                // urcime target
                ArrayList<ArrayList<Double> > target = Helpers.numberMatrix(classCount, 1, 0);
                int targetClass = (int)Math.round(testSet.get(j).get(inputSize - 1));
                target.get(targetClass).set(0, 1.0);
                                
                if (Helpers.getCategory(net) != Helpers.getCategory(target)) {
                    errorCount++;
                }
            }
            double goodPercentage = (0.0 + errorCount) / testCount;
            System.out.println(goodPercentage);
        }
        
    }
    
}
