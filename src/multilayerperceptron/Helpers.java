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
public class Helpers {
    
    static Double letterToNumber(char c) {
        return 0.0 + c - 'A'; 
    }
    
    static char numberToLetter(int n) {
        return (char) ('A' + n);
    }
    
    // overi, ci vektory maju rovnake velkosti
    private static void checkVectorSizesForEquality(ArrayList<Double> v1, ArrayList<Double> v2){
        if (v1.size() != v2.size()){
            System.err.println("vector sizes do not match!!!");
            System.exit(-1);
        }
    }
    
    static double crossProduct(ArrayList<Double> v1, ArrayList<Double> v2){
        checkVectorSizesForEquality(v1, v2);
        
        Double ret = 0.0;
        for (int i = 0; i < v1.size(); i++){
            ret += v1.get(i) * v2.get(i);
        }
        
        return ret;
    }
    
    static ArrayList<Double> sumVectors(ArrayList<Double> v1, ArrayList<Double> v2){
        checkVectorSizesForEquality(v1, v2);
        
        ArrayList<Double> ret = new ArrayList<>();
        for(int i = 0; i < v1.size(); i++){
            ret.add(v1.get(i) + v2.get(i));
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > scalarProduct(ArrayList<ArrayList<Double> > matrix, double scalar){
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        int rows = matrix.size();
        int cols = matrix.get(0).size();
        
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix.get(r).get(c) * scalar);
            }
        }
        
        return ret;
    }
    
    static double average(ArrayList<Double> numbers){
        double sum = 0;
        for (double x : numbers){
            sum += x;
        }
        return sum / numbers.size();
    }
    
    static double sigmoid(double y){
        return 1.0/(1.0 + Math.pow(Math.E, -y));
    }
    
    static double sigmoidDerivative(double y){
        return sigmoid(y) * (1 - sigmoid(y));
    }
    
    static ArrayList<ArrayList<Double> > transpose(ArrayList<ArrayList<Double> > matrix){
        int rows = matrix.size();
        int cols = matrix.get(0).size();
        
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        
        for (int c = 0; c < cols; c++){
            ret.add(new ArrayList<Double>());
            for (int r = 0; r < rows; r++){
                ret.get(c).add(matrix.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > matrixProduct(ArrayList<ArrayList<Double> > matrix1, ArrayList<ArrayList<Double> > matrix2){
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        int matrix1Rows = matrix1.size();
        int matrix1Cols = matrix1.get(0).size();
        int matrix2Rows = matrix2.size();
        int matrix2Cols = matrix2.get(0).size();
        
        if (matrix1Cols != matrix2Rows){
            System.err.println(matrix1);
            System.err.println(matrix2);
            System.err.println("NESEDIA VELKOSTI MATIC PRI NASOBENI");
            System.exit(0);
        }
        
        for (int r = 0; r < matrix1Rows; r++){
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < matrix2Cols; c++){
                Double result = 0.0;
                for (int i = 0; i < matrix1Cols; i++){
                    result += matrix1.get(r).get(i) * matrix2.get(i).get(c);
                }
                ret.get(r).add(result);
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > randMatrix (int rows, int cols) {
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(Math.random());
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > vectorToMatrix (ArrayList<Double> vector) {
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int i = 0; i < vector.size(); i++) {
            ret.add(new ArrayList<Double>());
            ret.get(i).add(vector.get(i));
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > matrixSigmoid(ArrayList<ArrayList<Double> > netMatrix) {
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int r = 0; r < netMatrix.size(); r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < netMatrix.get(r).size(); c++) {
                ret.get(r).add(Helpers.sigmoid(netMatrix.get(r).get(c)));
            }
        }
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > matrixDeepCopy(ArrayList<ArrayList<Double> > matrix) {
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int r = 0; r < matrix.size(); r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < matrix.get(r).size(); c++) {
                ret.get(r).add(matrix.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > appendBias(ArrayList<ArrayList<Double> > matrix) {
        ArrayList<ArrayList<Double> > ret = Helpers.matrixDeepCopy(matrix);
        int cols = ret.get(0).size();
        ret.add(new ArrayList<Double>());
        for (int c = 0; c < cols; c++) {
           ret.get(ret.size() - 1).add(-1.0);
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > numberMatrix(int rows, int cols, double number) {
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(number);
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > matrixSum(ArrayList<ArrayList<Double> > matrix1, ArrayList<ArrayList<Double> > matrix2){
        int rows = matrix1.size();
        int cols = matrix1.get(0).size();
        
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix1.get(r).get(c) + matrix2.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > matrixSubstract(ArrayList<ArrayList<Double> > matrix1, ArrayList<ArrayList<Double> > matrix2){
        int rows = matrix1.size();
        int cols = matrix1.get(0).size();
        
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix1.get(r).get(c) - matrix2.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > matrixComponentProduct(ArrayList<ArrayList<Double> > matrix1, ArrayList<ArrayList<Double> > matrix2) {
        int rows = matrix1.size();
        int cols = matrix1.get(0).size();
        
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix1.get(r).get(c) * matrix1.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static ArrayList<ArrayList<Double> > removeLastColumn(ArrayList<ArrayList<Double> > matrix) {
        int rows = matrix.size();
        int cols = matrix.get(0).size();
        ArrayList<ArrayList<Double> > ret = new ArrayList<>();
        
        for (int r = 0; r < rows; r++) {
            ret.add(new ArrayList<Double>());
            for (int c = 0; c < cols - 1; c++) {
                ret.get(r).add(matrix.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static int getCategory(ArrayList<ArrayList<Double> > net) {
        int rows = net.size();
        double max = 0; int maxPos = 0;
        for (int r = 0; r < rows; r++) {
            double cur = net.get(r).get(0);
            if (cur > max) {
                max = cur;
                maxPos = r;
            }
        }
        return maxPos;
    }
}

