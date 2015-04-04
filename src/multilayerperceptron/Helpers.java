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
    
    static Matrix scalarProduct(Matrix matrix, double scalar){
        Matrix ret = new Matrix();
        int rows = matrix.numRows();
        int cols = matrix.numCols();
        
        for (int r = 0; r < rows; r++) {
            ret.addRow();
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
    
    static Matrix transpose(Matrix matrix){
        int rows = matrix.numRows();
        int cols = matrix.numCols();
        
        Matrix ret = new Matrix();
        
        for (int c = 0; c < cols; c++){
            ret.addRow();
            for (int r = 0; r < rows; r++){
                ret.get(c).add(matrix.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static Matrix matrixProduct(Matrix matrix1, Matrix matrix2){
        Matrix ret = new Matrix();
        int matrix1Rows = matrix1.numRows();
        int matrix1Cols = matrix1.numCols();
        int matrix2Rows = matrix2.numRows();
        int matrix2Cols = matrix2.numCols();
        
        if (matrix1Cols != matrix2Rows){
            System.err.println(matrix1);
            System.err.println(matrix2);
            System.err.println("NESEDIA VELKOSTI MATIC PRI NASOBENI");
            System.exit(0);
        }
        
        for (int r = 0; r < matrix1Rows; r++){
            ret.addRow();
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
    
    static Matrix randMatrix (int rows, int cols) {
        Matrix ret = new Matrix();
        for (int r = 0; r < rows; r++) {
            ret.addRow();
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(Math.random());
            }
        }
        
        return ret;
    }
    
    static Matrix vectorToMatrix (ArrayList<Double> vector) {
        Matrix ret = new Matrix();
        for (int i = 0; i < vector.size(); i++) {
            ret.addRow();
            ret.get(i).add(vector.get(i));
        }
        
        return ret;
    }
    
    static Matrix matrixSigmoid(Matrix netMatrix) {
        Matrix ret = new Matrix();
        for (int r = 0; r < netMatrix.numRows(); r++) {
            ret.addRow();
            for (int c = 0; c < netMatrix.get(r).size(); c++) {
                ret.get(r).add(Helpers.sigmoid(netMatrix.get(r).get(c)));
            }
        }
        return ret;
    }
    
    static Matrix matrixDeepCopy(Matrix matrix) {
        Matrix ret = new Matrix();
        for (int r = 0; r < matrix.numRows(); r++) {
            ret.addRow();
            for (int c = 0; c < matrix.get(r).size(); c++) {
                ret.get(r).add(matrix.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static Matrix appendBias(Matrix matrix) {
        Matrix ret = Helpers.matrixDeepCopy(matrix);
        int cols = ret.numCols();
        ret.addRow();
        for (int c = 0; c < cols; c++) {
           ret.get(ret.numRows() - 1).add(-1.0);
        }
        
        return ret;
    }
    
    static Matrix numberMatrix(int rows, int cols, double number) {
        Matrix ret = new Matrix();
        for (int r = 0; r < rows; r++) {
            ret.addRow();
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(number);
            }
        }
        
        return ret;
    }
    
    static Matrix matrixSum(Matrix matrix1, Matrix matrix2){
        int rows = matrix1.numRows();
        int cols = matrix1.numCols();
        
        Matrix ret = new Matrix();
        for (int r = 0; r < rows; r++) {
            ret.addRow();
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix1.get(r).get(c) + matrix2.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static Matrix matrixSubstract(Matrix matrix1, Matrix matrix2){
        int rows = matrix1.numRows();
        int cols = matrix1.numCols();
        
        Matrix ret = new Matrix();
        for (int r = 0; r < rows; r++) {
            ret.addRow();
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix1.get(r).get(c) - matrix2.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static Matrix matrixComponentProduct(Matrix matrix1, Matrix matrix2) {
        int rows = matrix1.numRows();
        int cols = matrix1.numCols();
        if (rows != matrix2.numRows() || cols != matrix2.numCols()) {
            System.err.println("MATICE NEMAJU ROVNAKE ROZMERY");
            System.exit(0);
        }
        
        Matrix ret = new Matrix();
        
        for (int r = 0; r < rows; r++) {
            ret.addRow();
            for (int c = 0; c < cols; c++) {
                ret.get(r).add(matrix1.get(r).get(c) * matrix2.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static Matrix removeLastColumn(Matrix matrix) {
        int rows = matrix.numRows();
        int cols = matrix.numCols();
        Matrix ret = new Matrix();
        
        for (int r = 0; r < rows; r++) {
            ret.addRow();
            for (int c = 0; c < cols - 1; c++) {
                ret.get(r).add(matrix.get(r).get(c));
            }
        }
        
        return ret;
    }
    
    static int getCategory(Matrix net) {
        //System.out.println(net);
        int rows = net.numRows();
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

