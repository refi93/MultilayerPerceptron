/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multilayerperceptron;

/**
 *
 * @author raf
 */
public class Variables {
    public static int validationUnitsCount = 8;
    public static int classCount = 3;
    public static int epochCount = 100;
    public static double alpha = 0.1;
    public static double momentum = 0.1;
    public static int hiddenLayersCount = 16;
    public static Matrix trainDataAverage; // aby sme mohli normalizovat data na vstupe
    public static Matrix trainDataStdDev;
}
