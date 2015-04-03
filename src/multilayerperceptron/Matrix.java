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
public class Matrix {
    ArrayList<ArrayList<Double> > representation;
    
    public Matrix(){
        representation = new ArrayList< >();
    }
    
    public Double get(int r, int c) {
        return representation.get(r).get(c);
    }
    
    public void set(int r, int c, Double val) {
        representation.get(r).set(c, val);
    }
}
