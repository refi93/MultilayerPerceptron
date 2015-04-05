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
public class NeuronMemento {
    Matrix weightsHidden, weightsOut;
    double validationError;
    
    public NeuronMemento(Matrix weightsHidden, Matrix weightsOut, double validationError) {
        this.weightsHidden = Helpers.matrixDeepCopy(weightsHidden);
        this.weightsOut = Helpers.matrixDeepCopy(weightsOut);
        this.validationError = validationError;
    }
}
