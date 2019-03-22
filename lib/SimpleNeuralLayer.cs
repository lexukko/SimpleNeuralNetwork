using System;
using nn.common;

namespace common.nn{
    public class SimpleNeuralLayer{

        public float[,] inputs;
        public float[,] outputs;
        public float[,] bias; // mismo tamaño que outputs
        public float[,] weights;

        public Func<float,float> ActivationFunc;

        
        public SimpleNeuralLayer(int noInputs, int noOutputs, Func<float,float> func){ // # Outputs ==  # perceptrons ( neurons )

            if ( func == null)
                throw new System.ArgumentNullException("Funcion de activacion 'func' no puede ser nula.");

            if (!( noInputs > 0 && noOutputs > 0))
                throw new System.ArgumentException("Entradas y salidas de la capa ('noInputs', 'noOutputs') deben ser mayores a cero.");

            // inputs matrix Nx1
            this.inputs = new float[noInputs, 1];

            // outpus matrix Nx1
            this.outputs = new float[noOutputs, 1];

            // bias matrix Nx1
            this.bias = new float[noOutputs, 1];

            // weights matrix rows = noOutputs , cols = noInputs
            this.weights = new float[noOutputs, noInputs];

            // Activation Function
            this.ActivationFunc = func;

            // randomiza valores de pesos y bias 0.01f - 1.0f
            Matrix.map(ref this.weights, (val,row,col) => {
                return NNUtils.GetRandomNumber(0.01f,1.0f);
            });

            Matrix.map(ref this.bias, (val, row, col) => {
                return NNUtils.GetRandomNumber(0.01f, 1.0f);
            });

        }


        public void set(float [,] inputs){ // inject data to this layer "FeedForward" Nx1

            if ( inputs == null )
                throw new System.ArgumentNullException("Matriz 'inputs' no puede ser nula.");

            int nRows_inputs_param = inputs.GetLength(0);
            int nCols_inputs_param = inputs.GetLength(1);

            int nRows_inputs = this.inputs.GetLength(0);
            int nCols_inputs = this.inputs.GetLength(1);

            if (!(nRows_inputs_param == nRows_inputs && nCols_inputs_param == nCols_inputs))
                throw new System.ArgumentException(string.Format("Las dimensiones de la matriz 'inputs' deben ser [ {0} x {1} ]", nRows_inputs, nCols_inputs));

            // copia parametro inputs a this.inputs
            Matrix.map(ref this.inputs, (input, row, col) => {
                return inputs[row, col];
            });

            // dot product outputs = weights x inputs + bias
            Matrix.dot(this.weights, this.inputs, ref this.outputs);

            Matrix.map(ref this.outputs, (weight, row, col) => {
                return weight + this.bias[row, col];
            });

            // perform activation function
            Matrix.map(ref this.outputs, (output, row,col)=>{
                return this.ActivationFunc(output);
            });

        }

        // getters numero de entradas (noInputs) y numero de salidas (noOutputs)
        public int noInputs { get => this.inputs.GetLength(0); }
        public int noOutputs { get => this.outputs.GetLength(0); }

    }

}
