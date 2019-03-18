using System;
using nn.common;

namespace common.nn{
    public class SimpleNeuralLayer{
        public int noInputs = 0;
        public float[,] inputs;

        public int noOutputs = 0;
        public float[,] outputs;

        public int noWeights = 0;
        public float[,] weights;

        public Func<float,float> ActivationFunc;

        public bool isValid = false; 

        
        public SimpleNeuralLayer(int _noInputs, int _noOutputs, Func<float,float> _ActivationFunc){ // # Outputs ==  # perceptrons ( neurons )

            // sanity checks
            if (_noInputs>0 && _noOutputs>0 && _ActivationFunc!=null) {
                
                // outpus matrix 1xN
                this.noInputs = _noInputs;
                this.inputs = new float[1, _noInputs];

                // outpus matrix 1xN
                this.noOutputs = _noOutputs;
                this.outputs = new float[1, _noOutputs];

                // weights matrix rows = noInputs , cols = noOutpus
                this.noWeights = this.noInputs * this.noOutputs;
                this.weights = new float[this.noInputs, this.noOutputs];

                // Activation Function
                this.ActivationFunc = _ActivationFunc;

                // randomize values between 0.0f - 1.0f
                Matrix.map(ref this.weights, (val,row,col) => {
                    return NNUtils.GetRandomNumber(0.0f,1.0f);
                });

                // valid layer
                this.isValid = true;
            }
        }

        public void set(float [,] data){ // inject data to this layer "FeedForward"
            
            // sanity checks
            if (this.isValid && data!=null && data.GetLength(1) == this.noInputs){

                // copy matrix data 1xN to this.inputs
                Matrix.map(ref this.inputs, (val,row, col) => {
                    return data[row, col];
                });

                // dot product inputs x weights
                Matrix.dot(this.inputs,this.weights, ref this.outputs);

                // perform activation function
                Matrix.map(ref this.outputs, (val,row,col)=>{
                    return this.ActivationFunc(val);
                });

            }

        }

    }

}
