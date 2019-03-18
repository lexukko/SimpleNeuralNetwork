
using System;

namespace nn.common
{
    public class SimpleNeuralNetwork
    {
        public int noInputs;
        public int noHidden;
        public int noOutput;

        // this is necesary cuz we want to initialize this class also with the method load
        public SimpleNeuralNetwork(){
            // empty
        }

        // var snn = new NeuralNetwork(2,4,2);
        public SimpleNeuralNetwork(int _noInputs, int _noHidden, int _noOutputs){
            noInputs = _noInputs;
            noHidden = _noHidden;
            noOutput = _noOutputs;
        }

        public void train(float[,] inputs, float[,] outputs){ // inputs and outputs are matrixes of 1xN
            // --> FeedFordward
            // <-- BackPropagation (LinearRegresion)
        }

        public void predict(){ // predict an output given an input

        }

        public void save(){ // Save neural network configuration and data (weigths, bias) to a json file maybe to a binary file ( ProtocolBuffers, FlatBuffers)

        }

        public void load(){ // load neural network configuration and data (weigths, bias) from a json file maybe from a binary file ( ProtocolBuffers, FlatBuffers)

        }

    }
}
