
using System;
using System.Collections.Generic;
using common.nn;

namespace nn.common
{
    public class SimpleNeuralNetwork
    {
        public int xs_no = 0;           // Number of inputs
        public int ys_no = 0;           // Number of outpus
        public int outputLayerNo = 0;   // index of last layer

        public List<SimpleNeuralLayer> layers;

        public SimpleNeuralNetwork(){
            layers = new List<SimpleNeuralLayer>();
        }

        // Adds a layer
        public void add(SimpleNeuralLayer layer){
            if (layer != null) {
                layers.Add(layer);
                if (layers.Count == 1){
                    this.xs_no = layer.noInputs;
                    this.ys_no = layer.noOutputs;
                    this.outputLayerNo = layers.Count - 1;
                } else if (layers.Count > 1) {
                    this.ys_no = layer.noOutputs;
                    this.outputLayerNo = layers.Count - 1;
                }
            }
                
        }

        private void feedforward(float[] xs) {
            if (xs != null && xs.GetLength(0) == this.xs_no )
            {
                if (this.layers.Count > 0)
                {
                    for(int i = 0; i<this.layers.Count; i++){
                        if(i == 0){
                            this.layers[i].set(Matrix.ArrayToMatrix(xs));
                        } else {
                            this.layers[i].set(this.layers[i-1].outputs);
                        }
                    }
                }
            }
        }

        private void backpropagation(float[] ys)
        {
            if ( ys != null && ys.GetLength(0) == this.ys_no)
            {
                if (this.layers.Count > 0)
                {
                    for (int i = 0; i < this.layers.Count; i++)
                    {
                        if (i == 0)
                        {
                            
                        }
                        else
                        {
                            
                        }
                    }
                }
            }
        }

        public void train(float[] xs, float [] ys){ // inputs and outputs are matrixes of 1xN
            this.feedforward(xs);
            this.backpropagation(ys);
        }

        public float[] predict(float[] xs) // predict an output given an input
        {
            if (this.layers.Count > 0) {
                this.feedforward(xs);
                return Matrix.MatrixToArray(this.layers[this.outputLayerNo].outputs);
            }
            return null;
        }

        public void save(){ // Save neural network configuration and data (weigths, bias) to a json file maybe to a binary file ( ProtocolBuffers, FlatBuffers)

        }

        public void load(){ // load neural network configuration and data (weigths, bias) from a json file maybe from a binary file ( ProtocolBuffers, FlatBuffers)

        }

    }
}
