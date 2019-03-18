
using System;
using System.Collections.Generic;
using common.nn;

namespace nn.common
{
    public class SimpleNeuralNetwork
    {
        public float[,] inputdata;
        public int noInputs=0;

        public List<SimpleNeuralLayer> layers;

        // this is necesary cuz we want to initialize this class also with the method load
        public SimpleNeuralNetwork(int _noInputs){
            layers = new List<SimpleNeuralLayer>();
            this.noInputs = _noInputs;
            this.inputdata = new float[1,_noInputs];
        }

        // Adds a layer
        public void add(SimpleNeuralLayer layer){
            if(layer!=null)
                layers.Add(layer);
        }


        public void train(float[] inputs){ // inputs and outputs are matrixes of 1xN
            if(inputs!=null && inputs.GetLength(0) == this.noInputs){
                
                // --> FeedFordward
                Matrix.ArrayToMatrix(inputs, ref this.inputdata);
                if(this.layers.Count>0){
                    for(int i = 0; i<this.layers.Count; i++){
                        if(i == 0){
                            this.layers[i].set(this.inputdata);
                        } else {
                            this.layers[i].set(this.layers[i-1].outputs);
                        }
                    }
                }

                // <-- BackPropagation (LinearRegresion)

            }
            
            
        }

        public void predict(){ // predict an output given an input

        }

        public void save(){ // Save neural network configuration and data (weigths, bias) to a json file maybe to a binary file ( ProtocolBuffers, FlatBuffers)

        }

        public void load(){ // load neural network configuration and data (weigths, bias) from a json file maybe from a binary file ( ProtocolBuffers, FlatBuffers)

        }

    }
}
