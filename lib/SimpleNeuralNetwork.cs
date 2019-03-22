
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

        public SimpleNeuralNetwork() {
            layers = new List<SimpleNeuralLayer>();
        }

        // Adds a layer
        public void add(SimpleNeuralLayer layer) {

            if (layer == null)
                throw new System.ArgumentNullException("'layer' no puede ser nulo.");

            layers.Add(layer);

            if (layers.Count == 1) {
                this.xs_no = layer.noInputs;
                this.ys_no = layer.noOutputs;
                this.outputLayerNo = layers.Count - 1;
            } else if (layers.Count > 1) {
                this.ys_no = layer.noOutputs;
                this.outputLayerNo = layers.Count - 1;
            }

        }

        private void feedforward(float[,] xs) { // Nx1

            if (xs == null)
                throw new System.ArgumentNullException("'xs' no puede ser nulo.");

            int nRows_xs = xs.GetLength(0);

            if (nRows_xs != this.xs_no)
                throw new System.ArgumentException("numero de registros de 'xs' debe ser igual al numero de entradas de la capa.");

            if (this.layers.Count > 0)
            {
                for (int i = 0; i < this.layers.Count; i++) {
                    if (i == 0) {
                        this.layers[i].set(xs);
                    } else {
                        this.layers[i].set(this.layers[i - 1].outputs);
                    }
                }
            }

        }


        private void backpropagation(float[,] ys)
        {
            if (ys == null)
                throw new System.ArgumentNullException("'ys' no puede ser nulo.");

            int nRows_ys = ys.GetLength(0);

            if (nRows_ys != this.ys_no)
                throw new System.ArgumentException("numero de registros de 'ys' debe ser igual al numero de entradas de la capa.");

                if (this.layers.Count > 0) {
                    for (int i = 0; i < this.layers.Count; i++) {
                        if (i == 0) {
                            
                        } else { 
                            
                        }
                    }
                }
        }

        public void train(float[] xs, float [] ys){ // inputs and outputs are matrixes of 1xN

            if (!(this.layers.Count > 0))
                throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");

            this.feedforward(Matrix.fromArray(xs));
            //this.backpropagation(Matrix.fromArray(ys));
        }

        public float[] predict(float[] xs) // predict an output given an input
        {
            if (!(this.layers.Count > 0))
                throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");

            this.feedforward(Matrix.fromArray(xs));
            return Matrix.toArray(this.layers[this.outputLayerNo].outputs);
        }

        public void save(){ // Save neural network configuration and data (weigths, bias) to a json file maybe to a binary file ( ProtocolBuffers, FlatBuffers)

            if (!(this.layers.Count > 0))
                throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");
        }

        public void load(){ // load neural network configuration and data (weigths, bias) from a json file maybe from a binary file ( ProtocolBuffers, FlatBuffers)

            if (this.layers.Count > 0)
                throw new System.ArgumentException("La red neuronal necesita no tener capas para utilizar este metodo.");
        }

    }
}
