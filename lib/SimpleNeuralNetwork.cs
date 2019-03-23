
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

        float learning_rate;

        public List<SimpleNeuralLayer> layers;

        public SimpleNeuralNetwork(float learning_rate) {
            layers = new List<SimpleNeuralLayer>();
            this.learning_rate = learning_rate;
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
            int nCols_ys = ys.GetLength(1);

            if (nRows_ys != this.ys_no)
                throw new System.ArgumentException("numero de registros de 'ys' debe ser igual al numero de entradas de la capa.");

            for (int i = this.layers.Count - 1; i >= 0; i--) {

                var errors = new float[nRows_ys, nCols_ys];

                // errors = ys - outputs
                Matrix.map(ref errors, (error, row, col) => {
                    return ys[row, col] - this.layers[i].outputs[row, col];
                });

                // ultimo layer
                if (i == (this.layers.Count - 1)) {

                    //                     gradiente
                    // deltas = lr * E * dfunc(outputs) * Matrix.traspose(inputs)

                    // gradiente = Matrix.map
                    var gradients = (float[,]) this.layers[i].outputs.Clone();
                    Matrix.map(ref gradients, (g, row, col) => {
                        return this.layers[i].dfunc(g);
                    });

                    // gradiente * error * learning_rate
                    Matrix.map(ref gradients, (g, row, col) => {
                        return g * errors[row, col] * this.learning_rate;
                    });

                    // gradiente * Matrix.traspose(inputs)
                    var deltas = Matrix.newInstance(gradients, Matrix.traspose(this.layers[i].inputs)); // reserva matriz para operacion suma producto
                    Matrix.dot(gradients, Matrix.traspose(this.layers[i].inputs), ref deltas);

                    // ajustar pesos con deltas, suma delta a cada peso
                    Matrix.map(ref this.layers[i].weights, (w, row, col) => {
                        return w + deltas[row, col];
                    });

                    // ajustar bias con gradiente, suma gradiente a cada bias
                    Matrix.map(ref this.layers[i].bias, (b, row, col) => {
                        return b + gradients[row, col];
                    });


                } else {
                    // layer n - 1

                    // error para sublayers
                    var errors_2 = Matrix.newInstance(Matrix.traspose(this.layers[i+1].weights), errors);
                    Matrix.dot(Matrix.traspose(this.layers[i + 1].weights), errors, ref errors_2);

                    // gradiente
                    var gradients = (float[,])this.layers[i].outputs.Clone();
                    Matrix.map(ref gradients, (g, row, col) => {
                        return this.layers[i].dfunc(g);
                    });

                    // gradiente * error * learning_rate
                    Matrix.map(ref gradients, (g, row, col) => {
                        return g * errors_2[row, col] * this.learning_rate;
                    });

                    // gradiente * Matrix.traspose(inputs)
                    var deltas = Matrix.newInstance(gradients, Matrix.traspose(this.layers[i].inputs)); // reserva matriz para operacion suma producto
                    Matrix.dot(gradients, Matrix.traspose(this.layers[i].inputs), ref deltas);

                    // ajustar pesos con deltas, suma delta a cada peso
                    Matrix.map(ref this.layers[i].weights, (w, row, col) => {
                        return w + deltas[row, col];
                    });

                    // ajustar bias con gradiente, suma gradiente a cada bias
                    Matrix.map(ref this.layers[i].bias, (b, row, col) => {
                        return b + gradients[row, col];
                    });

                }
            }

        }

        public void train(float[] xs, float [] ys){ // inputs and outputs are matrixes of 1xN

            if (!(this.layers.Count > 0))
                throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");

            this.feedforward(Matrix.fromArray(xs));
            this.backpropagation(Matrix.fromArray(ys));
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
