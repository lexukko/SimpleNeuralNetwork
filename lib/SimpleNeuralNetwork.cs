
using System;
using System.Collections.Generic;
using common.nn;

namespace nn.common
{
    public class SimpleNeuralNetwork
    {
        public int NoInputs;
        public int NoOutputs;
        public int outputLayerNo;
        public float lr;
        public List<SimpleNeuralLayer> layers;

        public SimpleNeuralNetwork(float _lr) {
            layers = new List<SimpleNeuralLayer>();
            lr = _lr;
        }

        public void Add(SimpleNeuralLayer layer) {
            if (layer == null) throw new System.ArgumentNullException("'layer' no puede ser nulo.");
            layers.Add(layer);
            if (layers.Count == 1) {
                NoInputs = layer.NoInputs;
                NoOutputs = layer.NoOutputs;
                outputLayerNo = layers.Count - 1;
            } else if (layers.Count > 1) {
                NoOutputs = layer.NoOutputs;
                outputLayerNo = layers.Count - 1;
            }
        }

        private void FeedFordward(Matrix inputs) { // Nx1
            if (inputs == null) throw new System.ArgumentNullException("'inputs' no puede ser nulo.");
            if (inputs.nRows != NoInputs) throw new System.ArgumentException("numero de registros de 'inputs' debe ser igual al numero de entradas de la capa.");
            if (!(layers.Count > 0)) throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");

            for (int i = 0; i < layers.Count; i++) {
                if (i == 0){
                    layers[i].Feed(inputs);
                } else {
                    layers[i].Feed(layers[i - 1].outputs);
                }
            }
        }

        //                     gradiente
        // deltas = lr * E * dfunc(outputs) * Matrix.traspose(inputs)
        private void BackPropagation(Matrix targets){
            if (targets == null) throw new System.ArgumentNullException("'outputs' no puede ser nulo.");
            if (targets.nRows != NoOutputs) throw new System.ArgumentException("numero de registros de 'outputs' debe ser igual al numero de entradas de la capa.");
            for (int i = outputLayerNo; i >= 0; i--) {

                if (i == outputLayerNo){
                    layers[i].errors.Copy(targets);
                    layers[i].errors.Sub(layers[i].outputs);
                } else {
                    var prev_weights = Matrix.Transpose(layers[i + 1].weights);
                    layers[i].errors.Copy(Matrix.Dot(prev_weights, layers[i+1].errors));
                }

                var gradients = new Matrix(layers[i].outputs);
                gradients.Map((v, r, c) => {
                    return layers[i].dfunc(v);
                });
                gradients.Mult(layers[i].errors);
                gradients.Mult(lr);
                var deltas = Matrix.Dot(gradients, Matrix.Transpose(layers[i].inputs));
                // Ajustar pesos de la capa
                layers[i].weights.Add(deltas);
                // Ajustas bias de la capa
                layers[i].bias.Add(gradients);

            }
        }

        public void Train(float [] inputs, float [] targets){
            if (!(this.layers.Count > 0)) throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");
            FeedFordward(Matrix.FromArray(inputs));
            BackPropagation(Matrix.FromArray(targets));
        }

        public float[] Predict(float[] inputs){
            if (!(this.layers.Count > 0)) throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");

            FeedFordward(Matrix.FromArray(inputs));
            return layers[outputLayerNo].outputs.ToArray();
        }

        public void Save() {
            if (!(this.layers.Count > 0)) throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");
        }

        public void Load() {
            if (this.layers.Count > 0) throw new System.ArgumentException("La red neuronal necesita no tener capas para utilizar este metodo.");
        }

    }
}
