
using System;
using System.Collections.Generic;
using System.IO;
using common.nn;
using Newtonsoft.Json;
using static common.nn.Activation;

namespace nn.common
{
    public class SimpleNeuralNetwork
    {
        public int NoInputs;
        public int NoOutputs;
        public int outputLayerNo;
        public float learning_rate;
        public FunctionsEnum ActivationFunction;

        public List<SimpleNeuralLayer> layers;
        

        [JsonIgnore]
        public Func<float, float> func;
        [JsonIgnore]
        public Func<float, float> dfunc;

        [JsonConstructor]
        public SimpleNeuralNetwork() {
        }

        public SimpleNeuralNetwork(int _NoInputs, float _learning_rate, FunctionsEnum _ActivationFunction) {
            //if (_func == null) throw new System.ArgumentNullException("funcion '_func' no puede ser nula.");
            //if (_dfunc == null) throw new System.ArgumentNullException("funcion '_dfunc' no puede ser nula.");
            if (!(_NoInputs > 0)) throw new System.ArgumentException("'_NoInputs' debe ser mayor a 0.");
            if (!(_learning_rate > 0.0f && _learning_rate <= 1.0f)) throw new System.ArgumentException("'_learning_rate' debe ser un valor mayor a 0.0 y menor o igual a 1.0 .");

            layers = new List<SimpleNeuralLayer>();
            learning_rate = _learning_rate;
            ActivationFunction = _ActivationFunction;
            func = Activation.FunctionsLst[(int)ActivationFunction].Item1;
            dfunc = Activation.FunctionsLst[(int)ActivationFunction].Item2;
            NoInputs = _NoInputs;
        }

        public void Add(int _NoOutputs) {
            if (!(_NoOutputs > 0)) throw new System.ArgumentException("'_NoOutputs' debe ser mayor a 0.");
            if (layers.Count == 0){
                layers.Add(new SimpleNeuralLayer(NoInputs, _NoOutputs));
            } else {
                layers.Add(new SimpleNeuralLayer(layers[layers.Count-1].NoOutputs, _NoOutputs));
            }
            NoOutputs = _NoOutputs;
            outputLayerNo = layers.Count - 1;
        }

        private void FeedFordward(Matrix inputs) { // Nx1
            if (inputs == null) throw new System.ArgumentNullException("'inputs' no puede ser nulo.");
            if (inputs.nRows != NoInputs) throw new System.ArgumentException("numero de registros de 'inputs' debe ser igual al numero de entradas de la capa.");
            if (!(layers.Count > 0)) throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");

            for (int i = 0; i < layers.Count; i++) {
                if (i == 0){
                    layers[i].Feed(inputs, func);
                } else {
                    layers[i].Feed(layers[i - 1].outputs, func);
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
                    return dfunc(v);
                });
                gradients.Mult(layers[i].errors);
                gradients.Mult(learning_rate);
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

        public static void Save(SimpleNeuralNetwork snn, String filepath) {
            if (!(snn.layers.Count > 0)) throw new System.ArgumentException("La red neuronal necesita almenos una capa para poder operar.");
            File.WriteAllText(filepath, JsonConvert.SerializeObject(snn, Formatting.Indented));
        }

        public static SimpleNeuralNetwork Load(String filepath) {
            SimpleNeuralNetwork snn = JsonConvert.DeserializeObject<SimpleNeuralNetwork>(File.ReadAllText(filepath));
            
            // manualmente inicializa funciones activacion por que al deserializar solo traemos el enum
            snn.func = Activation.FunctionsLst[(int)snn.ActivationFunction].Item1;
            snn.dfunc = Activation.FunctionsLst[(int)snn.ActivationFunction].Item2;
            
            // incializa manualmente las matrices inputs, errors, outputs de cada layer
            foreach(SimpleNeuralLayer snl in snn.layers) {
                snl.inputs = new Matrix(snl.NoInputs, 1);
                snl.errors = new Matrix(snl.NoOutputs, 1);
                snl.outputs = new Matrix(snl.NoOutputs, 1);
            }

            return snn;
        }

    }
}
