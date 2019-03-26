using System;
using nn.common;

namespace common.nn{
    public class SimpleNeuralLayer{

        public Matrix inputs;
        public Matrix errors;
        public Matrix outputs;
        public Matrix bias;
        public Matrix weights;

        public int NoInputs = 0;
        public int NoOutputs = 0;


        public SimpleNeuralLayer(int _noInputs, int _noOutputs){ // # Outputs ==  # perceptrons ( neurons )
            if (!( _noInputs > 0 && _noOutputs > 0)) throw new System.ArgumentException("Entradas y salidas de la capa ('_noInputs', '_noOutputs') deben ser mayores a cero.");

            NoInputs = _noInputs;
            NoOutputs = _noOutputs;

            inputs = new Matrix(NoInputs, 1);
            errors = new Matrix(NoOutputs, 1);
            outputs = new Matrix(NoOutputs, 1);
            bias = new Matrix(NoOutputs, 1);
            weights = new Matrix(NoOutputs, NoInputs);

            // randomiza valores de pesos y bias 0.01f - 1.0f
            weights.Map((v, r, c) => {
                return NNUtils.GetRandomNumber(-1.0f, 1.0f);
            });

            bias.Map((v, r, c) => {
                return NNUtils.GetRandomNumber(-1.0f, 1.0f);
            });
        }


        public void Feed(Matrix _inputs, Func<float, float> _func) {
            if ( _inputs == null ) throw new System.ArgumentNullException("Matriz '_inputs' no puede ser nula.");
            if ( _func == null) throw new System.ArgumentNullException("funcion '_func' no puede ser nula.");
            inputs.Copy(_inputs);
            // outputs = func ( weights x inputs + bias )
            outputs.Copy(Matrix.Dot(weights, inputs));
            outputs.Add(bias);
            outputs.Map((v, r, c) => {
                return _func(v);
            });
        }

    }
}
