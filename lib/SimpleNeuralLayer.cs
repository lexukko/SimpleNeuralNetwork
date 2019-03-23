using System;
using nn.common;

namespace common.nn{
    public class SimpleNeuralLayer{

        public Matrix inputs;
        public Matrix outputs;
        public Matrix bias;
        public Matrix weights;

        public Func<float,float>   func;
        public Func<float, float> dfunc;


        public SimpleNeuralLayer(int noInputs, int noOutputs, Func<float,float> _func, Func<float, float> _dfunc){ // # Outputs ==  # perceptrons ( neurons )
            if ( _func == null) throw new System.ArgumentNullException("Funcion de activacion '_func' no puede ser nula.");
            if ( _dfunc == null) throw new System.ArgumentNullException("Funcion de activacion '_dfunc' no puede ser nula.");
            if (!( noInputs > 0 && noOutputs > 0)) throw new System.ArgumentException("Entradas y salidas de la capa ('noInputs', 'noOutputs') deben ser mayores a cero.");

            inputs = new Matrix(noInputs, 1);
            outputs = new Matrix(noOutputs, 1);
            bias = new Matrix(noOutputs, 1);
            weights = new Matrix(noOutputs, noInputs);
            func = _func;
            dfunc = _dfunc;

            // randomiza valores de pesos y bias 0.01f - 1.0f
            weights.Map((v, r, c) => {
                return NNUtils.GetRandomNumber(-1.0f, 1.0f);
            });

            bias.Map((v, r, c) => {
                return NNUtils.GetRandomNumber(-1.0f, 1.0f);
            });
        }


        public void Feed(Matrix _inputs) {
            if ( inputs == null ) throw new System.ArgumentNullException("Matriz 'inputs' no puede ser nula.");
            inputs.Copy(_inputs);
            // outputs = func ( weights x inputs + bias )
            outputs.Copy(Matrix.Dot(weights,inputs));
            outputs.Add(bias);
            outputs.Map((v, r, c) => {
                return func(v);
            });
        }

        // getters numero de entradas (noInputs) y numero de salidas (noOutputs)
        public int NoInputs { get => inputs.nRows; }
        public int NoOutputs { get => outputs.nRows; }

    }

}
