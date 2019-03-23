using System;
using System.Collections.Generic;
using common.nn;
using nn.common;

namespace nn
{
    class Program
    {
        static void Main(string[] args)
        {


            var snn = new SimpleNeuralNetwork(0.1f);
            snn.add(new SimpleNeuralLayer(2, 4, ActivationFunctions.Sigmoid, ActivationFunctions.dSigmoid));
            snn.add(new SimpleNeuralLayer(4, 1, ActivationFunctions.Sigmoid, ActivationFunctions.dSigmoid));

            // train
            var training_data = new Dictionary<int, Tuple<float[],float[]>>();
            training_data[0] = new Tuple<float[], float[]> ( new float[] { 0, 0 }, new float[] { 0 } );
            training_data[1] = new Tuple<float[], float[]> ( new float[] { 0, 1 }, new float[] { 1 });
            training_data[2] = new Tuple<float[], float[]> ( new float[] { 1, 0 }, new float[] { 1 });
            training_data[3] = new Tuple<float[], float[]> ( new float[] { 1, 1 }, new float[] { 0 });

            Console.WriteLine("Entrenamiento:\n");
            Random random = new Random();
            int j = 0;
            for (int i = 0; i < 10000; i++) {
                j =  random.Next(4);
                //Console.WriteLine(string.Format("xs [ {0}, {1} ] = ys [ {2} ]", training_data[j].Item1[0], training_data[j].Item1[1], training_data[j].Item2[0]));
                snn.train(training_data[j].Item1, training_data[j].Item2);
            }

            // predict
            Console.WriteLine("\nPredicciones:\n");
            for (int i = 0; i < 4; i++) {
                snn.predict(training_data[i].Item1);
                Console.WriteLine(string.Format("xs [ {0}, {1} ] = {2}", training_data[i].Item1[0], training_data[i].Item1[1], Matrix.toString(snn.layers[1].outputs)));
            }


            Console.ReadKey();

        }
    }
}
