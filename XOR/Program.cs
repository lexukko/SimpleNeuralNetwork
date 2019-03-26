using System;
using System.Collections.Generic;
using common.nn;
using nn.common;

namespace nn
{
    class Program
    {

        static void Train_predict_and_save(string filepath) {

            // train & testing data
            var training_data = new Dictionary<int, Tuple<float[], float[]>>
            {   //                                               ys                  xs
                [0] = new Tuple<float[], float[]>(new float[] { 0, 1 }, new float[] { 1 }),
                [1] = new Tuple<float[], float[]>(new float[] { 1, 0 }, new float[] { 1 }),
                [2] = new Tuple<float[], float[]>(new float[] { 0, 0 }, new float[] { 0 }),
                [3] = new Tuple<float[], float[]>(new float[] { 1, 1 }, new float[] { 0 })
            };

            var snn = new SimpleNeuralNetwork(2, 0.4f, Activation.FunctionsEnum.Sigmoid);
            snn.Add(2);
            snn.Add(1);

            // train
            Console.WriteLine("Entrenamiento:\n");
            Random random = new Random();
            int j = 0;
            Console.WriteLine("Training ...");
            for (int i = 0; i < 100000; i++)
            {
                j = random.Next(4);
                snn.Train(training_data[j].Item1, training_data[j].Item2);
            }

            // predict
            Console.WriteLine("\nPredicciones:\n");
            for (int i = 0; i < 4; i++)
            {
                var res = snn.Predict(training_data[i].Item1);
                Console.WriteLine(string.Format("xs [ {0}, {1} ] = {2}", training_data[i].Item1[0], training_data[i].Item1[1], res[0]));
            }

            SimpleNeuralNetwork.Save(snn, filepath);
        }

        static void Load_and_predict(string filepath)
        {

            // train & testing data
            var training_data = new Dictionary<int, Tuple<float[], float[]>>
            {   //                                               ys                  xs
                [0] = new Tuple<float[], float[]>(new float[] { 0, 1 }, new float[] { 1 }),
                [1] = new Tuple<float[], float[]>(new float[] { 1, 0 }, new float[] { 1 }),
                [2] = new Tuple<float[], float[]>(new float[] { 0, 0 }, new float[] { 0 }),
                [3] = new Tuple<float[], float[]>(new float[] { 1, 1 }, new float[] { 0 })
            };

            var snn = SimpleNeuralNetwork.Load(filepath);

            // predict
            Console.WriteLine("\nPredicciones:\n");
            for (int i = 0; i < 4; i++)
            {
                var res = snn.Predict(training_data[i].Item1);
                Console.WriteLine(string.Format("xs [ {0}, {1} ] = {2}", training_data[i].Item1[0], training_data[i].Item1[1], res[0]));
            }

        }


        static void Main(string[] args)
        {

            Train_predict_and_save("xor.json");
            Load_and_predict("xor.json");
            Console.ReadKey();

        }
    }
}
