using System;
using common.nn;
using nn.common;

namespace nn
{
    class Program
    {
        static void Main(string[] args)
        {
            // matrix cols x rows
            // m1 = 1x2 , m2 = 2x2 : res = m1.rows x m2.cols  = 1x2
            //float[,] m1 = { { 1, 2 } };
            //float[,] m2 = { { 1, 2 }, { 3, 4 } };
            //var res1 = Matrix.newInstance(m1, m2);
            //Matrix.dot(m1, m2, ref res1);
            //var m3 = Matrix.traspose(m1);
            //var m4 = Matrix.traspose(m2);
            //var res2 = Matrix.newInstance(m4, m3);
            //Matrix.dot(m4, m3, ref res2);
            //Console.WriteLine(Matrix.toString(res1));
            //Console.WriteLine(Matrix.toString(res2));

            var snn = new SimpleNeuralNetwork();
            snn.add(new SimpleNeuralLayer(2, 4, ActivationFunctions.Sigmoid));
            snn.add(new SimpleNeuralLayer(4, 4, ActivationFunctions.Sigmoid));

            snn.train(new float[] { 1, 0 }, new float[] { 0, 0 });

            Console.WriteLine(Matrix.toString(snn.layers[0].inputs));
            Console.WriteLine(Matrix.toString(snn.layers[0].weights));
            Console.WriteLine(Matrix.toString(snn.layers[0].outputs));


            Console.WriteLine(Matrix.toString(snn.layers[1].inputs));
            Console.WriteLine(Matrix.toString(snn.layers[1].weights));
            Console.WriteLine(Matrix.toString(snn.layers[1].outputs));


            Console.ReadKey();

        }
    }
}
