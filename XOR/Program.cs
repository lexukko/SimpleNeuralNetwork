using System;
using common.nn;
using nn.common;

namespace nn
{
    class Program
    {


 

        static void Main(string[] args)
        {

            // Layer example

            /*
             
            var snl = new SimpleNeuralLayer(2, 4, ActivationFunctions.Sigmoid);
            snl.set(new float[,]{{1,2}});

            Console.WriteLine(Matrix.toString(snl.inputs));
            Console.WriteLine(Matrix.toString(snl.weights));
            Console.WriteLine(Matrix.toString(snl.outputs));
            */

            var snn = new SimpleNeuralNetwork();
            snn.add(new SimpleNeuralLayer(2,4,ActivationFunctions.Sigmoid));
            snn.add(new SimpleNeuralLayer(4,4,ActivationFunctions.Sigmoid));

            snn.train(new float[] {1,0}, new float[] { 0, 0 });
            
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
