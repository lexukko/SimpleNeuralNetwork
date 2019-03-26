using System;
using System.Collections.Generic;

namespace common.nn{

        public static class Activation {

        public enum FunctionsEnum : byte {
            Sigmoid = 0
        };

        // list of tuples

        public static Dictionary<int, Tuple<Func<float, float>, Func<float, float>>> FunctionsLst = new Dictionary<int, Tuple<Func<float, float>, Func<float, float>>>
        {
            [0] = new Tuple<Func<float, float>, Func<float, float>>( Sigmoid, dSigmoid )
        };

        // sigmoid

        public static float Sigmoid(float x) {
            return (float) (1.0/(1.0 + Math.Exp(-x)));
        }

        public static float dSigmoid(float x) { // x ya tiene aplicado sigmoid entonces sigmoid(x) = x
            return x * (1 - x);
        }

            // tanh

            


    }
}