using System;

namespace common.nn{

        public static class ActivationFunctions {

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