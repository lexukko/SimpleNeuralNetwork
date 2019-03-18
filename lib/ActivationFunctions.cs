using System;

namespace common.nn{

        public static class ActivationFunctions {
            public static float Sigmoid(float value)
            {
                return (float) (1.0/(1.0 + Math.Exp(-value)));
            }
        }
}