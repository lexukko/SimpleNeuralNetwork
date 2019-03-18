using System;
using nn.common;

namespace common.nn{

        public static class NNUtils{
            public static float GetRandomNumber(float minimum, float maximum)
            { 
                Random random = new Random();
                return (float)random.NextDouble() * (maximum - minimum) + minimum;
            }    
        }

}