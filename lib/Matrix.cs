
using System;

namespace nn.common
{
    public static class Matrix
    {
        // check if a matrix is valid
        public static bool IsValid(float[,] m){
            return ( m!=null &&  m.GetLength(0)>0 && m.GetLength(1)>0 );
        }

        // element wise operations between two matrixes: sum, sub, mult, div ... etc
        public static float[,] ewise(float[,] m1, float [,] m2, Func<float,float,float> func){
            
            // sanity checks
            if(!Matrix.IsValid(m1)) return null;
            if(!Matrix.IsValid(m2)) return null;

            int nRows_m1 = m1.GetLength(0);
            int nCols_m1 = m1.GetLength(1);
            int nRows_m2 = m2.GetLength(0);
            int nCols_m2 = m2.GetLength(1);
            
            float [,] res = new float [nRows_m1, nCols_m1];             

            if (nRows_m1 == nRows_m2 && nCols_m1 == nCols_m2){

                for(int row = 0; row<nRows_m1; row++){
                    for(int col = 0; col<nCols_m1; col++){
                        res[row, col] = func(m1[row, col],m2[row, col]);
                    }
                }
            }

            return res;
        }

        // dot product : sum of products
        public static float[,] dot (float[,] m1, float [,] m2){

            // sanity checks
            if(!Matrix.IsValid(m1)) return null;
            if(!Matrix.IsValid(m2)) return null;

            int nRows_m1 = m1.GetLength(0);
            int nCols_m1 = m1.GetLength(1);
            int nRows_m2 = m2.GetLength(0);
            int nCols_m2 = m2.GetLength(1);
            
            float [,] res = null;
            float sum;

            if ( nCols_m1 == nRows_m2){
                res = new float[nRows_m1, nCols_m2];
                for(int row = 0; row<nRows_m1; row++){
                    for(int col = 0; col<nCols_m2; col++){
                        sum = 0;
                        for(int e = 0; e<nCols_m1; e++){
                            sum += m1[row, e] * m2[e, col];
                        }
                        res[row, col] = sum;
                    }
                }
            }

            return res;
        }

        // dot product : sum of products
        public static void dot (float[,] m1, float [,] m2, ref float [,] res){

            // sanity checks
            if(!Matrix.IsValid(m1)) return;
            if(!Matrix.IsValid(m2)) return;
            if(!Matrix.IsValid(res)) return;
            if(res.GetLength(0) != m1.GetLength(0) || res.GetLength(1) != m2.GetLength(1)) return;

            int nRows_m1 = m1.GetLength(0);
            int nCols_m1 = m1.GetLength(1);
            int nRows_m2 = m2.GetLength(0);
            int nCols_m2 = m2.GetLength(1);
            
            float sum;

            if ( nCols_m1 == nRows_m2){
                for(int row = 0; row<nRows_m1; row++){
                    for(int col = 0; col<nCols_m2; col++){
                        sum = 0;
                        for(int e = 0; e<nCols_m1; e++){
                            sum += m1[row, e] * m2[e, col];
                        }
                        res[row, col] = sum;
                    }
                }
            }
        }

        // traspose matrix 2x3 to 3x2
        public static float[,] traspose(float [,] m){

            // sanity checks
            if(!Matrix.IsValid(m)) return null;

            int nRows = m.GetLength(0);
            int nCols = m.GetLength(1);
            float [,] res = new float[nCols, nRows];
        
            for(int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    res[col, row] = m[row, col];
                }
            }

            return res;
        }

        // escalar operations : mult, sum, div, sub , etc
        public static float[,]  map(float [,] m, Func<float,int,int,float> func){
            
            // sanity checks
            if(!Matrix.IsValid(m)) return null;

            int nRows = m.GetLength(0);
            int nCols = m.GetLength(1);
            float [,] res = new float[nRows, nCols];

            for(int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    res[row, col] = func(m[row,col], row, col);
                }
            }

            return res;
        }

        // escalar operations : mult, sum, div, sub , etc
        public static void  map(ref float [,] m, Func<float,int,int,float> func){
            
            // sanity checks
            if(!Matrix.IsValid(m)) return;

            int nRows = m.GetLength(0);
            int nCols = m.GetLength(1);

            for(int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    m[row, col] = func(m[row,col], row, col);
                }
            }

        }

        // copy array to a Matrix [1,N], return numbers of values written
        public static int ArrayToMatrix(float[] arr, ref float[,] m){
            
            // sanity checks
            if (!(arr!=null && arr.GetLength(0)>0)) return 0;
            if(!Matrix.IsValid(m)) return 0;

            int arrSize = arr.GetLength(0);
            
            for(int i = 0; i< arrSize; i++)
                m[0,i] = arr[i];

            return arrSize;
        }


        // simple
        public static string toString(float[,] m, string valueDelimiter = "\t\t", string rowDelimiter = "\n"){
            
            // sanity checks
            if(!Matrix.IsValid(m)) return null;

            
            int nRows = m.GetLength(0);
            int nCols = m.GetLength(1);
            string result = string.Format("\n[ {0} x {1} ]\n\n---\n\n",nRows, nCols);

            for(int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    result += m[row,col];
                    result += valueDelimiter;
                }
                result += rowDelimiter;
            }

            return result;
        }

    }
}
