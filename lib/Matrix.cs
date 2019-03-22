
using System;

namespace nn.common
{
    public static class Matrix
    {

        // regresa matriz para el resultado de una operacion producto suma (dot)
        public static float[,] newInstance(float [,] m1, float [,] m2)
        {
            if (m1 == null)
                throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");

            if (m2 == null)
                throw new System.ArgumentNullException("Matriz 'm2' no puede ser nula.");

            int nRows_m1 = m1.GetLength(0);
            int nCols_m1 = m1.GetLength(1);
            int nRows_m2 = m2.GetLength(0);
            int nCols_m2 = m2.GetLength(1);

            if (!(nRows_m1 > 0 && nCols_m1 > 0 && nRows_m2 > 0 && nCols_m2 > 0))
                throw new System.ArgumentException("Matrices ('m1', 'm2') necesitan tener un tamaño minimo de un registro y una columna.");

            if (nCols_m1 != nRows_m2)
                throw new System.ArgumentException("Columnas de la matriz 'm1' deben ser iguales a los registros de la matriz 'm2'.");


            return new float[nRows_m1, nCols_m2];
        }

        // producto suma de dos matrices
        public static void dot (float[,] m1, float [,] m2, ref float [,] res){

            if ( m1 == null )
                throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");

            if ( m2 == null )
                throw new System.ArgumentNullException("Matriz 'm2' no puede ser nula.");

            if ( res == null )
                throw new System.ArgumentNullException("Matriz 'res' no puede ser nula.");

            int nRows_m1 = m1.GetLength(0);
            int nCols_m1 = m1.GetLength(1);
            int nRows_m2 = m2.GetLength(0);
            int nCols_m2 = m2.GetLength(1);
            int nRows_res = res.GetLength(0);
            int nCols_res = res.GetLength(1);

            if (!( nRows_m1 > 0 && nCols_m1>0 && nRows_m2 > 0 && nCols_m2 > 0 ))
                throw new System.ArgumentException("Matrices ('m1', 'm2') necesitan tener un tamaño minimo de un registro y una columna.");

            if ( nCols_m1 != nRows_m2 )
                throw new System.ArgumentException("Columnas de la matriz 'm1' deben ser iguales a los registros de la matriz 'm2'.");

            if (!(nRows_res == nRows_m1 && nCols_res == nCols_m2))
                throw new System.ArgumentException("Registros de la matriz 'm1' deben ser iguales a los registros de la matriz 'res' y las columnas de la matriz 'm2' deben ser iguales a las columnas de la matriz 'res'.");

            float sum = 0;

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

        // traspose matrix 2x3 to 3x2
        public static float[,] traspose(float [,] m1){

            if (m1 == null)
                throw new System.ArgumentException("Matriz 'm1' no puede ser nula.");

            int nRows = m1.GetLength(0);
            int nCols = m1.GetLength(1);

            if (!(nRows > 0 && nCols > 0 ))
                throw new System.ArgumentException("Matriz 'm1' necesita tener un tamaño minimo de un registro y una columna.");

            var res = new float[nCols, nRows];
        
            for(int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    res[col, row] = m1[row, col];
                }
            }

            return res;
        }

        // ejecuta funcion func para cada elemento de la matriz m1
        public static void  map(ref float [,] m1, Func<float,int,int,float> func){

            if (m1 == null)
                throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");

            if (func == null)
                throw new System.ArgumentNullException("Funcion 'func' no puede ser nula.");

            int nRows = m1.GetLength(0);
            int nCols = m1.GetLength(1);

            if (!(nRows > 0 && nCols > 0))
                throw new System.ArgumentException("Matriz 'm1' necesita tener un tamaño minimo de un registro y una columna.");

            for (int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    m1[row, col] = func(m1[row,col], row, col);
                }
            }

        }

        // float array to float Matrix [N,1]
        public static float[,] fromArray(float[] arr){

            if (arr == null)
                throw new System.ArgumentNullException("Array 'arr' no puede ser nulo.");

            int arrSize = arr.GetLength(0);

            if (!(arrSize > 0))
                throw new System.ArgumentException("Array 'arr' necesita un tamaño minimo de 1.");

            float[,] m1 = new float[arrSize,1];

            for(int i = 0; i< arrSize; i++)
                m1[i,0] = arr[i];

            return m1;
        }

        // float Matrix [N,1] to float array
        public static float[] toArray(float [,] m1) {

            if (m1 == null)
                throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");

            int nRows = m1.GetLength(0);
            int nCols = m1.GetLength(1);

            if (!(nRows > 0 && nCols > 0))
                throw new System.ArgumentException("Matriz 'm1' necesita tener un tamaño minimo de un registro y una columna.");

            float[] arr = new float[nRows];

            for (int i = 0; i < nRows; i++)
                arr[i] = m1[i,0];

            return arr;
        }


        // simple
        public static string toString(float[,] m1, string valueDelimiter = "\t\t", string rowDelimiter = "\n"){

            if (m1 == null)
                throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");

            int nRows = m1.GetLength(0);
            int nCols = m1.GetLength(1);

            if (!(nRows > 0 && nCols > 0))
                throw new System.ArgumentException("Matriz 'm1' necesita tener un tamaño minimo de un registro y una columna.");

            string result = string.Format("\n[ {0} x {1} ]\n\n---\n\n",nRows, nCols);

            for(int row = 0; row<nRows; row++){
                for(int col = 0; col<nCols; col++){
                    result += m1[row,col];
                    result += valueDelimiter;
                }
                result += rowDelimiter;
            }

            return result;
        }

    }
}
