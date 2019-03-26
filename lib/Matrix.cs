
using Newtonsoft.Json;
using System;

namespace nn.common
{
    public class Matrix
    {
        public float[,] data;
        public int nRows, nCols;

        [JsonConstructor]
        public Matrix() {

        }

        public Matrix(int _nRows, int _nCols) {
            if (!(_nRows > 0 && _nCols > 0)) throw new System.ArgumentException("Tamaño minimo aceptado 1x1.");
            nRows = _nRows;
            nCols = _nCols;
            data = new float[nRows, nCols];
        }

        public Matrix(float [,] m1){
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            nRows = m1.GetLength(0);
            nCols = m1.GetLength(1);
            if (!(nRows > 0 && nCols > 0)) throw new System.ArgumentException("Tamaño minimo aceptado 1x1.");
            data = new float[nRows, nCols];
            Map((v, r, c) => {
                return m1[r, c];
            });
        }

        public Matrix(Matrix m1){
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            nRows = m1.nRows;
            nCols = m1.nCols;
            data = new float[m1.nRows, m1.nCols];
            Copy(m1);
        }

        // Copy
        public void Copy(Matrix m1) {
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            if (!(m1.nRows == nRows && m1.nCols == nCols)) throw new System.ArgumentException("las dimensiones de 'm1' son distintas a las de la matriz interna.");
            Map((v, r, c) => {
                return m1.data[r, c];
            });
        }

        // Array[N] -> Matrix[N,1]
        public static Matrix FromArray(float[] arr) {
            if (arr == null) throw new System.ArgumentNullException("Array 'arr' no puede ser nulo.");
            int arrSize = arr.GetLength(0);
            if (!(arrSize > 0)) throw new System.ArgumentException("Array 'arr' necesita un tamaño minimo de 1.");
            var m1 = new Matrix(arrSize, 1); // Nx1
            for (int i = 0; i < arrSize; i++)
                m1.data[i, 0] = arr[i];
            return m1;
        }

        // this Matrix[N,1] -> Array[N]
        public float[] ToArray() {
            float[] arr = new float[nRows];
            for (int i = 0; i < nRows; i++)
                arr[i] = data[i, 0];
            return arr;
        }

        public static void Map(ref Matrix m1, Func<float, int, int, float> func) {
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            if (func == null) throw new System.ArgumentNullException("Funcion 'func' no puede ser nula.");
            for (int row = 0; row < m1.nRows; row++) {
                for (int col = 0; col < m1.nCols; col++) {
                    m1.data[row, col] = func(m1.data[row, col], row, col);
                }
            }
        }

        public void Map(Func<float, int, int, float> func) {
            if (func == null) throw new System.ArgumentNullException("Funcion 'func' no puede ser nula.");
            for (int row = 0; row < nRows; row++) {
                for (int col = 0; col < nCols; col++) {
                    data[row, col] = func(data[row, col], row, col);
                }
            }
        }

        // Matrix[R,C] -> Matrix[C,R]
        public static Matrix Transpose(Matrix m1) {
            if (m1 == null) throw new System.ArgumentException("Matriz 'm1' no puede ser nula.");
            var res = new Matrix(m1.nCols, m1.nRows);
            for(int row = 0; row<m1.nRows; row++){
                for(int col = 0; col<m1.nCols; col++){
                    res.data[col, row] = m1.data[row, col];
                }
            }
            return res;
        }

        // Matrix.ToString()
        public static string ToString(Matrix m1, string valueDelimiter = "\t\t", string rowDelimiter = "\n"){
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            string result = "";
            for(int row = 0; row<m1.nRows; row++){
                for(int col = 0; col<m1.nCols; col++){
                    result += m1.data[row,col];
                    result += valueDelimiter;
                }
                result += rowDelimiter;
            }
            return result;
        }

        // producto suma de dos matrices
        public static Matrix Dot(Matrix m1, Matrix m2)
        {
            if (m1 == null || m2 == null) throw new System.ArgumentNullException("No se admiten parametros nulos.");
            if (m1.nCols != m2.nRows) throw new System.ArgumentException("# Columnas de 'm1' diferente a # registros 'm2'.");
            float sum = 0;
            var res = new Matrix(m1.nRows, m2.nCols);
            for (int row = 0; row < m1.nRows; row++)
            {
                for (int col = 0; col < m2.nCols; col++)
                {
                    sum = 0;
                    for (int e = 0; e < m1.nCols; e++)
                        sum += m1.data[row, e] * m2.data[e, col];
                    res.data[row, col] = sum;
                }
            }
            return res;
        }

        // add, sub, mult, div element wise

        public void Add(Matrix m1) {
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            if (!(m1.nRows == nRows && m1.nCols == nCols)) throw new System.ArgumentException("las dimensiones de 'm1' son distintas a las de la matriz interna.");
            this.Map((v, r, c) => {
                return v + m1.data[r, c];
            });
        }

        public void Sub(Matrix m1) {
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            if (!(m1.nRows == nRows && m1.nCols == nCols)) throw new System.ArgumentException("las dimensiones de 'm1' son distintas a las de la matriz interna.");
            this.Map((v, r, c) => {
                return v - m1.data[r, c];
            });
        }

        public void Mult(Matrix m1) {
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            if (!(m1.nRows == nRows && m1.nCols == nCols)) throw new System.ArgumentException("las dimensiones de 'm1' son distintas a las de la matriz interna.");
            this.Map((v, r, c) => {
                return v * m1.data[r, c];
            });
        }

        public void Div(Matrix m1) {
            if (m1 == null) throw new System.ArgumentNullException("Matriz 'm1' no puede ser nula.");
            if (!(m1.nRows == nRows && m1.nCols == nCols)) throw new System.ArgumentException("las dimensiones de 'm1' son distintas a las de la matriz interna.");
            this.Map((v, r, c) => {
                return v / m1.data[r, c];
            });
        }

        // add, sub, mult, div escalar

        public void Add(float a) {
            this.Map((v, rows, cols) => {
                return v + a;
            });
        }

        public void Sub(float a) {
            this.Map((v, rows, cols) => {
                return v - a;
            });
        }

        public void Mult(float a) {
            this.Map((v, rows, cols) => {
                return v * a;
            });
        }

        public void Div(float a) {
            this.Map((v, rows, cols) => {
                return v / a;
            });
        }

    }
}
