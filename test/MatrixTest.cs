using System;
using System.Collections.Generic;
using nn.common;
using Xunit;

namespace test
{
    public class MatrixTest
    {
        // data Generators
        // https://www.mathsisfun.com/algebra/matrix-multiplying.html

        public static IEnumerable<object[]> traspose_data =>
            new List<object[]>
            {
                new object[] { new float[,]{{1},{1}}            , new float[,]{{1,1}}               },
                new object[] { new float[,]{{1,2,3},{4,5,6}}    , new float[,]{{1,4},{2,5},{3,6}}   }
            };

        public static IEnumerable<object[]> dot_data =>
            new List<object[]>
            {
                new object[] { new float[,]{{1,2},{3,4}}        , new float[,]{{2,0},{1,2}}, new float[,]{{4,4},{10,8}}                 },
                new object[] { new float[,]{{2,0},{1,2}}        , new float[,]{{1,2},{3,4}}, new float[,]{{2,4},{7,10}}                 },
                new object[] { new float[,]{{1,2,3},{4,5,6}}    , new float[,]{{7,8},{9,10},{11,12}}, new float[,]{{58,64},{139,154}}   }
            };
        

        // Facts

        [Fact]
        public void Matrix_Escalar_Mult_Check()
        {
            Matrix ma = new Matrix(new float[,] { {1,2,3},{4,5,6}});
            float [,] b = { { 2, 4, 6 }, { 8, 10, 12 } };

            ma.Mult(2);        
            Assert.Equal(ma.data, b); // escalar multiply
        }



        // Theorys
        
        [Theory]
        [MemberData(nameof(traspose_data))]
        public void Matrix_traspose_check(float[,] a,float[,] b)
        {
            Matrix ma = new Matrix(a);
            Matrix mb = new Matrix(b);
            Assert.Equal(a, Matrix.Transpose(mb).data);
            Assert.Equal(b, Matrix.Transpose(ma).data);
        }

        [Theory]
        [MemberData(nameof(dot_data))]
        public void Matrix_dot_check(float[,] a,float[,] b, float[,] result)
        {
            Matrix ma = new Matrix(a);
            Matrix mb = new Matrix(b);
            Matrix mresult = new Matrix(result);
            Assert.Equal(Matrix.Dot(ma,mb).data, result);
        }

    }
}
