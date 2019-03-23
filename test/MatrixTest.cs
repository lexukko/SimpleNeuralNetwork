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

        public static IEnumerable<object[]> isvalid_data =>
            new List<object[]>
            {
                new object[] { null                             , false   },
                new object[] { new float[,]{}                   , false   },
                new object[] { new float[,]{{}}                 , false   },
                new object[] { new float[,]{{},{}}              , false   },
                new object[] { new float[,]{{1},{1}}            , true    },
                new object[] { new float[,]{{1,1}}              , true    },
                new object[] { new float[,]{{1,2,3},{4,5,6}}    , true    },
                new object[] { new float[,]{{1,4},{2,5},{3,6}}  , true    }
            };

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
        public void Matrix_map_check()
        {
            float [,] a = {{1,2,3},{4,5,6}};
            float [,] b = {{2,4,6},{8,10,12}};
            float [,] c = {{2,3,4},{5,6,7}};

            var res1 = Matrix.map(a, (val, row, col) => {
                return val * 2;
            });
            
            Assert.Equal(b, res1); // Escalar matrix multiplication

            var res2 = Matrix.map(b, (val, row, col) => {
                return val / 2;
            });

            Assert.Equal(a, res2); // Escalar matrix division


            var res3 = Matrix.map(a, (val, row, col) => {
                return val + 1;
            });

            Assert.Equal(c, res3); // Escalar matrix adition

            var res4 = Matrix.map(c, (val, row, col) => {
                return val - 1;
            });

            Assert.Equal(a, res4); // Escalar matrix subtraction
            

        }

        [Fact]
        public void Matrix_ewise_check()
        {
            float [,] a = {{1,2,3},{4,5,6}};
            float [,] b = {{2,4,6},{8,10,12}};
            float [,] c = {{3,6,9},{12,15,18}};

            var res = Matrix.ewise(a,b,(val1, val2) => { // element wise adition
                return val1+ val2;
            });

            Assert.Equal(c, res);
        }

        [Fact]
        public void Matrix_ArrayToMatrix_check()
        {
            float [] a1 = null;
            float [,] m1 = null;
            Assert.Equal(0, Matrix.ArrayToMatrix(a1,ref m1));

            float [] a2 = {};
            float [,] m2 = {{}};
            Assert.Equal(0, Matrix.ArrayToMatrix(a2,ref m2));

            float [] a3 = {1};
            float [,] m3 = {{0}};
            Assert.Equal(1, Matrix.ArrayToMatrix(a3,ref m3));
            Assert.Equal(m3, new float[,]{{1}});

            float [] a4 = {1,2};
            float [,] m4 = {{0,0}};
            Assert.Equal(2, Matrix.ArrayToMatrix(a4,ref m4));
            Assert.Equal(m4, new float[,]{{1,2}});

        }

        // Theorys
        
 
        [Theory]
        [MemberData(nameof(isvalid_data))]
        public void Matrix_isvalid_check(float[,] a, bool b)
        {
            Assert.Equal(Matrix.IsValid(a), b);
        }

        [Theory]
        [MemberData(nameof(traspose_data))]
        public void Matrix_traspose_check(float[,] a,float[,] b)
        {
            Assert.Equal(a, Matrix.Traspose(b));
            Assert.Equal(b, Matrix.Traspose(a));
        }

        [Theory]
        [MemberData(nameof(dot_data))]
        public void Matrix_dot_check(float[,] a,float[,] b, float[,] result)
        {
            Assert.Equal(Matrix.dot(a,b), result);
        }

    }
}
