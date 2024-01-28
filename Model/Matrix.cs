// #define COL_MAJOR // reverse memory ordering in matrix
using System;

namespace NeuralNet.Model
{
    public class Matrix
    {
        public int Rows, Columns;
        float[,] data;
        public void Resize(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
#if COL_MAJOR
            data = new float[Columns, Rows];
#else
            data = new float[Rows, Columns];
#endif
        }

        public void Randomize(Func<float> source)
        {
            for (var i = 0; i < Rows; ++i)
                for (var j = 0; j < Columns; ++j)
                    this[i, j] = source();
        }

        public float this[int row, int col]
        {
#if COL_MAJOR
            get { return data[col, row]; }
            set { data[col, row] = value; }
#else
            get { return data[row, col]; }
            set { data[row, col] = value; }
#endif
        }

        // in place times
        public static void Times(Vector ans, Matrix m, Vector v)
        {
            for (var i = 0; i < m.Rows; ++i)
            {
                ans[i] = 0;
                for (var j = 0; j < m.Columns; ++j)
                    ans[i] += m[i, j] * v[j];
            }
        }

        public static Vector operator *(Matrix m, Vector v)
        {
            var ans = new Vector(m.Rows);
            for (var i = 0; i < m.Rows; ++i)
            {
                ans[i] = 0;
                for (var j = 0; j < m.Columns; ++j)
                    ans[i] += m[i, j] * v[j];
            }
            return ans;
        }
        public static Vector operator *(Vector v, Matrix m)
        {
            var ans = new Vector(m.Columns);
            for (var j = 0; j < m.Columns; ++j)
            {
                ans[j] = 0;
                for (var i = 0; i < m.Rows; ++i)
                    ans[j] += v[i] * m[i, j];
            }
            return ans;
        }
    }
}
