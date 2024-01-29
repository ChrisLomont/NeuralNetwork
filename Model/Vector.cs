using System;
using System.Text;

namespace Lomont.NeuralNet.Model
{
    public class Vector
    {
        public int Size { get; private set; }
        float[] data;

        public Vector(int size)
        {
            Resize(size);
        }
        public Vector(params float[] values)
        {
            Resize(values.Length);
            Array.Copy(values, data, Size);
        }

        public void Resize(int size)
        {
            Size = size;
            data = new float[Size];
        }
        public void Randomize(Func<float> source)
        {
            for (var i = 0; i < Size; ++i)
                data[i] = source();
        }

        public static Vector operator *(Vector v, float scalar)
        {
            var ans = new Vector(v.Size);
            for (var i = 0; i < v.Size; ++i)
                ans[i] = v[i] * scalar;
            return ans;
        }

        public static Vector operator *(float scalar, Vector v)
        {
            return v * scalar;
        }

        public static void Add(Vector ans, Vector delta)
        {
            for (var i = 0; i < ans.Size; ++i)
                ans[i] += delta[i];
        }
        public static Vector operator +(Vector left, Vector right)
        {
            var ans = new Vector(left.Size);
            for (var i = 0; i < ans.Size; ++i)
                ans[i] = left[i] + right[i];
            return ans;
        }
        public static Vector operator -(Vector left, Vector right)
        {
            var ans = new Vector(left.Size);
            for (var i = 0; i < ans.Size; ++i)
                ans[i] = left[i] - right[i];
            return ans;
        }

        public static Vector ComponentTimes(Vector left, Vector right)
        {
            var ans = new Vector(left.Size);
            for (var i = 0; i < ans.Size; ++i)
                ans[i] = left[i] * right[i];
            return ans;
        }

        public float LengthSquared()
        {
            var sum = 0.0f;
            foreach (var v in data)
                sum += v * v;
            return sum;
        }
        public float this[int index]
        {
            get => data[index];
            set => data[index] = value;
        }

        public override String ToString()
        {
            var sb = new StringBuilder();
            foreach (var f in data)
                sb.Append($"{f:F3} ");
            return sb.ToString();
        }
    }
}
