using System;

namespace Lomont.NeuralNet.Model
{
    public class ActivationPair : Tuple<Func<Vector, Vector>, Func<Vector, Vector>>
    {
        public ActivationPair(Func<Vector, Vector> func, Func<Vector, Vector> dFunc) : base(func, dFunc)
        {
        }
        public Func<Vector, Vector> func => this.Item1;
        public Func<Vector, Vector> dFunc => this.Item2;
    }

    public static class ActivationFunctions
    {
        #region Activation functions

        public enum ActivationType
        {
            Identity,
            ReLU,
            Logistic,
            Softmax
        }

        public static ActivationPair Get(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Identity:
                    return new ActivationPair(Vectorize(Identity), Vectorize(Identity));
                case ActivationType.ReLU:
                    return new ActivationPair(Vectorize(ReLU), Vectorize(dReLU));
                case ActivationType.Logistic:
                    return new ActivationPair(Vectorize(Logistic), Vectorize(dLogistic));
                case ActivationType.Softmax:
                    return new ActivationPair(Softmax, dSoftmax);
            }
            throw new Exception("Unknown activation type " + type);
        }


        // Rectified Linear Unit
        public static float ReLU(float x)
        {
            return Math.Max(0, x);
        }
        // Derivative of rectified linear unit
        public static float dReLU(float x)
        {
            return x < 0 ? 0 : 1;
        }

        // Softmax function
        public static Vector Softmax(Vector x)
        {
            // Note often overflows - subtract max from each
            var xm = x[0];
            for (var i = 1; i < x.Size; ++i)
                xm = Math.Max(x[i], xm);


            var ans = new Vector(x.Size);
            var sum = 0.0f;
            for (var i = 0; i < x.Size; ++i)
            {
                ans[i] = (float)Math.Exp(x[i] - xm);
                sum += ans[i];
            }
            return ans * (1.0f / sum);
        }
        // Derivative of Softmax function
        public static Vector dSoftmax(Vector x)
        {
            throw new Exception("Softmax derivative not implemented");
            // todo - use this with a different cost function, not the squares one
            // todo - tie these together: final activation func, derivative, and cost function
            // squares cost ok with logistic final
            var s = Softmax(x);
            for (var j = 0; j < x.Size; ++j)
            { // DjSi = Si(dij - Sj)

            }

        }

        // Logistic function
        public static float Logistic(float x)
        {
            return (float)(1.0 / (1 + Math.Exp(-x)));
        }
        // Derivative of Logistic function
        public static float dLogistic(float x)
        {
            return Logistic(x) * (1 - Logistic(x));
        }

        // Identity function f(x)=x
        public static float Identity(float x)
        {
            return x;
        }
        // Derivative of identity function f(x)=x
        public static float dIdentity(float x)
        {
            return 1;
        }

        #endregion

        #region Utility
        // turn a function on a single variable to one operating on vectors
        public static Func<Vector, Vector> Vectorize(Func<float, float> func)
        {
            Func<Vector, Vector> vFunc = v =>
            {
                var v2 = new Vector(v.Size);
                for (var i = 0; i < v.Size; ++i)
                    v2[i] = func(v[i]);
                return v2;
            };
            return vFunc;
        }
        #endregion

    }
}
