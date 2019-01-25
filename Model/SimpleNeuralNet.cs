// #define IN_PLACE // define for faster, in-place work TODO - speed up
using System;
using System.Diagnostics;
using static NeuralNet.Model.ActivationFunctions;

namespace NeuralNet.Model
{
    /// <summary>
    /// Usage:
    /// 1. Make NN
    /// 2. Zero deltas
    /// 3. For each datum, feed forward, then back propagate to deltas
    /// 4. Apply deltas, updates weights
    /// 5. Repeat 
    /// 
    /// 
    /// </summary>
    public sealed class SimpleNeuralNet
    {
        // create with given sizes
        // fills in network, populates weights and bias with (0,1) gaussian 
        public SimpleNeuralNet(params int[] layers)
        {
            if (layers != null && layers.Length > 1)
                Create(layers);
        }

        // Compute the output for the given input vector
        public Vector FeedForward(Vector input)
        {
            x[0] = input;
            for (var n = 0; n <= NumLayers-2; ++n)
            {
#if IN_PLACE
                Matrix.Times(z[n], W[n], x[n]);
                Vector.Add(z[n], b[n]);
                x[n + 1] = f[n + 1](z[n]);
#else
                z[n] = W[n] * x[n] + b[n];
                x[n + 1] = f[n + 1](z[n]);
#endif
            }
            return x[NumLayers - 1];
        }


        // Update weights via back propagation
        // errVector is often 'output - truth' vector
        public void Backpropagate(float learningRate, Vector errVector)
        {
            ZeroDeltas();
            BackpropagateToDeltas(errVector);
            ModifyWeights(learningRate,1);
        }


        // Update weights via back propagation
        // errVector is often 'output - truth' vector
        public void BackpropagateToDeltas(Vector errVector)
        {
            for (var n = NumLayers-2; n >= 0; --n)
            {
                // 1. Initial eps vector
                if (n == NumLayers-2)
                    eps[n] = Vector.ComponentTimes(errVector, df[n+1](z[n]));
                else // 2. next eps vector
                    eps[n] = Vector.ComponentTimes(eps[n+1]*W[n+1], df[n+1](z[n]));

                for (var i = 0; i < LayerNodeCounts[n + 1]; ++i)
                {
                    // 3. update bias
                    db[n][i] += eps[n][i];
                    // 4. update weight matrix
                    for (var j = 0; j < LayerNodeCounts[n]; ++j)
                        dW[n][i, j] += x[n][j] * eps[n][i];
                }
            }
        }

        // zero gradient deltas
        // call when starting a new (mini) batch
        // then call back propagate for each sample, then when done, call ModifyWeights
        public void ZeroDeltas()
        {
            for (var n = NumLayers - 2; n >= 0; --n)
            for (var i = 0; i < LayerNodeCounts[n + 1]; ++i)
            {
                db[n][i] = 0;
                for (var j = 0; j < LayerNodeCounts[n]; ++j)
                    dW[n][i, j] = 0;
            }
        }

        // call at end of minibatch
        public void ModifyWeights(float learningRate, int numSamples)
        {
            learningRate /= numSamples;
            for (var n = NumLayers - 2; n >= 0; --n)
            {
                for (var i = 0; i < LayerNodeCounts[n + 1]; ++i)
                {
                    // 3. update bias
                    b[n][i] -= learningRate * db[n][i];
                    // 4. update weight matrix
                    for (var j = 0; j < LayerNodeCounts[n]; ++j)
                        W[n][i, j] -= learningRate * dW[n][i, j];
                }
            }
        }


        // create with given sizes
        // fills in network, populates weights and bias with (0,1) gaussian 
        public void Create(params int[] layers)
        {
            NumLayers = layers.Length;
            if (NumLayers <= 1)
                throw new ArgumentException("Not enough layers","layers");
            LayerNodeCounts = new int[NumLayers];
            Array.Copy(layers, LayerNodeCounts, NumLayers);
            W = new Matrix[NumLayers - 1];
            b = new Vector[NumLayers - 1];
            dW = new Matrix[NumLayers - 1];
            db = new Vector[NumLayers - 1];
            x = new Vector[NumLayers];
            z = new Vector[NumLayers - 1];
            eps = new Vector[NumLayers-1];
            f = new Func<Vector, Vector>[NumLayers];  // entry 0 unused
            df = new Func<Vector, Vector>[NumLayers]; // entry unused

            var rand = new Gaussian(Random);
            for (var n = 0; n < NumLayers; ++n)
            {
                if (n != NumLayers - 1)
                {
                    W[n] = new Matrix();
                    b[n] = new Vector(layers[n + 1]);
                    W[n].Resize(layers[n + 1], layers[n]);
                    dW[n] = new Matrix();
                    db[n] = new Vector(layers[n + 1]);
                    dW[n].Resize(layers[n + 1], layers[n]);
#if IN_PLACE
                    z[n] = new Vector(layers[n+1]);
#endif

                    // random filling weights
                    W[n].Randomize(rand.Next);
                    b[n].Randomize(rand.Next);
                }
#if IN_PLACE
                x[n] = new Vector(layers[n]);
#endif


                // set up default activation functions
                // layer 0 identity
                // layers 1 to LayerCount-2 (inclusive) are ReLU
                // layer LayerCount-1 is softmax
                if (0 == n)
                {
                    f[n] = Vectorize(Identity);
                    df[n] = Vectorize(dIdentity);
                }
                else if (1 <= n && n <= NumLayers-2)
                {
                    f[n] = Vectorize(Logistic);// Vectorize(ReLU);
                    df[n] = Vectorize(dLogistic);//Vectorize(dReLU);
                }
                else if (n == NumLayers - 1)
                {
                    f[n] = Vectorize(Logistic);//Vectorize(ReLU); // Softmax;
                    df[n] = Vectorize(dLogistic);//Vectorize(dReLU); // dSoftmax;
                }
            }
        }

        // zero all weights. Useful for testing
        public void ZeroAll()
        {
            for (var n = 0; n < NumLayers-1; ++n)
            {
                for (var i = 0; i < LayerNodeCounts[n + 1]; ++i)
                {
                    Debug.Assert(!float.IsNaN(b[n][i]));
                    b[n][i] = 0;
                    for (var j = 0; j < LayerNodeCounts[n]; ++j)
                    {
                        Debug.Assert(!float.IsNaN(W[n][i,j]));
                        W[n][i, j] = 0;
                    }
                }
            }
        }

        // get name of variable and worst value
        // worst value is largest in abs value, or NaN or Inf if exists
        public (string name, float worstValue) Bounds()
        {
            var name = "";
            var val = 0.0f;
            var isBad = false;
            WalkValues((f, s) =>
            {
                if (Single.IsNaN(f) || Single.IsInfinity(f))
                {
                    isBad = true;
                    name = s;
                    val = f;
                }
                if (!isBad)
                {
                    var a = Math.Abs(f);
                    if (a > val)
                    {
                        val = a;
                        name = s;
                    }
                }
            });
            return (name,val);
        }

        /// <summary>
        /// Check internals for large float values, NaN, etc
        /// </summary>
        public void Check()
        {
            WalkValues((f, s) =>
            {
                if (Single.IsNaN(f) || Single.IsInfinity(f))
                    Console.WriteLine("bad float found at " + s);
            });
        }

        void WalkValues(Action<float,string> action)
        {
            Action<Vector,string> TestV = (v,s) =>
            {
                if (v == null) return;
                for (var i = 0; i < v.Size; ++i)
                    action(v[i], s);
            };
            Action<Matrix,string> TestM = (m,s) =>
            {
                if (m == null) return;
                for (var i = 0; i < m.Rows; ++i)
                for (var j = 0; j < m.Columns; ++j)
                    action(m[i,j],s);
            };

            for (var n = 0; n < NumLayers - 1; ++n)
            {
                TestV(b[n],$"b[{n}]");
                TestM(W[n],$"W[{n}]");
                TestV(z[n],$"z[{n}]");
                TestV(x[n],$"x[{n}]");
                TestV(eps[n],"eps[{n}]");
            }

            TestV(x[NumLayers-1],$"x[{NumLayers-1}]");
        }

        #region Data

        // if left alone, uses random seed
        public Random Random = null;

        // number of layers, 1+
        public int NumLayers;
        // sizes of each node count
        public int[] LayerNodeCounts;

        // weights, entry n maps inputs from layer n towards layer n+1
        public Matrix[] W;
        // bias
        public Vector[] b;

        // delta weights, entry n maps inputs from layer n towards layer n+1
        public Matrix[] dW;
        // delta bias
        public Vector[] db;

        // layer functions
        public Func<Vector, Vector>[] f;
        // layer function differentials
        public Func<Vector, Vector>[] df;

        // space for back propagation
        public Vector[] eps;

        // Intermediate values Wx+b
        public Vector[] z;
        // Intermediates f(z)
        public Vector[] x; 


        #endregion


    }
}
