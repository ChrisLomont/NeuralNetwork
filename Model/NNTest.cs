using static NeuralNet.Model.ActivationFunctions;

namespace NeuralNet.Model
{
    // Test the NN
    static class NNTest
    {
        // test sizes to make test algorithms stay in index bounds
        public static void TestSizes()
        {
            var nn = new SimpleNeuralNet();
            nn.Create(2, 3, 5, 1, 4, 3);
            var output = nn.FeedForward(new Vector(1, 2));
            nn.Backpropagate(0.1f, output);
        }

        // test example from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        public static void Test1()
        {
            var nn = new SimpleNeuralNet();
            nn.Create(2, 2, 2);
            nn.W[0][0, 0] = 0.15f;
            nn.W[0][0, 1] = 0.20f;
            nn.W[0][1, 0] = 0.25f;
            nn.W[0][1, 1] = 0.30f;
            nn.b[0][0] = 0.35f;
            nn.b[0][1] = 0.35f;

            nn.W[1][0, 0] = 0.40f;
            nn.W[1][0, 1] = 0.45f;
            nn.W[1][1, 0] = 0.50f;
            nn.W[1][1, 1] = 0.55f;
            nn.b[1][0] = 0.60f;
            nn.b[1][1] = 0.60f;

            nn.f[1] = Vectorize(Logistic);
            nn.df[1] = Vectorize(dLogistic);
            nn.f[2] = Vectorize(Logistic);
            nn.df[2] = Vectorize(dLogistic);

            var input = new Vector(0.05f, 0.10f);
            // should output ~ 0.75136507, 0.772928465
            var output = nn.FeedForward(input);

            // desired output
            var t1 = new Vector(0.01f, 0.99f);

            // error vec
            var evec = output - t1;

            // should be 0.29837110+
            var error = 0.5 * evec.LengthSquared();

            var learning = 0.50f;
            nn.Backpropagate(learning, evec);

            output = nn.FeedForward(input);

            // error vec
            evec = output - t1;

            // should be 0.291027924 - TODO - example did not update bias vecs!
            error = 0.5 * evec.LengthSquared();
        }

    }
}
