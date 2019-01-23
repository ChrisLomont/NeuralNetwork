using System;

namespace NeuralNet.Model
{
    // source for (0,1) Gaussian rand numbers
    public class Gaussian
    {
        // nonzero seed to set it
        public Gaussian(Random rand = null)
        {

            if (rand == null)
                rand = new Random();
            else
                this.rand = rand;
        }

        private Random rand;

        private double z1;
        private double sigma = 1.0, mu = 0.0;
        private bool generate = false; // set to generate first pass

        public float Next()
        {
            generate = !generate; // generate every other pass
            if (!generate)
                return (float)(z1 * sigma + mu);

            double u1, u2;
            do
            {
                u1 = rand.NextDouble();
                u2 = rand.NextDouble();
            } while (u1 <= Double.MinValue);

            var z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2);
            return (float)(z0 * sigma + mu);
        }
    }
}
