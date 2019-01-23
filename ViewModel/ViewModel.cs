using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Command;
using NeuralNet.Model;
using static NeuralNet.Model.ActivationFunctions;
using Matrix = NeuralNet.Model.Matrix;
using Vector = NeuralNet.Model.Vector;

/* TODO
 * 1. Save/load NN state between runs, and training info (how it got here)
 * 2. Multithreaded for perf
 * 3. Plot training and test curves to show over fit, etc.
 *
 */

namespace NeuralNet.ViewModel
{
    class ViewModel : ViewModelBase
    {
        public Dispatcher Dispatcher;
        public ViewModel()
        {
            StartCommand = new RelayCommand(Start);
            ShowDataCommand = new RelayCommand(ShowData);
            //NNTest.TestSizes();
            //Test();
            //NNTest.Test1();
            //LearnFixed(0.1f);
            //LearnFixed(0.01f);
            //LearnLinear(0.01f);
            //TrainMNIST();
        }
        public RelayCommand StartCommand { get;  }
        public RelayCommand ShowDataCommand { get; }


        private float learningRate = 3.0f;
        public float LearningRate
        {
            get => learningRate;
            set
            {
                Set<float>(() => this.LearningRate, ref learningRate, value);
            }
        }

        private int hiddenLayerSize = 15;
        public int HiddenLayerSize
        {
            get => hiddenLayerSize;
            set
            {
                Set<int>(() => this.HiddenLayerSize, ref hiddenLayerSize, value);
            }
        }

        private int miniBatchSize = 10;
        public int MiniBatchSize
        {
            get => miniBatchSize;
            set
            {
                Set<int>(() => this.MiniBatchSize, ref miniBatchSize, value);
            }
        }

        private int randSeed = 30;
        public int RandSeed
        {
            get => randSeed;
            set
            {
                Set<int>(() => this.RandSeed, ref randSeed, value);
            }
        }

        private int epochs = 30;
        public int Epochs
        {
            get => epochs;
            set
            {
                Set<int>(() => this.Epochs, ref epochs, value);
            }
        }
        void Start()
        {
            Messages.Clear();
            TrainMNIST();
        }

        void ShowData()
        {
            Results.Clear();
            for (var ti = 0; ti < dataSet.TestSet.Count; ++ti)
            {
                var p = dataSet.TestSet[ti];
                var computed = Trainer.MaxIndex(neuralNet.FeedForward(p.input));
                var desired = Trainer.MaxIndex(p.output);

                var pixels = new byte[28 * 28 * 3];
                for (var i = 0; i < 28; ++i)
                for (var j = 0; j < 28; ++j)
                {
                    var index = i + j * 28;
                    var c = (byte)(255-255*p.input[index]);
                    index *= 3;
                    pixels[index++] = c;
                    pixels[index++] = c;
                    pixels[index  ] = c;
                }

                var wb = new WriteableBitmap(28,28,96.0,96.0,PixelFormats.Rgb24,null);
                wb.WritePixels(new Int32Rect(0,0,28,28), pixels, 28*3, 0);

                Results.Add(new Result
                {
                  Text = $"{ti}: truth = {desired}, net obtained {computed}",
                  Background = desired == computed? new SolidColorBrush(Colors.LightGreen): new SolidColorBrush(Colors.LightPink),
                  bmp =  wb
                });
            }
        }

        public ObservableCollection<Result> Results { get;  } = new ObservableCollection<Result>();

        public class Result
        {
            public WriteableBitmap bmp { get; set; }
            public string Text { get; set; }
            public Brush Background { get; set; }
        }


        Trainer trainer;
        SimpleNeuralNet neuralNet;
        DataSet dataSet;
        void TrainMNIST()
        {
            // try parameters from 
            // http://neuralnetworksanddeeplearning.com/chap1.html

            /* Sigmoid neurons
             * three layer network: 784 = 28x28, 15, 10 = digits 0-9, highest activation value wins
             * Input 0.0 = white, 1.0 = black pixels
             * gaussian normal (0,1) weight initialization
             * cost function = 1/2n sum over vector diff squared |y(x)-a|^2
             * Stochastic Gradient Descent:
             *    pick random m inputs, called mini-batch, estimate gradient as avg of per item cost gradients
             *    repeat, new random points, until all training items done, called epoch
             * minibatch size 10
             * training size 60,000
             * learning rate = 3.0
             * train 30 epochs
             * Should have ~95% success on test
             */

            // make repeatable
            var rand = new Random(1234);

            neuralNet = new SimpleNeuralNet
            {
                Random = rand
            };
            neuralNet.Create(28 * 28, hiddenLayerSize, 10);

            dataSet = DataSet.LoadMNIST(@"..\..\MNIST");

            trainer = new Trainer();

            Task.Factory.StartNew(
                () =>
                {
                    int pass = 0;
                    try
                    {
                        Log("Training started....");
                        trainer.Train(
                            neuralNet,
                            dataSet,
                            rand,
                            result =>
                            {
                                if ((result.batchIndex % 1000) == 0)
                                    Log(result.ToString());
                            },
                            Epochs,
                            MiniBatchSize,
                            LearningRate
                        );
                        Log("Training done.");
                    }
                    catch (Exception ex)
                    {
                        Log(ex.ToString());
                    }
                }
            );
        }

        // log from any thread
        void Log(string msg)
        {
            Dispatch(() => Messages.Add(msg));
        }

        void Dispatch(Action action)
        {
            Dispatcher.Invoke(action);
        }

        void Test()
        {
            var net = new SimpleNeuralNet();
            net.Random = new Random(1234); // repeatable
            // make a simple net
            net.Create(2, 2, 1);
            net.W[0][0, 0] = 1;
            net.W[0][1, 0] = 2;
            net.W[0][0, 1] = 3;
            net.W[0][1, 1] = 4;
            net.b[0][0] = 5;
            net.b[0][1] = 6;

            net.W[1][0, 0] = -1;
            net.W[1][0, 1] =  1;
            net.b[1][0] = 7;

            net.f[1] = Vectorize(Identity);
            net.f[2] = Vectorize(Identity);
            net.df[1] = Vectorize(dIdentity);
            net.df[2] = Vectorize(dIdentity);


            var output = net.FeedForward(new Vector(3.0f, 5.0f)); // should answer 16
            Messages.Add($"Test: 16={output}");

            net.Backpropagate(0.1f,output); // train towards zero vector
            output = net.FeedForward(new Vector(3.0f, 5.0f)); // should answer 16
            Messages.Add($"Test: 16={output}");

        }

        public ObservableCollection<string> Messages { get;  } = new ObservableCollection<string>();

        //learning 0.1 here too wild, diverges, 0.01 works
        public void LearnFixed(float learningRate)
        {
            Messages.Add($"Learn fixed, rate {learningRate:F4}");
            var net = new SimpleNeuralNet();
            net.Random = new Random(1);
            net.Create(2, 2, 1);
            for (var n = 1; n < net.NumLayers; ++n)
            {
                net.f[n] = Vectorize(Identity);
                net.df[n] = Vectorize(dIdentity);
            }

            var input = new Vector(0.4f, 1.3f);
            var ans = new Vector(6.0f);
            for (var p = 0; p < 100; ++p)
            {
                var output = net.FeedForward(input);
                net.Backpropagate(learningRate, output - ans);
                var (name, val) = net.Bounds();
                Messages.Add($"{p}: {output}, {name}:{val}");
            }
        }

        public void LearnLinear(float learningRate)
        {
            Messages.Add($"Learn linear, rate {learningRate:F4}");

            // make dataset, same training and test
            var rand = new Random(1234);
            var inputSize = rand.Next(2, 5);
            var outputSize = rand.Next(2, 5);

            var trans = new Matrix();
            trans.Resize(outputSize,inputSize);
            var gauss = new Gaussian(rand);
            trans.Randomize(gauss.Next);

            var ds = new DataSet();
            for (var i = 0; i < 10; ++i)
            {
                var input = new Vector(inputSize);
                input.Randomize(gauss.Next);
                var dp = new DataPoint(input, trans*input);
                ds.TrainingSet.Add(dp);
                ds.TestSet.Add(dp);
            }

            trainer = new Trainer();

            var net = new SimpleNeuralNet();
            net.Random = new Random(1234);
            net.Create(inputSize, inputSize*outputSize, outputSize);
            for (var n = 1; n < net.NumLayers; ++n)
            {
                net.f[n] = Vectorize(Identity);
                net.df[n] = Vectorize(dIdentity);
            }

            //trainer.Train
            //    (
            //    net, ds, message:Messages.Add, singleStep:true
            //    );
        }


    }
}
