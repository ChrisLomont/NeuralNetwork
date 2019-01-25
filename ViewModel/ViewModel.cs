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
            SelectedExperiment = Experiments[1];
        }
        public RelayCommand StartCommand { get;  }
        public RelayCommand ShowDataCommand { get; }

        private Experiment selectedExperiment = null;
        public Experiment SelectedExperiment
        {
            get => selectedExperiment;
            set
            {
                Set<Experiment>(() => this.SelectedExperiment, ref selectedExperiment, value);
            }
        }


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

        private int randSeed = 12345;
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
            TrainExperiment();
        }

        void ShowData()
        {
            Results.Clear();
            for (var ti = 0; ti < dataSet.TestSet.Count; ++ti)
            {
                var p = dataSet.TestSet[ti];

                var result =  selectedExperiment.HowToVisualize(ti,p.input, p.output, neuralNet.FeedForward(p.input));

                Results.Add(result);
            }
        }

        public ObservableCollection<Result> Results { get;  } = new ObservableCollection<Result>();

        public class Result
        {
            public WriteableBitmap bmp { get; set; }
            public string Text { get; set; }
            public Brush Background { get; set; }
        }


        SimpleNeuralNet neuralNet;
        DataSet dataSet;

        /* An experiment consists of how to get data and how to visualize
         *
         *
         *
         *
         */


        public class Experiment
        {
            public string Name { get; set; }
            public Func<DataSet> GetData { get; set; }
            // given data point index, input point, desired output point, computed output point, create a result to visualize
            public Func<int, Vector, Vector, Vector, Result> HowToVisualize { get; set; }
        };

        public ObservableCollection<Experiment> Experiments { get; set;  } = new ObservableCollection<Experiment>
        {
            /* Testing set to debug network
             *
             *
             */
            new Experiment {Name="Testing", GetData = ()=>DataSet.LoadSimple(1000,200,10), HowToVisualize = (ti,inputPt,outputPt,computedPt) =>
                {
                var computed = Trainer.MaxIndex(computedPt);
                var desired = Trainer.MaxIndex(outputPt);

            return new Result
            {
            Text = $"{ti}: truth = {desired}, net obtained {computed}, {outputPt} ~ {computedPt}",
            Background = desired == computed? new SolidColorBrush(Colors.LightGreen) : new SolidColorBrush(Colors.LightPink)
            };
            }},


        /* MNIST digit recognition 
         * try parameters from 
         * http://neuralnetworksanddeeplearning.com/chap1.html
         *
         * Sigmoid neurons
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
        new Experiment{Name = "MNIST", GetData = ()=>DataSet.LoadMNIST(@"..\..\MNIST"),
                HowToVisualize = (ti,inputPt,outputPt,computedPt) =>
                {
                    var computed = Trainer.MaxIndex(computedPt);
                    var desired = Trainer.MaxIndex(outputPt);

                    var pixels = new byte[28 * 28 * 3];
                    for (var i = 0; i < 28; ++i)
                    for (var j = 0; j < 28; ++j)
                    {
                        var index = i + j * 28;
                        var c = (byte)(255 - 255 * inputPt[index]);
                        index *= 3;
                        pixels[index++] = c;
                        pixels[index++] = c;
                        pixels[index] = c;
                    }

                    var wb = new WriteableBitmap(28, 28, 96.0, 96.0, PixelFormats.Rgb24, null);
                    wb.WritePixels(new Int32Rect(0, 0, 28, 28), pixels, 28 * 3, 0);

                    return new Result
                    {
                        Text = $"{ti}: truth = {desired}, net obtained {computed}",
                        Background = desired == computed ? new SolidColorBrush(Colors.LightGreen) : new SolidColorBrush(Colors.LightPink),
                        bmp = wb
                    };
                }}
        };


        void TrainExperiment()
        {

            // make repeatable
            var rand = new Random(RandSeed);

            dataSet = selectedExperiment.GetData();

            neuralNet = new SimpleNeuralNet
            {
                Random = rand
            };
            var p1 = dataSet.TrainingSet[0];

            neuralNet.Create(p1.input.Size, hiddenLayerSize, p1.output.Size);

            var trainer = new Trainer();

            Task.Factory.StartNew(
                () =>
                {
                    // want 1-10 per epoch
                    var trainingLength = dataSet.TrainingSet.Count;
                    var batches = (trainingLength + miniBatchSize - 1) / miniBatchSize;
                    var printFreq = Math.Max(batches / 10, 1);
                    try
                    {
                        Log("Training started....");
                        trainer.Train(
                            neuralNet,
                            dataSet,
                            rand,
                            result =>
                            {
                                if ((result.batchIndex % printFreq) == 0)
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


    }
}
