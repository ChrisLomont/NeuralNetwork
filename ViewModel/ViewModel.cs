using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Command;
using NeuralNet.Model;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using static NeuralNet.Model.ActivationFunctions;
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
            StartStopCommand = new RelayCommand(StartStop);
            ShowDataCommand = new RelayCommand(ShowData);
            ClearGuessCommand = new RelayCommand(ClearGuess);
            SelectedExperiment = Experiments[1];
            ClearGuess();
        }
        public RelayCommand StartStopCommand { get; }
        public RelayCommand ShowDataCommand { get; }
        public RelayCommand ClearGuessCommand { get; }

        private Experiment selectedExperiment = null;
        public Experiment SelectedExperiment
        {
            get => selectedExperiment;
            set
            {
                Set<Experiment>(() => this.SelectedExperiment, ref selectedExperiment, value);
            }
        }

        private string guessedText = "?";
        public string GuessedText
        {
            get => guessedText;
            set
            {
                Set<string>(() => this.GuessedText, ref guessedText, value);
            }
        }

        private WriteableBitmap guessedImage = null;
        public WriteableBitmap GuessedImage
        {
            get => guessedImage;
            set
            {
                Set<WriteableBitmap>(() => this.GuessedImage, ref guessedImage, value);
            }
        }


        void ClearGuess()
        {
            var pixels = new byte[28 * 28 * 3];
            for (var i = 0; i < 28 * 28 * 3; ++i)
                pixels[i] = 255;
            var img = new WriteableBitmap(28, 28, 96.0, 96.0, PixelFormats.Rgb24, null);
            img.WritePixels(new Int32Rect(0, 0, 28, 28), pixels, 28 * 3, 0);
            GuessedImage = img;
        }

        void Draw(Point last, Point point)
        {
            if (guessedImage == null)
                ClearGuess();
            var w = guessedImage.PixelWidth;
            var h = guessedImage.PixelHeight;
            byte[] pixels = new byte[w * h * 3];
            var stride = w * 3;
            guessedImage.CopyPixels(pixels, stride, 0);

            var x1 = (int)last.X;
            var y1 = (int)last.Y;
            var index = (x1 + y1 * w) * 3;
            pixels[index++] = 0;
            pixels[index++] = 0;
            pixels[index++] = 0;

            guessedImage.WritePixels(new Int32Rect(0, 0, w, h), pixels, stride, 0);

            // trigger redraw
            var temp = guessedImage;
            GuessedImage = null;
            GuessedImage = temp;
        }


        // d = 0,1,2 down, move, up
        private bool down = false;
        private Point last;
        internal void Mouse(int d, Point point)
        {
            if (d == 0)
            {
                down = true;
                last = point;
            }

            if (d == 1 && down)
            {
                // draw
                Draw(last, point);
                last = point;
            }
            if (d == 2)
            {
                // guess
                down = false;
                GuessedText = MakeGuess().ToString();
            }
        }

        int MakeGuess()
        { // from image
            if (neuralNet != null && GuessedImage != null)
            {
                Vector input = new Vector(28 * 28);
                var pixels = new byte[28 * 28 * 3];
                GuessedImage.CopyPixels(new Int32Rect(0, 0, 28, 28), pixels, 28 * 3, 0);

                for (var i = 0; i < 28 * 28; ++i)
                {
                    // grayscale, invert, 0-1
                    var c = 255 - pixels[i * 3];
                    input[i] = c / 255.0f;
                }

                var output = neuralNet.FeedForward(input);
                return Trainer.MaxIndex(output);
            }
            return -1;
        }



        private string startStopText = "Start";
        public string StartStopText
        {
            get => startStopText;
            set
            {
                Set<string>(() => this.StartStopText, ref startStopText, value);
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
        private string successRatio = "0/0";
        public string SuccessRatio
        {
            get => successRatio;
            set
            {
                Set<string>(() => this.SuccessRatio, ref successRatio, value);
            }
        }
        private bool showFailures = false;
        public bool ShowFailures
        {
            get => showFailures;
            set
            {
                if (Set<bool>(() => this.ShowFailures, ref showFailures, value))
                    ShowData();
            }
        }


        private string hiddenLayerSize = "15";
        public string HiddenLayerSize
        {
            get => hiddenLayerSize;
            set
            {
                Set<string>(() => this.HiddenLayerSize, ref hiddenLayerSize, value);
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
        void StartStop()
        {
            try
            {
                if (StartStopText == "Start")
                {
                    StartStopText = "Stop";
                    Messages.Clear();
                    TrainExperiment();
                }
                else
                {
                    StartStopText = "Start";
                    if (trainer != null)
                        trainer.Stop();
                }
            }
            catch (Exception ex)
            {
                Messages.Add("Exception: " + ex);
            }

        }

        void ShowData()
        {
            Results.Clear();
            for (var ti = 0; ti < dataSet.TestSet.Count; ++ti)
            {
                var p = dataSet.TestSet[ti];

                var result = selectedExperiment.HowToVisualize(ti, p.input, p.output, neuralNet.FeedForward(p.input));
                if (!result.success || !ShowFailures)
                    Results.Add(result);
            }

            var successes = dataSet.TestSet.Count - Results.Count(r => !r.success);
            SuccessRatio = $"{successes}/{dataSet.TestSet.Count}";
        }

        public ObservableCollection<Result> Results { get; } = new ObservableCollection<Result>();

        public class Result
        {
            public WriteableBitmap bmp { get; set; }
            public string Text { get; set; }
            public Brush Background { get; set; }
            public bool success;
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

        public ObservableCollection<Experiment> Experiments { get; set; } = new ObservableCollection<Experiment>
        {
            /* Testing set to debug network
             *
             *
             */
            new Experiment {Name="Testing", GetData = ()=>DataSet.LoadSimple(1000,200,10), HowToVisualize = (ti,inputPt,outputPt,computedPt) =>
                {
                var computed = Trainer.MaxIndex(computedPt);
                var desired = Trainer.MaxIndex(outputPt);

                    var success = desired == computed;
            return new Result
            {
            Text = $"{ti}: truth = {desired}, net obtained {computed}, {outputPt} ~ {computedPt}",
            Background = success ? new SolidColorBrush(Colors.LightGreen) : new SolidColorBrush(Colors.LightPink),
                success = success
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

                    var success = desired == computed;
                    return new Result
                    {
                        Text = $"{ti}: truth = {desired}, net obtained {computed}",
                        Background = success ? new SolidColorBrush(Colors.LightGreen) : new SolidColorBrush(Colors.LightPink),
                        bmp = wb,
                        success = success
                    };
                }}
        };

        void AddLayers(List<int> layers, string layerText)
        {
            var words = layerText.Split(new char[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);
            foreach (var w in words)
            {
                if (!Int16.TryParse(w, out var val))
                    throw new Exception("Invalid layers descriptor " + w);
                layers.Add(val);
            }
        }

        private Trainer trainer;


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

            var layers = new List<int>();
            layers.Add(p1.input.Size);
            AddLayers(layers, hiddenLayerSize);
            layers.Add(p1.output.Size);

            neuralNet.Create(layers.ToArray());

            trainer = new Trainer();

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
            net.W[1][0, 1] = 1;
            net.b[1][0] = 7;

            net.f[1] = Vectorize(Identity);
            net.f[2] = Vectorize(Identity);
            net.df[1] = Vectorize(dIdentity);
            net.df[2] = Vectorize(dIdentity);


            var output = net.FeedForward(new Vector(3.0f, 5.0f)); // should answer 16
            Messages.Add($"Test: 16={output}");

            net.Backpropagate(0.1f, output); // train towards zero vector
            output = net.FeedForward(new Vector(3.0f, 5.0f)); // should answer 16
            Messages.Add($"Test: 16={output}");
        }

        public ObservableCollection<string> Messages { get; } = new ObservableCollection<string>();


    }
}
